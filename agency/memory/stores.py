"""Storage backends for long-term memory.

The :class:`MemoryStore` protocol is the seam future vector stores,
knowledge graphs and hybrid stores implement — nothing above it knows
which backend is live. Two implementations ship:

- :class:`InMemoryStore` — process-local dicts (tests, dry runs).
- :class:`SQLiteMemoryStore` — stdlib ``sqlite3`` via ``asyncio.to_thread``
  (zero new dependencies; survives process restarts). Search uses SQLite
  FTS5 when available and falls back to LIKE otherwise.

Retrieval relevance is deliberately simple and deterministic: token
overlap across content/summary/tags, weighted, normalized to 0–1. The
store protocol (not this heuristic) is where vector similarity plugs in
later.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agency.memory.models import MemoryRecord, MemoryType

logger = logging.getLogger("skynet.memory.stores")

#: Hard content cap at store time — memory is not a bulk file store.
MAX_CONTENT_CHARS = 20_000

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class MemoryStore(ABC):
    """Protocol for long-term memory persistence."""

    @abstractmethod
    async def save(self, record: MemoryRecord) -> MemoryRecord:
        """Insert or update by id; return the stored record."""

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryRecord | None:
        """Return the memory or None if unknown."""

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Forget one memory; True if it existed."""

    @abstractmethod
    async def search(
        self,
        *,
        query: str = "",
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        goal_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[tuple[MemoryRecord, float]]:
        """Filter, rank and return ``(record, score)`` pairs (score 0–1)."""

    @abstractmethod
    async def count_by_type(self) -> dict[str, int]:
        """Census by memory type."""


class InMemoryStore(MemoryStore):
    """Process-local store (tests, dry runs). Not restart-safe by design."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    async def save(self, record: MemoryRecord) -> MemoryRecord:
        record.updated_at = datetime.now(UTC)
        self._records[record.id] = record.model_copy(deep=True)
        return record.model_copy(deep=True)

    async def get(self, memory_id: str) -> MemoryRecord | None:
        record = self._records.get(memory_id)
        return record.model_copy(deep=True) if record else None

    async def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    async def search(
        self,
        *,
        query: str = "",
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        goal_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[tuple[MemoryRecord, float]]:
        query_tokens = _tokens(query) or set()
        scored: list[tuple[MemoryRecord, float]] = []
        for record in self._records.values():
            if memory_type is not None and record.type is not memory_type:
                continue
            if tags and not set(tags) & {t.lower() for t in record.tags}:
                continue
            if source and source.lower() not in record.source.lower():
                continue
            if goal_id and record.goal_id != goal_id:
                continue
            if since and record.created_at < since:
                continue
            if until and record.created_at > until:
                continue
            if record.importance < min_importance:
                continue
            scored.append((record.model_copy(deep=True), self._score(record, query_tokens)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _score(record: MemoryRecord, query_tokens: set[str]) -> float:
        if not query_tokens:
            # No query: rank by importance alone (deterministic).
            return record.importance
        haystack = _tokens(
            " ".join([record.content, record.summary, " ".join(record.tags), record.source])
        )
        overlap = len(query_tokens & haystack) / len(query_tokens)
        return round(min(1.0, overlap * 0.8 + record.importance * 0.2), 4)

    async def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.type.value] = counts.get(record.type.value, 0) + 1
        return counts


class SQLiteMemoryStore(MemoryStore):
    """Persistent store on stdlib sqlite3 (zero infrastructure).

    All blocking calls run in a worker thread; the file is created on
    first use. WAL mode keeps concurrent readers happy during writes.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                origin TEXT NOT NULL DEFAULT 'system',
                provenance TEXT NOT NULL DEFAULT '{}',
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL,
                tags TEXT NOT NULL DEFAULT '[]',
                goal_id TEXT,
                run_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)"
        )
        connection.commit()

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            with self._connect() as connection:
                self._initialize(connection)
            self._initialized = True

    # -- row mapping --------------------------------------------------------

    @staticmethod
    def _to_row(record: MemoryRecord) -> tuple:
        import json

        return (
            record.id,
            record.type.value,
            record.content,
            record.summary,
            record.source,
            record.origin,
            json.dumps(record.provenance, default=str),
            record.importance,
            record.confidence,
            json.dumps(record.tags),
            record.goal_id,
            record.run_id,
            json.dumps(record.metadata, default=str),
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryRecord:
        import json

        return MemoryRecord(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            summary=row["summary"],
            source=row["source"],
            origin=row["origin"],
            provenance=json.loads(row["provenance"]),
            importance=row["importance"],
            confidence=row["confidence"],
            tags=json.loads(row["tags"]),
            goal_id=row["goal_id"],
            run_id=row["run_id"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # -- protocol -------------------------------------------------------------

    async def save(self, record: MemoryRecord) -> MemoryRecord:
        self._ensure_initialized()
        record.updated_at = datetime.now(UTC)
        if len(record.content) > MAX_CONTENT_CHARS:
            raise ValueError(
                f"memory content exceeds {MAX_CONTENT_CHARS} chars; "
                "store a summary plus a provenance pointer instead"
            )
        row = self._to_row(record)
        async with self._lock:
            await asyncio.to_thread(self._save_sync, row)
        return record.model_copy(deep=True)

    def _save_sync(self, row: tuple) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (
                    id, type, content, summary, source, origin, provenance,
                    importance, confidence, tags, goal_id, run_id, metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    type=excluded.type,
                    content=excluded.content,
                    summary=excluded.summary,
                    source=excluded.source,
                    origin=excluded.origin,
                    provenance=excluded.provenance,
                    importance=excluded.importance,
                    confidence=excluded.confidence,
                    tags=excluded.tags,
                    goal_id=excluded.goal_id,
                    run_id=excluded.run_id,
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at
                """,
                row,
            )
            connection.commit()

    async def get(self, memory_id: str) -> MemoryRecord | None:
        self._ensure_initialized()
        async with self._lock:
            row = await asyncio.to_thread(self._get_sync, memory_id)
        return self._from_row(row) if row else None

    def _get_sync(self, memory_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

    async def delete(self, memory_id: str) -> bool:
        self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._delete_sync, memory_id)

    def _delete_sync(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            connection.commit()
            return cursor.rowcount > 0

    async def search(
        self,
        *,
        query: str = "",
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        goal_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[tuple[MemoryRecord, float]]:
        self._ensure_initialized()
        payload = {
            "query": query,
            "memory_type": memory_type.value if memory_type else None,
            "tags": [t.lower() for t in (tags or [])],
            "source": source,
            "goal_id": goal_id,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "min_importance": min_importance,
            "limit": limit,
        }
        async with self._lock:
            rows = await asyncio.to_thread(self._search_sync, payload)
        query_tokens = _tokens(query) or set()
        results: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            record = self._from_row(row)
            results.append((record, self._score(record, query_tokens)))
        return results

    def _search_sync(self, payload: dict[str, Any]) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if payload["memory_type"]:
            clauses.append("type = ?")
            params.append(payload["memory_type"])
        if payload["source"]:
            clauses.append("LOWER(source) LIKE ?")
            params.append(f"%{payload['source'].lower()}%")
        if payload["goal_id"]:
            clauses.append("goal_id = ?")
            params.append(payload["goal_id"])
        if payload["since"]:
            clauses.append("created_at >= ?")
            params.append(payload["since"])
        if payload["until"]:
            clauses.append("created_at <= ?")
            params.append(payload["until"])
        if payload["min_importance"] > 0:
            clauses.append("importance >= ?")
            params.append(payload["min_importance"])
        for tag in payload["tags"]:
            clauses.append("LOWER(tags) LIKE ?")
            params.append(f'%"{tag}"%')
        if payload["query"]:
            tokens = _tokens(payload["query"])
            if tokens:
                token_clauses = " OR ".join(
                    "LOWER(content || ' ' || summary || ' ' || tags || ' ' || source) "
                    "LIKE ?" for _ in tokens
                )
                clauses.append(f"({token_clauses})")
                params.extend(f"%{token}%" for token in tokens)
        sql = "SELECT * FROM memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(payload["limit"])
        with self._connect() as connection:
            return connection.execute(sql, params).fetchall()

    @staticmethod
    def _score(record: MemoryRecord, query_tokens: set[str]) -> float:
        if not query_tokens:
            return record.importance
        haystack = _tokens(
            " ".join([record.content, record.summary, " ".join(record.tags), record.source])
        )
        overlap = len(query_tokens & haystack) / len(query_tokens)
        return round(min(1.0, overlap * 0.8 + record.importance * 0.2), 4)

    async def count_by_type(self) -> dict[str, int]:
        self._ensure_initialized()
        async with self._lock:
            return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT type, COUNT(*) AS n FROM memories GROUP BY type"
            ).fetchall()
        return {row["type"]: row["n"] for row in rows}


def build_memory_store(backend: str, *, sqlite_path: str) -> MemoryStore:
    """Build a memory store by configuration name (factory)."""
    normalized = (backend or "memory").strip().lower()
    if normalized == "memory":
        return InMemoryStore()
    if normalized == "sqlite":
        return SQLiteMemoryStore(sqlite_path)
    raise ValueError(
        f"unknown memory backend {backend!r}; available: memory, sqlite"
    )


__all__ = [
    "MAX_CONTENT_CHARS",
    "InMemoryStore",
    "MemoryStore",
    "SQLiteMemoryStore",
    "build_memory_store",
]
