"""MemoryManager — the single doorway to Skynet's long-term memory.

Owns importance gating (candidates below ``memory_importance_threshold``
are not stored automatically), recall budgeting (recall never dumps the
store into a prompt; a character budget plus ``max_results`` bound every
recall) and usage logging. Storage backends live behind
:class:`agency.memory.stores.MemoryStore` — this class never touches SQL
or dicts directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from agency.memory.models import MemoryRecord, MemoryStats, MemoryType, ScoredMemory
from agency.memory.stores import MemoryStore

#: Search results are (record, relevance-score) pairs.
ScoredPair = tuple[MemoryRecord, float]

_ALLOWED_ORIGINS = frozenset(
    {
        "observation",
        "experience",
        "conversation",
        "research",
        "human",
        "consolidation",
        "system",
    }
)


def _normalize_origin(origin: str) -> str:
    """Map unknown origin strings to ``system`` (origin is a closed set)."""
    return origin if origin in _ALLOWED_ORIGINS else "system"

logger = logging.getLogger("skynet.memory.manager")


class MemoryManager:
    """Facility operations over one store: store, recall, search, forget."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        importance_threshold: float = 0.4,
        max_results: int = 10,
        recall_max_chars: int = 4000,
    ) -> None:
        self._store = store
        self._importance_threshold = importance_threshold
        self._max_results = max_results
        self._recall_max_chars = recall_max_chars

    # -- introspection ------------------------------------------------------------

    @property
    def importance_threshold(self) -> float:
        """Minimum importance for automatic storage (read-only)."""
        return self._importance_threshold

    @property
    def max_results(self) -> int:
        """Default cap on returned memories (read-only)."""
        return self._max_results

    @property
    def recall_max_chars(self) -> int:
        """Character budget for recall injection (read-only)."""
        return self._recall_max_chars

    # -- write path -------------------------------------------------------------

    async def store(
        self,
        *,
        content: str,
        memory_type: MemoryType | str = MemoryType.EPISODIC,
        summary: str = "",
        source: str = "",
        origin: str = "system",
        provenance: dict[str, Any] | None = None,
        importance: float = 0.5,
        confidence: float | None = None,
        tags: list[str] | None = None,
        goal_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> MemoryRecord | None:
        """Store one memory, gated by the importance threshold.

        ``force=True`` bypasses the gate (operator inserts, consolidation
        commits). Returns the stored record, or ``None`` when the candidate
        fell below the importance threshold (logged, not silent: the caller
        sees the rejection).
        """
        record = MemoryRecord(
            type=MemoryType(memory_type),
            content=content,
            summary=summary,
            source=source,
            origin=_normalize_origin(origin),  # type: ignore[arg-type]
            provenance=dict(provenance or {}),
            importance=importance,
            confidence=confidence,
            tags=[t.strip().lower() for t in (tags or []) if t.strip()],
            goal_id=goal_id,
            run_id=run_id,
            metadata=dict(metadata or {}),
        )
        if not force and record.importance < self._importance_threshold:
            logger.info(
                "memory below importance threshold (%.2f < %.2f): %s",
                record.importance,
                self._importance_threshold,
                record.summary or record.content[:60],
            )
            return None
        stored = await self._store.save(record)
        logger.info(
            "memory stored: id=%s type=%s importance=%.2f origin=%s",
            stored.id,
            stored.type.value,
            stored.importance,
            stored.origin,
        )
        return stored

    # -- read path ---------------------------------------------------------------

    async def recall(
        self,
        query: str,
        *,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        max_results: int | None = None,
        max_chars: int | None = None,
    ) -> list[MemoryRecord]:
        """Retrieve relevant memories, budgeted for prompt injection.

        This is the "Memory Recall" stage: a bounded slice of relevant
        memory, never the whole store. Returns ``ScoredMemory.memory``
        sorted by relevance.
        """
        results = await self._store.search(
            query=query,
            memory_type=MemoryType(memory_type) if memory_type else None,
            tags=tags,
            limit=max_results or self._max_results,
        )
        budget = max_chars or self._recall_max_chars
        chosen: list[MemoryRecord] = []
        used = 0
        for record, _score in results:
            size = len(record.content) + len(record.summary)
            if used + size > budget and chosen:
                break
            chosen.append(record)
            used += size
        return chosen

    async def search(
        self,
        *,
        query: str = "",
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        goal_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        min_importance: float = 0.0,
        limit: int | None = None,
    ) -> list[ScoredPair]:
        """Full-fidelity search returning ``(record, score)`` pairs."""
        results = await self._store.search(
            query=query,
            memory_type=MemoryType(memory_type) if memory_type else None,
            tags=tags,
            source=source,
            goal_id=goal_id,
            since=since,
            until=until,
            min_importance=min_importance,
            limit=limit or self._max_results,
        )
        return results

    async def get(self, memory_id: str) -> MemoryRecord | None:
        return await self._store.get(memory_id)

    async def forget(self, memory_id: str) -> bool:
        """Forget one memory (explicit deletion; the only delete path)."""
        deleted = await self._store.delete(memory_id)
        if deleted:
            logger.info("memory forgotten: id=%s", memory_id)
        return deleted

    async def stats(self) -> MemoryStats:
        by_type = await self._store.count_by_type()
        total = sum(by_type.values())
        return MemoryStats(total=total, by_type=by_type)


__all__ = ["MemoryManager", "MemoryStats", "ScoredMemory", "ScoredPair"]
