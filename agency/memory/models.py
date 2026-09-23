"""Structured models for Skynet's long-term memory.

Three memory types (blueprint §7.2):

- ``episodic``   — what happened: research sessions, conversations, actions,
  failures, notable events.
- ``semantic``   — learned knowledge: facts, claims, findings with sources.
- ``procedural`` — strategies and workflows that demonstrably worked.

Every memory keeps a **provenance chain** (memory → experience/observation →
source → URL/participant), so Skynet can always answer "why do I remember
this?". Memory content is data, never instructions (see ``docs/SKYNET_MEMORY.md``).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


#: Where a memory's content came from — the first hop of the provenance chain.
MemoryOrigin = Literal[
    "observation",
    "experience",
    "conversation",
    "research",
    "experiment",
    "improvement",
    "human",
    "consolidation",
    "system",
]


class MemoryRecord(BaseModel):
    """One long-term memory with full provenance."""

    id: str = Field(default_factory=_new_id)
    type: MemoryType = MemoryType.EPISODIC
    #: Bounded content body (capped at store time; huge payloads rejected).
    content: str
    #: One-line digest used by search and recall rendering.
    summary: str = ""
    #: First provenance hop, e.g. ``web:en.wikipedia.org``, ``ai:mock:agent-1``,
    #: ``experience:action``, ``human``.
    source: str = ""
    origin: MemoryOrigin = "system"
    #: Full provenance chain (arbitrary JSON: observation ids, URLs,
    #: conversation/message ids, run/goal linkage).
    provenance: dict[str, Any] = Field(default_factory=dict)
    #: 0.0–1.0 salience; candidates below the configured threshold are not
    #: stored automatically (importance mechanism, blueprint §7.6).
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    goal_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ScoredMemory(BaseModel):
    """A memory plus its relevance score for one query."""

    memory: MemoryRecord
    score: float = Field(ge=0.0, le=1.0)


class MemoryStats(BaseModel):
    """Store census (used by ``memory-stats`` and health checks)."""

    total: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
    oldest: datetime | None = None
    newest: datetime | None = None


__all__ = ["MemoryRecord", "MemoryStats", "MemoryType", "ScoredMemory"]
