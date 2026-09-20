"""Experience recording — the structured memory of everything that happens.

Every observation, decision, action, evaluation and error of a run becomes
an :class:`ExperienceRecord` through an :class:`ExperienceSink`. This is the
foundation future learning builds on; no learning happens here yet.
Sinks are the storage seam: in-memory for tests/dry runs, PostgreSQL
(``experiences`` table) for real runs, and future vector/knowledge sinks
implement the same one-method protocol.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database.models import Experience as ExperienceORM

ExperienceKind = Literal["observation", "decision", "action", "evaluation", "error", "summary"]
ExperienceOutcome = Literal["success", "failure", "neutral"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class ExperienceRecord(BaseModel):
    """One structured experience, tied to a run and (optionally) a goal."""

    id: str = Field(default_factory=_new_id)
    run_id: str | None = None
    goal_id: str | None = None
    kind: ExperienceKind
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    outcome: ExperienceOutcome = "neutral"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)


class ExperienceSink(ABC):
    """Storage seam for experience records."""

    @abstractmethod
    async def save(self, records: Sequence[ExperienceRecord]) -> int:
        """Persist records; return the number persisted."""


class InMemoryExperienceSink(ExperienceSink):
    """Process-local sink for tests and dry runs."""

    def __init__(self) -> None:
        self.records: list[ExperienceRecord] = []

    async def save(self, records: Sequence[ExperienceRecord]) -> int:
        self.records.extend(record.model_copy(deep=True) for record in records)
        return len(records)


class DatabaseExperienceSink(ExperienceSink):
    """Persists records into the ``experiences`` table."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save(self, records: Sequence[ExperienceRecord]) -> int:
        if not records:
            return 0
        rows = [
            ExperienceORM(
                id=uuid.UUID(record.id),
                run_id=uuid.UUID(record.run_id) if record.run_id else None,
                goal_id=uuid.UUID(record.goal_id) if record.goal_id else None,
                kind=record.kind,
                summary=record.summary,
                payload=record.payload,
                outcome=record.outcome,
                confidence=record.confidence,
            )
            for record in records
        ]
        async with self._session_factory() as session:
            session.add_all(rows)
            await session.commit()
        return len(rows)


class ExperienceRecorder:
    """Buffers records and flushes them to the sink.

    The loop flushes after each stage, so a crashed run still leaves every
    completed stage recorded (durability over batching elegance for now).
    """

    def __init__(self, sink: ExperienceSink) -> None:
        self._sink = sink
        self._buffer: list[ExperienceRecord] = []

    def record(
        self,
        *,
        run_id: str | None,
        goal_id: str | None,
        kind: ExperienceKind,
        summary: str,
        payload: dict[str, Any] | None = None,
        outcome: ExperienceOutcome = "neutral",
        confidence: float | None = None,
    ) -> ExperienceRecord:
        entry = ExperienceRecord(
            run_id=run_id,
            goal_id=goal_id,
            kind=kind,
            summary=summary,
            payload=payload or {},
            outcome=outcome,
            confidence=confidence,
        )
        # Buffer a snapshot: later mutation of the returned record must not
        # change what gets persisted.
        self._buffer.append(entry.model_copy(deep=True))
        return entry

    @property
    def pending(self) -> int:
        return len(self._buffer)

    async def flush(self) -> int:
        """Persist and clear the buffer; return the number persisted."""
        if not self._buffer:
            return 0
        batch, self._buffer = self._buffer, []
        return await self._sink.save(batch)
