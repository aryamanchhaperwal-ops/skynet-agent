"""Minimal storage abstraction for the Skynet core.

Scope discipline: this phase persists only **goals** and **runs**; trace
events persist via :class:`agency.trace.DatabaseTraceSink` and experiences
via :class:`agency.experience.DatabaseExperienceSink` (one seam per
aggregate, all following the same async pattern). Conversations, knowledge,
experiments and improvements will get their own seams in later phases —
this module will not become a god object.

Two backends implement the same protocol:

- :class:`MemoryStorage` — in-process dicts (tests, dry runs).
- :class:`DatabaseStorage` — the existing async SQLAlchemy stack
  (``database.session``), mapping to the additive ORM models in
  ``database.models``.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agency.goals import Goal
from database.base import Base
from database.models import Experience as ExperienceORM
from database.models import Goal as GoalORM
from database.models import Run as RunORM
from database.models import RunEvent as RunEventORM
from database.session import get_engine


def _utcnow() -> datetime:
    return datetime.now(UTC)


#: Tables owned by the Skynet core; ``ensure_core_schema`` creates exactly
#: these and nothing else (the God's Eye ``events`` table is untouched).
CORE_TABLES: tuple[Any, ...] = (
    GoalORM.__table__,
    RunORM.__table__,
    RunEventORM.__table__,
    ExperienceORM.__table__,
)


async def ensure_core_schema() -> None:
    """Idempotently create the Skynet core tables (goals, runs, run_events,
    experiences). Explicitly restricted to ``CORE_TABLES``; never touches
    the God's Eye schema. Alembic migrations supersede this in a later
    phase — until then this keeps first runs working with zero ceremony.
    """

    def _create(sync_connection: Any) -> None:
        Base.metadata.create_all(bind=sync_connection, tables=list(CORE_TABLES))

    async with get_engine().begin() as connection:
        await connection.run_sync(_create)


class RunRecord:
    """In-memory run descriptor (persisted via ``Storage.create_run``).

    Kept as a small dataclass rather than pydantic: it is loop-internal
    bookkeeping; serialization responsibility lives with the run row and
    the run summary.
    """

    __slots__ = (
        "error",
        "finished_at",
        "goal_id",
        "id",
        "perception",
        "planner",
        "result",
        "started_at",
        "status",
        "steps_executed",
        "steps_planned",
    )

    def __init__(
        self,
        *,
        id: str,
        goal_id: str | None,
        status: str,
        planner: str,
        perception: str,
        started_at: datetime | None = None,
    ) -> None:
        self.id = id
        self.goal_id = goal_id
        self.status = status
        self.planner = planner
        self.perception = perception
        self.steps_planned: list[dict[str, Any]] = []
        self.steps_executed = 0
        self.result: dict[str, Any] = {}
        self.error: str | None = None
        self.started_at = started_at or _utcnow()
        self.finished_at: datetime | None = None


class Storage(ABC):
    """Persistence protocol for goals and runs."""

    # -- Goals ---------------------------------------------------------------

    @abstractmethod
    async def create_goal(self, goal: Goal) -> Goal:
        """Persist a new goal and return it with storage timestamps."""

    @abstractmethod
    async def get_goal(self, goal_id: str) -> Goal | None:
        """Return the goal or None if unknown."""

    @abstractmethod
    async def update_goal_status(self, goal_id: str, status: str) -> Goal | None:
        """Update the goal status; return the updated goal or None."""

    # -- Runs ------------------------------------------------------------------

    @abstractmethod
    async def create_run(self, run: RunRecord) -> RunRecord:
        """Persist a new run."""

    @abstractmethod
    async def update_run_plan(self, run_id: str, steps_planned: list[dict[str, Any]]) -> None:
        """Record the plan on the run row after PLAN_CREATED."""

    @abstractmethod
    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        result: dict[str, Any],
        error: str | None,
        steps_executed: int,
    ) -> RunRecord | None:
        """Finalize a run; return the updated record or None if unknown."""

    @abstractmethod
    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return the run or None if unknown."""


class MemoryStorage(Storage):
    """Process-local storage (tests, dry runs). Not restart-safe by design."""

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}
        self._runs: dict[str, RunRecord] = {}

    # -- Goals ---------------------------------------------------------------

    async def create_goal(self, goal: Goal) -> Goal:
        now = _utcnow()
        stored = goal.model_copy(deep=True)
        stored.created_at = now
        stored.updated_at = now
        self._goals[stored.id] = stored
        return stored.model_copy(deep=True)

    async def get_goal(self, goal_id: str) -> Goal | None:
        goal = self._goals.get(goal_id)
        return goal.model_copy(deep=True) if goal else None

    async def update_goal_status(self, goal_id: str, status: str) -> Goal | None:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        goal.status = status
        goal.updated_at = _utcnow()
        return goal.model_copy(deep=True)

    # -- Runs ------------------------------------------------------------------

    async def create_run(self, run: RunRecord) -> RunRecord:
        self._runs[run.id] = run
        return run

    async def update_run_plan(self, run_id: str, steps_planned: list[dict[str, Any]]) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.steps_planned = steps_planned

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        result: dict[str, Any],
        error: str | None,
        steps_executed: int,
    ) -> RunRecord | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        run.status = status
        run.result = result
        run.error = error
        run.steps_executed = steps_executed
        run.finished_at = _utcnow()
        return run

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)


class DatabaseStorage(Storage):
    """PostgreSQL storage on the existing async SQLAlchemy stack."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # -- Goals ---------------------------------------------------------------

    async def create_goal(self, goal: Goal) -> Goal:
        row = GoalORM(
            id=uuid.UUID(goal.id),
            parent_id=uuid.UUID(goal.parent_id) if goal.parent_id else None,
            title=goal.title,
            description=goal.description,
            status=goal.status,
            priority=goal.priority,
            origin=goal.origin,
            success_criteria=goal.success_criteria,
            params=goal.params,
            meta=goal.metadata,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
        return self._goal_from_row(row)

    async def get_goal(self, goal_id: str) -> Goal | None:
        async with self._session_factory() as session:
            row = await session.get(GoalORM, uuid.UUID(goal_id))
            return self._goal_from_row(row) if row else None

    async def update_goal_status(self, goal_id: str, status: str) -> Goal | None:
        async with self._session_factory() as session:
            row = await session.get(GoalORM, uuid.UUID(goal_id))
            if row is None:
                return None
            row.status = status
            await session.commit()
            await session.refresh(row)
            return self._goal_from_row(row)

    # -- Runs ------------------------------------------------------------------

    async def create_run(self, run: RunRecord) -> RunRecord:
        row = RunORM(
            id=uuid.UUID(run.id),
            goal_id=uuid.UUID(run.goal_id) if run.goal_id else None,
            status=run.status,
            planner=run.planner,
            perception=run.perception,
            steps_planned=run.steps_planned,
            steps_executed=run.steps_executed,
            result=run.result,
            error=run.error,
            started_at=run.started_at,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        return run

    async def update_run_plan(self, run_id: str, steps_planned: list[dict[str, Any]]) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunORM, uuid.UUID(run_id))
            if row is not None:
                row.steps_planned = steps_planned
                await session.commit()

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        result: dict[str, Any],
        error: str | None,
        steps_executed: int,
    ) -> RunRecord | None:
        async with self._session_factory() as session:
            row = await session.get(RunORM, uuid.UUID(run_id))
            if row is None:
                return None
            row.status = status
            row.result = result
            row.error = error
            row.steps_executed = steps_executed
            row.finished_at = _utcnow()
            await session.commit()
            return RunRecord(
                id=str(row.id),
                goal_id=str(row.goal_id) if row.goal_id else None,
                status=row.status,
                planner=row.planner,
                perception=row.perception,
                started_at=row.started_at,
            )

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(RunORM).where(RunORM.id == uuid.UUID(run_id))
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            record = RunRecord(
                id=str(row.id),
                goal_id=str(row.goal_id) if row.goal_id else None,
                status=row.status,
                planner=row.planner,
                perception=row.perception,
                started_at=row.started_at,
            )
            record.steps_planned = row.steps_planned
            record.steps_executed = row.steps_executed
            record.result = row.result
            record.error = row.error
            record.finished_at = row.finished_at
            return record

    # -- Mapping helpers --------------------------------------------------------

    @staticmethod
    def _goal_from_row(row: GoalORM) -> Goal:
        return Goal(
            id=str(row.id),
            title=row.title,
            description=row.description,
            status=row.status,
            priority=row.priority,
            origin=row.origin,
            params=row.params,
            success_criteria=row.success_criteria,
            parent_id=str(row.parent_id) if row.parent_id else None,
            metadata=row.meta,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
