"""Goal domain models and the GoalManager.

The manager owns the goal lifecycle state machine (``pending`` → ``running``
→ ``completed``/``failed``/``cancelled``, with ``paused`` reachable from
``pending``/``running``) and persists through the :class:`Storage`
abstraction (``agency.storage``). Invalid transitions raise
:class:`GoalTransitionError` — the loop cannot smuggle a goal through an
illegal path, and future pause/resume support rides on the same rules.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (storage imports goals)
    from agency.storage import Storage


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


GOAL_STATUSES: frozenset[str] = frozenset(
    {"pending", "running", "paused", "completed", "failed", "cancelled"}
)

#: Directed edges of the goal lifecycle. Terminal states have no out-edges.
VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running", "paused", "cancelled", "failed"}),
    "running": frozenset({"paused", "completed", "failed", "cancelled"}),
    "paused": frozenset({"running", "cancelled", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class GoalTransitionError(ValueError):
    """Raised when a goal status transition is not allowed."""


class GoalSpec(BaseModel):
    """Input for creating a goal."""

    title: str = Field(min_length=1)
    description: str = ""
    priority: int = 0
    #: ``human`` or ``self``.
    origin: str = "human"
    #: Planner hints (e.g. explicit ``steps`` for the deterministic planner).
    params: dict[str, Any] = Field(default_factory=dict)
    #: Structured success criteria for evaluators (Phase P6+).
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Goal(BaseModel):
    """Full goal record (spec + lifecycle + timestamps)."""

    id: str = Field(default_factory=_new_id)
    title: str = Field(min_length=1)
    description: str = ""
    status: str = "pending"
    priority: int = 0
    origin: str = "human"
    params: dict[str, Any] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_spec(cls, spec: GoalSpec) -> Goal:
        return cls(**spec.model_dump())


class GoalManager:
    """Creates goals and enforces the lifecycle state machine."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    async def create(self, spec: GoalSpec) -> Goal:
        if spec.origin not in {"human", "self"}:
            raise ValueError(f"goal origin must be 'human' or 'self', got {spec.origin!r}")
        goal = Goal.from_spec(spec)
        return await self._storage.create_goal(goal)

    async def get(self, goal_id: str) -> Goal:
        goal = await self._storage.get_goal(goal_id)
        if goal is None:
            raise KeyError(goal_id)
        return goal

    async def transition(self, goal_id: str, new_status: str) -> Goal:
        """Move a goal to ``new_status`` if the transition is legal."""
        if new_status not in GOAL_STATUSES:
            raise GoalTransitionError(
                f"unknown goal status {new_status!r}; expected one of {sorted(GOAL_STATUSES)}"
            )
        goal = await self.get(goal_id)
        if new_status == goal.status:
            return goal  # idempotent no-op
        allowed = VALID_TRANSITIONS[goal.status]
        if new_status not in allowed:
            raise GoalTransitionError(
                f"illegal goal transition {goal.status!r} -> {new_status!r} "
                f"(allowed: {sorted(allowed) or 'none — terminal state'})"
            )
        updated = await self._storage.update_goal_status(goal_id, new_status)
        if updated is None:
            raise KeyError(goal_id)
        return updated

    # Convenience wrappers for the common loop paths ------------------------

    async def mark_running(self, goal_id: str) -> Goal:
        return await self.transition(goal_id, "running")

    async def mark_completed(self, goal_id: str) -> Goal:
        return await self.transition(goal_id, "completed")

    async def mark_failed(self, goal_id: str) -> Goal:
        return await self.transition(goal_id, "failed")

    async def mark_cancelled(self, goal_id: str) -> Goal:
        return await self.transition(goal_id, "cancelled")


def new_goal_id() -> str:
    """Public helper for callers that pre-allocate goal ids."""
    return uuid.uuid4().hex
