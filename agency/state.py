"""Serializable runtime state of one Skynet run.

The state is the loop's working memory: current goal, step cursor,
observations, action records, errors and free-form context. It is a pydantic
model, so ``state.model_dump(mode="json")`` / ``AgentState.model_validate``
is the serialization round-trip (the run summary embeds it, and future
resumable runs will persist it as JSONB).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agency.actions.base import ActionRecord
from agency.observation import Observation

StateStatus = Literal["initialized", "running", "completed", "failed", "cancelled"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class AgentState(BaseModel):
    """Structured, serializable runtime state of one run."""

    run_id: str = Field(default_factory=_new_id)
    goal_id: str | None = None
    goal_title: str = ""
    status: StateStatus = "initialized"
    #: 0-based index of the plan step currently being executed.
    current_step: int = 0
    observations: list[Observation] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    #: Free-form, JSON-serializable scratch space for planners/evaluators.
    context: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.touch()

    def add_action(self, record: ActionRecord) -> None:
        self.actions.append(record)
        self.touch()

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.touch()
