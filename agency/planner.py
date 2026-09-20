"""Planner abstraction.

``plan(goal, state) -> Plan`` is the seam where reasoning arrives: the
deterministic planner below is a placeholder strategy whose only job is to
exercise the loop. An LLM-based planner (intelligence phase) implements the
same ABC and is injected in :func:`agency.bootstrap.build_core` — nothing
else in the core changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agency.actions.base import ActionSpec
from agency.goals import Goal
from agency.state import AgentState


class Plan(BaseModel):
    """An ordered list of action steps with a rationale."""

    steps: list[ActionSpec] = Field(default_factory=list)
    rationale: str = ""

    @property
    def step_types(self) -> list[str]:
        return [step.type for step in self.steps]


class Planner(ABC):
    """Strategy interface: given a goal and current state, produce a plan."""

    #: Component name recorded on the run row.
    name: str = "planner"

    @abstractmethod
    async def plan(self, goal: Goal, state: AgentState) -> Plan:
        """Produce the plan for this run."""


class DeterministicPlanner(Planner):
    """Placeholder strategy, fully replaceable.

    Strategy (in order):
    1. If the goal carries explicit ``params["steps"]`` (a list of
       ``{"type": ..., "params": {...}}`` dicts), honor it verbatim — this
       is how tests and demos exercise precise flows deterministically.
    2. Otherwise emit a single default step (``default_action``), which for
       the foundation is the God's Eye read-only fetch.
    """

    name = "deterministic"

    def __init__(
        self,
        default_action: str = "gods_eye_latest_events",
        default_params: dict[str, Any] | None = None,
    ) -> None:
        self.default_action = default_action
        self.default_params = dict(default_params or {"limit": 25})

    async def plan(self, goal: Goal, state: AgentState) -> Plan:
        explicit = goal.params.get("steps")
        if isinstance(explicit, list):
            steps: list[ActionSpec] = []
            for raw in explicit:
                if not isinstance(raw, dict) or "type" not in raw:
                    raise ValueError(
                        "goal params['steps'] entries must be objects with a 'type' key"
                    )
                params = raw.get("params")
                if params is not None and not isinstance(params, dict):
                    raise ValueError("goal params['steps'] entry 'params' must be an object")
                steps.append(ActionSpec(type=str(raw["type"]), params=dict(params or {})))
            return Plan(steps=steps, rationale="goal-specified plan")
        if self.default_action:
            return Plan(
                steps=[ActionSpec(type=self.default_action, params=dict(self.default_params))],
                rationale="deterministic default action",
            )
        return Plan(steps=[], rationale="no default action configured")
