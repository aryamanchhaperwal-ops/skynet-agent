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

    #: True when the planner's output is *model-derived* and therefore
    #: untrusted control input: the loop validates such plans before
    #: executing any step. Deterministic plans (code, not a model) are
    #: honoured verbatim, exactly as before.
    plans_are_untrusted: bool = False

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

    #: Conservative preference order for a safe, read-only default action
    #: used when the configured ``default_action`` is not registered/allowed
    #: for the run. Research actions first (they produce observations), then
    #: the always-available echo probe.
    _SAFE_DEFAULTS: tuple[str, ...] = (
        "web_research",
        "web_search",
        "web_fetch",
        "web_extract",
        "echo",
    )

    def __init__(
        self,
        default_action: str = "gods_eye_latest_events",
        default_params: dict[str, Any] | None = None,
        *,
        allowed_actions: frozenset[str] | None = None,
    ) -> None:
        self.default_action = default_action
        self.default_params = dict(default_params or {"limit": 25})
        #: When supplied, the default action must be in this set (allowlist ∩
        #: registry). Anything else is substituted for a registered, allowed,
        #: read-only action so a fallback can never name an unrunnable step.
        #: ``None`` keeps the historical behaviour (no filtering).
        self._allowed = frozenset(allowed_actions) if allowed_actions is not None else None

    def _resolve_default(self, goal: Goal) -> tuple[str, dict[str, Any], str]:
        """Pick the action this planner may actually emit.

        Without an ``allowed_actions`` set the configured default is used
        unchanged (both exist to be exercised by tests). With one, the
        default is honoured only if it is inside the set; otherwise a
        deterministic safe choice is substituted, and the rationale records
        the substitution (never silent).
        """
        if self._allowed is None:
            return self.default_action, dict(self.default_params), "deterministic default action"
        if not self._allowed:
            # Nothing is executable: emit no step at all rather than one the
            # executor must refuse. Callers treat an empty plan as a failure.
            return "", {}, "no allowed actions available"
        if self.default_action and self.default_action in self._allowed:
            return self.default_action, dict(self.default_params), "deterministic default action"
        for candidate in self._SAFE_DEFAULTS:
            if candidate in self._allowed:
                return (
                    candidate,
                    self._params_for(candidate, goal),
                    f"deterministic safe fallback: {candidate!r} "
                    f"(configured default {self.default_action!r} not allowed)",
                )
        # No preferred action available: use the alphabetically first allowed
        # action so the choice is deterministic and still registered.
        chosen = sorted(self._allowed)[0]
        return (
            chosen,
            {},
            f"deterministic safe fallback: {chosen!r} "
            f"(configured default {self.default_action!r} not allowed)",
        )

    @staticmethod
    def _params_for(action: str, goal: Goal) -> dict[str, Any]:
        """Minimal parameters for a substituted action, derived from the goal."""
        if action == "web_research":
            text = (goal.description or goal.title or "").strip()[:200]
            return {"goal": text} if text else {}
        if action in {"web_search", "web_fetch", "web_extract"}:
            text = (goal.title or "").strip()[:200]
            key = "url" if action in {"web_fetch", "web_extract"} else "query"
            return {key: text} if text else {}
        return {}

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
        action, params, rationale = self._resolve_default(goal)
        if action:
            return Plan(
                steps=[ActionSpec(type=action, params=params)],
                rationale=rationale,
            )
        return Plan(steps=[], rationale="no default action configured")
