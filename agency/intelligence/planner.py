"""LLM-backed planner with a deterministic safety net.

Contract: explicit goal steps (``goal.params["steps"]``) always win —
operator-specified plans are honored verbatim regardless of strategy, so
tests and demos stay deterministic even with the LLM strategy enabled.
Otherwise the model proposes steps, which are validated (schema + action
allow-list + step cap) before use; any failure falls back to the wrapped
deterministic planner. The LLM never breaks planning.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agency.actions.base import ActionSpec
from agency.goals import Goal
from agency.intelligence.service import IntelligenceService
from agency.planner import DeterministicPlanner, Plan, Planner
from agency.state import AgentState

logger = logging.getLogger("skynet.intelligence.planner")


class LLMPlanStep(BaseModel):
    """One model-proposed step (validated before it becomes an ActionSpec)."""

    type: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class LLMPlanPayload(BaseModel):
    """Schema the model must satisfy for its plan to be used."""

    steps: list[LLMPlanStep] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=2000)


class LLMPlanner(Planner):
    """Model-proposed plans, validated and fallback-protected.

    ``fallback`` is always the deterministic planner (or any Planner);
    ``allowed_actions`` filters hallucinated tool names — proposed steps
    referencing unregistered actions are dropped, and a plan left empty by
    filtering falls back entirely.
    """

    name = "llm"

    def __init__(
        self,
        service: IntelligenceService,
        *,
        fallback: Planner,
        allowed_actions: frozenset[str] = frozenset(),
        max_steps: int = 10,
    ) -> None:
        self._service = service
        self._fallback = fallback
        self._allowed = allowed_actions
        self._max_steps = max(1, max_steps)

    async def plan(self, goal: Goal, state: AgentState) -> Plan:
        explicit = goal.params.get("steps")
        if isinstance(explicit, list):
            # Operator-specified plan: same contract as the deterministic
            # planner, honored without a model call.
            return await self._fallback.plan(goal, state)

        system = (
            "You are the planning module of Skynet, an autonomous research "
            "agent. Produce a short, safe, reversible action plan. Choose "
            "only from the allowed actions. Respond with JSON only."
        )
        user = (
            f"Goal: {goal.title}\n"
            f"Description: {goal.description or '(none)'}\n"
            f"Allowed actions: {sorted(self._allowed) or '(unknown)'}\n"
            f"Maximum steps: {self._max_steps}\n"
            f"Observations so far: {len(state.observations)}; "
            f"errors so far: {len(state.errors)}.\n"
            'Respond as {"steps": [{"type": ..., "params": {...}}], '
            '"rationale": "..."}'
        )
        payload = await self._service.structured(system, user, LLMPlanPayload)
        if payload is None:
            logger.info("llm planner: no usable model plan; falling back")
            return await self._fallback.plan(goal, state)

        steps = [
            ActionSpec(type=step.type, params=dict(step.params))
            for step in payload.steps
            if not self._allowed or step.type in self._allowed
        ]
        if not steps:
            logger.info(
                "llm planner: model plan empty or entirely disallowed; falling back"
            )
            return await self._fallback.plan(goal, state)
        if len(steps) > self._max_steps:
            steps = steps[: self._max_steps]
        return Plan(
            steps=steps,
            rationale=payload.rationale or "llm-proposed plan",
        )


__all__ = ["DeterministicPlanner", "LLMPlanPayload", "LLMPlanStep", "LLMPlanner"]
