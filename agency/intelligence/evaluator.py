"""LLM-backed evaluator with a deterministic safety net.

The model receives the goal, actions with results, and observations, and
must answer through a strict schema (score/passed/reason/evidence/
confidence). Deterministic evaluation is the fallback for *every* failure
mode; an LLM evaluation is recorded as ``source="llm"`` and its notes
carry the model's stated reason — recorded, never treated as objective
truth.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agency.actions.base import ActionResult, ActionSpec
from agency.evaluator import DeterministicEvaluator, Evaluation, Evaluator
from agency.intelligence.service import IntelligenceService
from agency.state import AgentState

logger = logging.getLogger("skynet.intelligence.evaluator")


class LLMEvaluationPayload(BaseModel):
    """Schema the model must satisfy for its verdict to be used."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = Field(default="", max_length=2000)
    evidence: list[str] = Field(default_factory=list, max_length=20)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LLMEvaluator(Evaluator):
    """Model-judged evaluations, recorded with provenance and fallback.

    Judge prompt includes goal, executed actions (type/success/error/duration)
    and observation summaries — never raw external content at full length
    (bounded per observation) so prompts stay small.
    """

    name = "llm"

    #: Per-observation summary cap in the judge prompt (chars).
    OBSERVATION_SNIPPET_CHARS = 200

    def __init__(self, service: IntelligenceService) -> None:
        self._service = service
        self._fallback = DeterministicEvaluator()

    def _render_state(self, state: AgentState) -> str:
        actions = [
            {
                "type": record.spec.type,
                "success": bool(record.result and record.result.success),
                "error": record.result.error if record.result else None,
            }
            for record in state.actions
        ]
        observations = [
            (obs.summary or str(obs.source))[: self.OBSERVATION_SNIPPET_CHARS]
            for obs in state.observations
        ]
        import json

        return json.dumps(
            {
                "goal": state.goal_title,
                "errors": state.errors[:10],
                "actions": actions,
                "observation_summaries": observations,
            },
            indent=2,
        )

    async def evaluate_action(
        self, spec: ActionSpec, result: ActionResult, state: AgentState
    ) -> Evaluation:
        return await self._fallback.evaluate_action(spec, result, state)

    async def evaluate_run(self, state: AgentState) -> Evaluation:
        system = (
            "You are the evaluation module of Skynet. Judge whether the "
            "recorded actions plausibly achieved the goal. Be conservative: "
            "when evidence is thin, lower the score and passed=false unless "
            "actions clearly succeeded. Respond with JSON only."
        )
        user = (
            "Judge this run.\n"
            f"Run state:\n{self._render_state(state)}\n"
            'Respond as {"score": 0.0-1.0, "passed": true/false, '
            '"reason": "...", "evidence": [...], "confidence": 0.0-1.0}'
        )
        payload = await self._service.structured(system, user, LLMEvaluationPayload)
        if payload is None:
            logger.info("llm evaluator: no usable verdict; falling back")
            return await self._fallback.evaluate_run(state)
        notes = payload.reason or "llm evaluation"
        if payload.evidence:
            notes += " | evidence: " + "; ".join(payload.evidence[:5])
        notes += " (judge: llm, not objective truth)"
        return Evaluation(
            subject=f"run:{state.run_id}",
            level="run",
            score=payload.score,
            passed=payload.passed,
            notes=notes,
            source="llm",
            confidence=payload.confidence,
        )


__all__ = ["LLMEvaluationPayload", "LLMEvaluator"]
