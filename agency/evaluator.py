"""Evaluator abstraction.

Evaluations are the quantitative seam future learning hangs off: the
deterministic evaluator below scores structural success only. The same ABC
later admits LLM-judge, benchmark, statistical, human-feedback and
multi-agent evaluators without touching the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from agency.actions.base import ActionResult, ActionSpec
from agency.state import AgentState

EvaluationLevel = Literal["action", "run"]


class Evaluation(BaseModel):
    """Verdict about one action or one whole run."""

    subject: str
    level: EvaluationLevel
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    notes: str = ""


class Evaluator(ABC):
    """Strategy interface for judging actions and runs."""

    #: Component name recorded on the run row.
    name: str = "evaluator"

    @abstractmethod
    async def evaluate_action(
        self, spec: ActionSpec, result: ActionResult, state: AgentState
    ) -> Evaluation:
        """Judge one executed action."""

    @abstractmethod
    async def evaluate_run(self, state: AgentState) -> Evaluation:
        """Judge the run as a whole."""


class DeterministicEvaluator(Evaluator):
    """Structural, deterministic evaluation.

    - Action: passes iff the result succeeded; score is 1.0/0.0.
    - Run: passes iff there are no recorded errors and every executed action
      succeeded (a run with zero actions and no errors passes). Score is the
      fraction of successful actions.

    Deliberately does NOT read ``state.status``: the run status is the loop's
    verdict about *cycle integrity*; the evaluator judges the *work*. The two
    signals stay independent (see ``RunSummary`` in ``agency.loop``).
    """

    name = "deterministic"

    async def evaluate_action(
        self, spec: ActionSpec, result: ActionResult, state: AgentState
    ) -> Evaluation:
        del state  # deterministic policy needs no history (yet)
        notes = (
            f"action {result.action_type} completed in {result.duration_ms} ms"
            if result.success
            else f"action {result.action_type} failed: {result.error}"
        )
        return Evaluation(
            subject=f"action:{result.action_id}",
            level="action",
            score=1.0 if result.success else 0.0,
            passed=result.success,
            notes=notes,
        )

    async def evaluate_run(self, state: AgentState) -> Evaluation:
        total = len(state.actions)
        successes = sum(1 for record in state.actions if record.result and record.result.success)
        score = successes / total if total else 1.0
        passed = not state.errors and successes == total
        notes = f"{successes}/{total} actions succeeded, {len(state.errors)} error(s)"
        return Evaluation(
            subject=f"run:{state.run_id}",
            level="run",
            score=score,
            passed=passed,
            notes=notes,
        )
