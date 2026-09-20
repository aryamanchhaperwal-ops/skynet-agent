"""Evaluator abstraction and DeterministicEvaluator tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agency.actions.base import ActionRecord, ActionResult, ActionSpec
from agency.evaluator import DeterministicEvaluator, Evaluation, Evaluator
from agency.state import AgentState


def make_result(success: bool, action_type: str = "echo") -> ActionResult:
    spec = ActionSpec(type=action_type)
    return ActionResult(
        action_id=spec.id,
        action_type=action_type,
        success=success,
        output={"x": 1} if success else None,
        error=None if success else "ActionError: nope",
    )


def record(success: bool) -> ActionRecord:
    spec = ActionSpec(type="echo")
    return ActionRecord(spec=spec, result=make_result(success))


@pytest.mark.asyncio
async def test_successful_action_scores_full() -> None:
    spec = ActionSpec(type="echo")
    evaluation = await DeterministicEvaluator().evaluate_action(spec, make_result(True), AgentState())
    assert evaluation.passed is True
    assert evaluation.score == 1.0
    assert evaluation.level == "action"
    assert evaluation.subject.startswith("action:")
    assert "completed" in evaluation.notes


@pytest.mark.asyncio
async def test_failed_action_scores_zero() -> None:
    spec = ActionSpec(type="echo")
    result = make_result(False)
    evaluation = await DeterministicEvaluator().evaluate_action(spec, result, AgentState())
    assert evaluation.passed is False
    assert evaluation.score == 0.0
    assert "failed" in evaluation.notes
    assert result.error in evaluation.notes


@pytest.mark.asyncio
async def test_run_with_all_successes_passes() -> None:
    state = AgentState(status="running")  # status must NOT influence the verdict
    state.add_action(record(True))
    state.add_action(record(True))
    evaluation = await DeterministicEvaluator().evaluate_run(state)
    assert evaluation.passed is True
    assert evaluation.score == 1.0
    assert evaluation.level == "run"


@pytest.mark.asyncio
async def test_run_with_failure_fails() -> None:
    state = AgentState()
    state.add_action(record(True))
    state.add_action(record(False))
    evaluation = await DeterministicEvaluator().evaluate_run(state)
    assert evaluation.passed is False
    assert evaluation.score == 0.5


@pytest.mark.asyncio
async def test_run_with_errors_fails_even_if_actions_succeeded() -> None:
    state = AgentState()
    state.add_action(record(True))
    state.add_error("perception failed: RuntimeError: db down")
    evaluation = await DeterministicEvaluator().evaluate_run(state)
    assert evaluation.passed is False
    assert evaluation.score == 1.0


@pytest.mark.asyncio
async def test_zero_action_error_free_run_passes() -> None:
    state = AgentState()
    evaluation = await DeterministicEvaluator().evaluate_run(state)
    assert evaluation.passed is True
    assert evaluation.score == 1.0
    assert "0/0" in evaluation.notes


def test_evaluator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Evaluator()  # type: ignore[abstract]


def test_evaluation_score_bounds() -> None:
    with pytest.raises(ValidationError):
        Evaluation(subject="x", level="action", score=1.5, passed=True)
    with pytest.raises(ValidationError):
        Evaluation(subject="x", level="run", score=-0.1, passed=False)


def test_evaluator_name() -> None:
    assert DeterministicEvaluator().name == "deterministic"
