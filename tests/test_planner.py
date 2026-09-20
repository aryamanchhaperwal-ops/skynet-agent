"""Planner abstraction and DeterministicPlanner tests."""

from __future__ import annotations

import pytest

from agency.actions.base import ActionSpec
from agency.goals import Goal, GoalSpec
from agency.planner import DeterministicPlanner, Plan, Planner
from agency.state import AgentState


def make_goal(params: dict | None = None) -> Goal:
    return Goal.from_spec(GoalSpec(title="G", params=params or {}))


@pytest.mark.asyncio
async def test_default_plan_is_single_default_action() -> None:
    planner = DeterministicPlanner(default_action="echo", default_params={"text": "hi"})
    plan = await planner.plan(make_goal(), AgentState())
    assert plan.step_types == ["echo"]
    assert plan.steps[0].params == {"text": "hi"}
    assert plan.rationale == "deterministic default action"


@pytest.mark.asyncio
async def test_goal_specified_steps_override_default() -> None:
    goal = make_goal({"steps": [{"type": "echo", "params": {"text": "a"}}]})
    plan = await DeterministicPlanner().plan(goal, AgentState())
    assert plan.step_types == ["echo"]
    assert plan.steps[0].params == {"text": "a"}
    assert plan.rationale == "goal-specified plan"


@pytest.mark.asyncio
async def test_goal_specified_multiple_steps_preserve_order() -> None:
    goal = make_goal({"steps": [{"type": "a"}, {"type": "b"}, {"type": "c"}]})
    plan = await DeterministicPlanner().plan(goal, AgentState())
    assert plan.step_types == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_malformed_step_missing_type_rejected() -> None:
    goal = make_goal({"steps": [{"params": {}}]})
    with pytest.raises(ValueError, match="'type' key"):
        await DeterministicPlanner().plan(goal, AgentState())


@pytest.mark.asyncio
async def test_malformed_step_params_rejected() -> None:
    goal = make_goal({"steps": [{"type": "echo", "params": "not-a-dict"}]})
    with pytest.raises(ValueError, match="must be an object"):
        await DeterministicPlanner().plan(goal, AgentState())


@pytest.mark.asyncio
async def test_no_default_action_yields_empty_plan() -> None:
    plan = await DeterministicPlanner(default_action="").plan(make_goal(), AgentState())
    assert plan.steps == []


def test_plan_step_types_property() -> None:
    plan = Plan(steps=[ActionSpec(type="x"), ActionSpec(type="y")], rationale="r")
    assert plan.step_types == ["x", "y"]


def test_planner_is_abstract() -> None:
    with pytest.raises(TypeError):
        Planner()  # type: ignore[abstract]


def test_planner_name_recorded() -> None:
    assert DeterministicPlanner().name == "deterministic"
