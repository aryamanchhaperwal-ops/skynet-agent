"""AgentState tests (serializability, mutators)."""

from __future__ import annotations

import pytest

from agency.actions.base import ActionRecord, ActionResult, ActionSpec
from agency.observation import Observation
from agency.state import AgentState


def make_record(success: bool = True) -> ActionRecord:
    spec = ActionSpec(type="echo")
    result = ActionResult(
        action_id=spec.id, action_type="echo", success=success,
        error=None if success else "boom",
    )
    return ActionRecord(spec=spec, result=result)


def test_defaults() -> None:
    state = AgentState()
    assert state.run_id
    assert state.status == "initialized"
    assert state.current_step == 0
    assert state.observations == [] and state.actions == [] and state.errors == []
    assert state.context == {}
    assert state.started_at.tzinfo is not None


def test_serialization_round_trip() -> None:
    state = AgentState(goal_title="demo")
    state.add_observation(Observation(source="gods_eye:usgs", kind="json", content={"events": []}))
    state.add_action(make_record(True))
    state.add_error("something went sideways")
    state.context["cursor"] = 42

    clone = AgentState.model_validate(state.model_dump(mode="json"))
    assert clone.run_id == state.run_id
    assert clone.goal_title == "demo"
    assert len(clone.observations) == 1
    assert clone.observations[0].source == "gods_eye:usgs"
    assert clone.actions[0].result.success is True
    assert clone.errors == ["something went sideways"]
    assert clone.context == {"cursor": 42}
    assert clone.updated_at == state.updated_at


def test_mutators_touch_updated_at() -> None:
    state = AgentState()
    before = state.updated_at
    state.add_observation(Observation(source="s"))
    state.add_action(make_record())
    state.add_error("e")
    assert state.updated_at >= before


def test_status_is_constrained() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentState(status="transcendent")  # type: ignore[arg-type]
