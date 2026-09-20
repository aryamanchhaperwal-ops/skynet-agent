"""Perception adapter tests (God's Eye integration boundary)."""

from __future__ import annotations

import pytest

from agency.perception import (
    GodsEyePerceptionAdapter,
    NullPerceptionAdapter,
    PerceptionAdapter,
    build_perception_adapter,
)
from agency.state import AgentState
from tests.conftest import BrokenSessionFactory


def test_perception_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        PerceptionAdapter()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_null_adapter_observes_nothing() -> None:
    assert await NullPerceptionAdapter().observe(AgentState()) == []


@pytest.mark.asyncio
async def test_null_adapter_name_recorded() -> None:
    assert NullPerceptionAdapter().name == "none"


@pytest.mark.asyncio
async def test_gods_eye_adapter_normalizes_events(fake_event_factory) -> None:
    adapter = GodsEyePerceptionAdapter(fake_event_factory, limit=10)
    observations = await adapter.observe(AgentState())
    assert adapter.name == "gods_eye"
    assert len(observations) == 1
    observation = observations[0]
    assert observation.source == "gods_eye:usgs"
    assert observation.kind == "json"
    assert observation.confidence == 1.0
    events = observation.content["events"]
    assert events[0]["event_id"] == "usgs-test-1"
    assert events[1]["magnitude"] == 2.1
    assert events[1]["depth"] is None


@pytest.mark.asyncio
async def test_gods_eye_adapter_rejects_bad_limits() -> None:
    factory = BrokenSessionFactory()
    for bad in (0, 201):
        with pytest.raises(ValueError, match="limit"):
            GodsEyePerceptionAdapter(factory, limit=bad)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_gods_eye_adapter_degrades_on_database_failure() -> None:
    adapter = GodsEyePerceptionAdapter(BrokenSessionFactory())  # type: ignore[arg-type]
    observations = await adapter.observe(AgentState())
    assert len(observations) == 1
    observation = observations[0]
    assert observation.kind == "error"
    assert observation.confidence == 0.0
    assert "database is down" in observation.summary


@pytest.mark.asyncio
async def test_error_observation_carries_run_and_goal_ids() -> None:
    adapter = GodsEyePerceptionAdapter(BrokenSessionFactory())  # type: ignore[arg-type]
    state = AgentState(run_id="run-9", goal_id="goal-9")
    observations = await adapter.observe(state)
    assert observations[0].run_id == "run-9"
    assert observations[0].goal_id == "goal-9"


def test_builder_selects_none_adapter() -> None:
    adapter = build_perception_adapter("none")
    assert isinstance(adapter, NullPerceptionAdapter)


def test_builder_selects_gods_eye_adapter(fake_event_factory) -> None:
    adapter = build_perception_adapter("gods_eye", fake_event_factory)
    assert isinstance(adapter, GodsEyePerceptionAdapter)


def test_builder_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown perception adapter"):
        build_perception_adapter("crystal_ball")


def test_builder_requires_session_factory_for_gods_eye() -> None:
    with pytest.raises(ValueError, match="requires a database session factory"):
        build_perception_adapter("gods_eye", None)
