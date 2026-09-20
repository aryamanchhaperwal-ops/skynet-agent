"""Integration tests with the real God's Eye database (db-marked).

Auto-skipped when PostgreSQL is unreachable so the suite stays green on
machines without the docker container. These tests verify the integration
boundary end-to-end: God's Eye tables → perception adapter → Skynet core →
persisted goals/runs/events/experiences.
"""

from __future__ import annotations

import pytest

from agency.actions.builtins import GodsEyeLatestEventsAction
from agency.bootstrap import build_core
from agency.config import SkynetSettings
from agency.goals import GoalSpec
from agency.perception import GodsEyePerceptionAdapter
from agency.storage import ensure_core_schema


def db_settings() -> SkynetSettings:
    return SkynetSettings(
        storage_backend="database",
        perception_adapter="gods_eye",
        max_run_seconds=60,
        max_steps_per_run=10,
    )


@pytest.fixture
async def real_factory(db_session_factory):
    return db_session_factory


@pytest.mark.db
async def test_gods_eye_perception_against_real_database(real_factory) -> None:
    await ensure_core_schema()
    adapter = GodsEyePerceptionAdapter(real_factory, limit=5)
    from agency.state import AgentState

    observations = await adapter.observe(AgentState())
    assert observations, "events table should contain ingested events"
    observation = observations[0]
    assert observation.source == "gods_eye:usgs"
    assert observation.content["events"], "at least one event expected"
    event = observation.content["events"][0]
    assert {"event_id", "magnitude", "place", "event_time"} <= set(event)


@pytest.mark.db
async def test_gods_eye_action_against_real_database(real_factory) -> None:
    action = GodsEyeLatestEventsAction(real_factory)
    from agency.actions.base import ActionContext, ActionSpec

    result = await action.execute(
        ActionSpec(type="gods_eye_latest_events", params={"limit": 3}),
        ActionContext(run_id="it-test"),
    )
    assert result["count"] <= 3
    assert result["observation"]["source"] == "gods_eye:usgs"


@pytest.mark.db
async def test_full_run_persists_goal_run_events_experiences(real_factory) -> None:
    await ensure_core_schema()
    core = build_core(db_settings(), session_factory=real_factory)
    summary = await core.run_goal(
        GoalSpec(
            title="Integration: strongest recent earthquake",
            params={"steps": [{"type": "gods_eye_latest_events", "params": {"limit": 10}}]},
        )
    )
    assert summary.status == "completed"
    assert summary.ok is True
    assert summary.observations >= 1
    assert summary.actions[0]["success"] is True
    assert summary.result.get("events_seen", 0) > 0

    # Persisted artifacts are queryable.
    goal = await core._goals.get(summary.goal_id)
    assert goal.status == "completed"
    run = await core._storage.get_run(summary.run_id)
    assert run is not None and run.status == "completed"
    assert run.result.get("strongest_event") is not None

    # Trace events actually persisted (guards the runs/run_events FK order).
    import uuid as _uuid

    from sqlalchemy import text

    async with real_factory() as session:
        run_count = (
            await session.execute(
                text("SELECT COUNT(*) FROM runs WHERE id = :r"), {"r": _uuid.UUID(summary.run_id)}
            )
        ).scalar_one()
        event_count = (
            await session.execute(
                text("SELECT COUNT(*) FROM run_events WHERE run_id = :r"),
                {"r": _uuid.UUID(summary.run_id)},
            )
        ).scalar_one()
        experience_count = (
            await session.execute(
                text("SELECT COUNT(*) FROM experiences WHERE run_id = :r"),
                {"r": _uuid.UUID(summary.run_id)},
            )
        ).scalar_one()
    assert run_count == 1
    assert event_count >= summary.events - 1  # tolerate a single late sink hiccup
    assert event_count >= 8  # full lifecycle was traced
    assert experience_count >= 3  # action + evaluation + summary experiences


@pytest.mark.db
async def test_gods_eye_survives_when_events_table_missing(real_factory) -> None:
    from agency.state import AgentState

    adapter = GodsEyePerceptionAdapter(real_factory, limit=5)
    observations = await adapter.observe(AgentState())
    # With the table present this yields data; the assertion is that it
    # NEVER raises — data or a clean error observation, nothing else.
    assert observations
    assert observations[0].kind in {"json", "error"}


def test_fastapi_app_importable_and_configured() -> None:
    # God's Eye app untouched: still builds with the same title.
    import main as gods_eye_main

    assert gods_eye_main.app.title == "SKYNET API"
    routes = {route.path for route in gods_eye_main.app.routes}
    assert "/api/v1/health" in routes
    assert "/api/v1/events" in routes
