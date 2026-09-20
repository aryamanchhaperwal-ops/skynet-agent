"""Storage abstraction tests (memory backend + DB smoke checks)."""

from __future__ import annotations

import pytest

from agency.goals import Goal, GoalSpec
from agency.storage import (
    CORE_TABLES,
    DatabaseStorage,
    MemoryStorage,
    RunRecord,
    Storage,
    ensure_core_schema,
)


def make_goal(title: str = "G") -> Goal:
    return Goal.from_spec(GoalSpec(title=title))


def make_run(goal_id: str | None = None) -> RunRecord:
    return RunRecord(
        id="run-1",
        goal_id=goal_id,
        status="running",
        planner="deterministic",
        perception="none",
    )


def test_storage_protocol_is_abstract() -> None:
    with pytest.raises(TypeError):
        Storage()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_memory_goal_round_trip() -> None:
    storage = MemoryStorage()
    created = await storage.create_goal(make_goal("roundtrip"))
    assert created.status == "pending"
    fetched = await storage.get_goal(created.id)
    assert fetched is not None and fetched.title == "roundtrip"
    assert await storage.get_goal("missing") is None


@pytest.mark.asyncio
async def test_memory_goal_status_update() -> None:
    storage = MemoryStorage()
    goal = await storage.create_goal(make_goal())
    updated = await storage.update_goal_status(goal.id, "running")
    assert updated is not None and updated.status == "running"
    assert await storage.update_goal_status("missing", "running") is None


@pytest.mark.asyncio
async def test_memory_goal_deep_copy_isolation() -> None:
    storage = MemoryStorage()
    goal = await storage.create_goal(make_goal("iso"))
    goal.title = "mutated-by-caller"
    fetched = await storage.get_goal(goal.id)
    assert fetched is not None and fetched.title == "iso"


@pytest.mark.asyncio
async def test_memory_run_lifecycle() -> None:
    storage = MemoryStorage()
    run = await storage.create_run(make_run())
    await storage.update_run_plan(run.id, [{"type": "echo"}])
    finished = await storage.finish_run(
        run.id, status="completed", result={"summary": "done"},
        error=None, steps_executed=1,
    )
    assert finished is not None
    assert finished.status == "completed"
    assert finished.steps_planned == [{"type": "echo"}]
    assert finished.steps_executed == 1
    assert finished.finished_at is not None
    fetched = await storage.get_run(run.id)
    assert fetched is not None and fetched.result == {"summary": "done"}
    assert await storage.get_run("missing") is None


@pytest.mark.asyncio
async def test_memory_finish_run_missing_returns_none() -> None:
    storage = MemoryStorage()
    assert (
        await storage.finish_run(
            "ghost", status="failed", result={}, error="e", steps_executed=0
        )
        is None
    )


def test_core_tables_are_exactly_the_skynet_additions() -> None:
    names = {table.name for table in CORE_TABLES}
    assert names == {"goals", "runs", "run_events", "experiences"}


def test_database_storage_is_lazy() -> None:
    # Construction must not touch the network or the engine.
    storage = DatabaseStorage(session_factory=None)  # type: ignore[arg-type]
    assert storage is not None


@pytest.mark.db
async def test_ensure_core_schema_creates_tables_only(db_session_factory) -> None:
    await ensure_core_schema()
    async with db_session_factory() as session:
        from sqlalchemy import text

        for table in ("goals", "runs", "run_events", "experiences"):
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = :t"
                ),
                {"t": table},
            )
            assert result.scalar_one() == 1, f"table {table} missing"
