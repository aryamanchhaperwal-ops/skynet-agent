"""Goal domain and GoalManager tests (memory storage)."""

from __future__ import annotations

import pytest

from agency.goals import (
    GOAL_STATUSES,
    VALID_TRANSITIONS,
    Goal,
    GoalManager,
    GoalSpec,
    GoalTransitionError,
)
from agency.storage import MemoryStorage


def test_goal_spec_defaults() -> None:
    spec = GoalSpec(title="Do a thing")
    assert spec.priority == 0
    assert spec.origin == "human"
    assert spec.params == {}
    assert spec.success_criteria == {}


def test_goal_spec_rejects_empty_title() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GoalSpec(title="")


def test_goal_from_spec_copies_fields() -> None:
    spec = GoalSpec(
        title="T",
        description="D",
        priority=5,
        params={"steps": [{"type": "echo"}]},
        metadata={"k": "v"},
    )
    goal = Goal.from_spec(spec)
    assert goal.title == "T"
    assert goal.priority == 5
    assert goal.params == {"steps": [{"type": "echo"}]}
    assert goal.metadata == {"k": "v"}
    assert goal.status == "pending"
    assert goal.id


def test_transition_table_terminal_states_have_no_out_edges() -> None:
    for terminal in ("completed", "failed", "cancelled"):
        assert VALID_TRANSITIONS[terminal] == frozenset()


def test_transition_table_completeness() -> None:
    assert set(VALID_TRANSITIONS) == GOAL_STATUSES
    assert GOAL_STATUSES == {"pending", "running", "paused", "completed", "failed", "cancelled"}


@pytest.mark.asyncio
async def test_manager_create_assigns_pending_status_and_timestamps() -> None:
    manager = GoalManager(MemoryStorage())
    goal = await manager.create(GoalSpec(title="G1"))
    assert goal.status == "pending"
    assert goal.created_at is not None
    assert goal.updated_at is not None


@pytest.mark.asyncio
async def test_manager_full_lifecycle_pending_running_completed() -> None:
    manager = GoalManager(MemoryStorage())
    goal = await manager.create(GoalSpec(title="G2"))
    goal = await manager.mark_running(goal.id)
    assert goal.status == "running"
    goal = await manager.mark_completed(goal.id)
    assert goal.status == "completed"


@pytest.mark.asyncio
async def test_manager_pause_and_resume() -> None:
    manager = GoalManager(MemoryStorage())
    goal = await manager.create(GoalSpec(title="G3"))
    goal = await manager.transition(goal.id, "paused")
    assert goal.status == "paused"
    goal = await manager.transition(goal.id, "running")
    assert goal.status == "running"


@pytest.mark.asyncio
async def test_illegal_transition_rejected() -> None:
    manager = GoalManager(MemoryStorage())
    goal = await manager.create(GoalSpec(title="G4"))
    with pytest.raises(GoalTransitionError, match="illegal goal transition"):
        await manager.transition(goal.id, "completed")  # pending -> completed not allowed


@pytest.mark.asyncio
async def test_terminal_state_is_frozen() -> None:
    manager = GoalManager(MemoryStorage())
    goal = await manager.create(GoalSpec(title="G5"))
    await manager.mark_failed(goal.id)
    with pytest.raises(GoalTransitionError, match="terminal state"):
        await manager.mark_completed(goal.id)


@pytest.mark.asyncio
async def test_same_status_transition_is_idempotent() -> None:
    manager = GoalManager(MemoryStorage())
    goal = await manager.create(GoalSpec(title="G6"))
    again = await manager.transition(goal.id, "pending")
    assert again.status == "pending"


@pytest.mark.asyncio
async def test_unknown_status_rejected() -> None:
    manager = GoalManager(MemoryStorage())
    goal = await manager.create(GoalSpec(title="G7"))
    with pytest.raises(GoalTransitionError, match="unknown goal status"):
        await manager.transition(goal.id, "zombified")


@pytest.mark.asyncio
async def test_unknown_goal_raises_key_error() -> None:
    manager = GoalManager(MemoryStorage())
    with pytest.raises(KeyError):
        await manager.get("nope")
    # A transition on an unknown goal surfaces the underlying KeyError
    # (documented behavior: the loop treats it as an unexpected failure).


@pytest.mark.asyncio
async def test_manager_rejects_bad_origin() -> None:
    manager = GoalManager(MemoryStorage())
    with pytest.raises(ValueError, match="origin"):
        await manager.create(GoalSpec(title="G8", origin="alien"))


@pytest.mark.asyncio
async def test_update_status_on_missing_goal_returns_none() -> None:
    storage = MemoryStorage()
    assert await storage.update_goal_status("missing", "running") is None
