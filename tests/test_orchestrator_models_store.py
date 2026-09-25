"""Orchestrator models, state machine and run store.

Covers: stage transitions (valid/invalid), run creation, persistence,
cross-process pause/resume/cancel, budgets, and failure recovery.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agency.orchestrator.models import (
    AutonomousRun,
    BudgetDecision,
    OrchestratorBudgets,
    RunStage,
    RunStatus,
)
from agency.orchestrator.state_machine import (
    InvalidTransitionError,
    allowed_stages,
    can_transition,
    require_transition,
)
from agency.orchestrator.store import RunStore

# -- state machine --------------------------------------------------------------------


def test_happy_path_transitions_are_allowed() -> None:
    chain = [
        (RunStage.INITIALIZE, RunStage.OBSERVE),
        (RunStage.OBSERVE, RunStage.PLAN),
        (RunStage.PLAN, RunStage.RESEARCH),
        (RunStage.RESEARCH, RunStage.COMMUNICATE),
        (RunStage.COMMUNICATE, RunStage.ACT),
        (RunStage.ACT, RunStage.EVALUATE),
        (RunStage.EVALUATE, RunStage.PLAN),
        (RunStage.EVALUATE, RunStage.LEARN),
        (RunStage.EVALUATE, RunStage.IMPROVE),
        (RunStage.LEARN, RunStage.COMPLETE),
        (RunStage.IMPROVE, RunStage.PLAN),
    ]
    for current, target in chain:
        assert can_transition(current, target), f"{current} -> {target}"
        require_transition(current, target)  # must not raise


def test_complete_stage_is_terminal() -> None:
    assert allowed_stages(RunStage.COMPLETE) == frozenset()
    with pytest.raises(InvalidTransitionError):
        require_transition(RunStage.COMPLETE, RunStage.PLAN)


def test_invalid_transitions_rejected() -> None:
    bad = [
        (RunStage.OBSERVE, RunStage.ACT),  # skipping plan+research
        (RunStage.ACT, RunStage.RESEARCH),  # act cannot go backwards
        (RunStage.INITIALIZE, RunStage.EVALUATE),
        (RunStage.RESEARCH, RunStage.COMPLETE),  # must pass through evaluate
    ]
    for current, target in bad:
        assert not can_transition(current, target), f"{current} -> {target}"
        with pytest.raises(InvalidTransitionError):
            require_transition(current, target)


def test_every_stage_has_an_entry_from_initialize_or_elsewhere() -> None:
    """No unreachable stages ( COMPLETE excepted — it is terminal )."""
    reachable = {RunStage.INITIALIZE}
    changed = True
    while changed:
        changed = False
        for stage in reachable.copy():
            for target in allowed_stages(stage):
                if target not in reachable:
                    reachable.add(target)
                    changed = True
    expected = set(RunStage) - {RunStage.COMPLETE}
    assert expected - reachable == set(), f"unreachable stages: {expected - reachable}"


# -- budgets --------------------------------------------------------------------------


def test_budget_check_proceed_pause_stop() -> None:
    budgets = OrchestratorBudgets(max_iterations=2, max_web_requests=3)
    assert budgets.check(
        iteration=0, runtime_seconds=0.0, web_requests=0, ai_requests=0,
        experiments=0,
    ) is BudgetDecision.PROCEED
    assert budgets.check(
        iteration=0, runtime_seconds=0.0, web_requests=4, ai_requests=0,
        experiments=0,
    ) is BudgetDecision.PAUSE
    assert budgets.check(
        iteration=2, runtime_seconds=0.0, web_requests=0, ai_requests=0,
        experiments=0,
    ) is BudgetDecision.STOP


def test_budget_cost_ceiling() -> None:
    budgets = OrchestratorBudgets(max_iterations=5, max_cost_usd=1.0)
    assert budgets.check(
        iteration=0, runtime_seconds=0.0, web_requests=0, ai_requests=0,
        experiments=0, cost_usd=1.5,
    ) is BudgetDecision.PAUSE


def test_budget_stop_takes_precedence_over_pause_at_iteration_cap() -> None:
    budgets = OrchestratorBudgets(max_iterations=1, max_web_requests=0)
    assert budgets.check(
        iteration=1, runtime_seconds=0.0, web_requests=0, ai_requests=0,
        experiments=0,
    ) is BudgetDecision.STOP


# -- run store ------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs.jsonl")


async def test_run_creation_and_persistence(store: RunStore) -> None:
    run = AutonomousRun(objective="investigate X")
    await store.save(run)
    loaded = await store.get(run.id)
    assert loaded is not None
    assert loaded.objective == "investigate X"
    assert loaded.status is RunStatus.RUNNING
    assert loaded.current_stage is RunStage.INITIALIZE
    assert loaded.iteration == 0


async def test_store_persists_across_instances(tmp_path: Path) -> None:
    """A *new* store instance (fresh process equivalent) sees saved state."""
    first = RunStore(tmp_path / "runs.jsonl")
    run = AutonomousRun(objective="survives restart")
    await store_write(first, run)
    second = RunStore(tmp_path / "runs.jsonl")
    loaded = await second.get(run.id)
    assert loaded is not None
    assert loaded.objective == "survives restart"


async def store_write(store: RunStore, run: AutonomousRun) -> None:
    await store.save(run)


async def test_pause_resume_cancel_transitions(store: RunStore) -> None:
    run = AutonomousRun(objective="lifecycle")
    await store.save(run)

    paused = await store.pause(run.id)
    assert paused is not None and paused.status is RunStatus.PAUSED
    assert (await store.get(run.id)).status is RunStatus.PAUSED  # type: ignore[union-attr]

    resumed = await store.resume(run.id)
    assert resumed is not None and resumed.status is RunStatus.RUNNING

    cancelled = await store.cancel(run.id)
    assert cancelled is not None and cancelled.status is RunStatus.CANCELLED
    # terminal runs refuse further transitions
    assert await store.pause(run.id) is None
    assert await store.resume(run.id) is None


async def test_find_resumable_returns_paused_and_waiting(store: RunStore) -> None:
    a = AutonomousRun(objective="paused run")
    await store.save(a)
    await store.pause(a.id)
    b = AutonomousRun(objective="waiting run")
    await store.save(b)
    b.status = RunStatus.WAITING
    await store.save(b)
    c = AutonomousRun(objective="done run")
    c.status = RunStatus.COMPLETED
    await store.save(c)

    resumable = await store.find_resumable()
    ids = {r.id for r in resumable}
    assert ids == {a.id, b.id}


async def test_malformed_lines_are_skipped_not_fatal(store: RunStore, tmp_path: Path) -> None:
    run = AutonomousRun(objective="valid")
    await store.save(run)
    await asyncio.to_thread(_append_garbage, store._path)  # type: ignore[attr-defined]
    runs = await store.list()
    assert [r.id for r in runs] == [run.id]


def _append_garbage(path: Path) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json at all\n")


async def test_run_record_round_trips_through_json(store: RunStore) -> None:
    run = AutonomousRun(
        objective="json round trip",
        budgets=OrchestratorBudgets(max_iterations=7),
    )
    run.evidence.append(
        __import__(
            "agency.orchestrator.models", fromlist=["Evidence"]
        ).Evidence(source="s", source_type="web", content="c")
    )
    await store.save(run)
    loaded = await store.get(run.id)
    assert loaded is not None
    assert loaded.budgets.max_iterations == 7
    assert loaded.evidence[0].source_type == "web"
    # the on-disk form is one JSON object per line
    lines = (store._path).read_text(encoding="utf-8").strip().splitlines()  # type: ignore[attr-defined]
    assert all(isinstance(json.loads(line), dict) for line in lines)
