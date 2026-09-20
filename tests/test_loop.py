"""SkynetCore loop tests (memory backend — deterministic and DB-free)."""

from __future__ import annotations

import json
import time

import pytest

import agency.loop as agency_loop
from agency.actions.base import (
    ActionSpec,
)
from agency.bootstrap import build_core
from agency.config import SkynetSettings
from agency.evaluator import DeterministicEvaluator
from agency.goals import GoalSpec
from agency.loop import RunSummary, SkynetCore
from agency.planner import DeterministicPlanner, Planner
from agency.storage import MemoryStorage
from agency.trace import MemoryTraceSink
from tests.test_actions import BoomAction


def make_settings(**overrides) -> SkynetSettings:
    base = dict(
        storage_backend="memory",
        perception_adapter="none",
        max_run_seconds=30,
        max_steps_per_run=10,
    )
    base.update(overrides)
    return SkynetSettings(**base)


def make_core(settings: SkynetSettings | None = None, **kwargs) -> SkynetCore:
    return build_core(settings or make_settings(), **kwargs)


class ScriptedPlanner(DeterministicPlanner):
    """Planner whose plan comes from a fixed list of action specs."""

    def __init__(self, specs: list[ActionSpec]) -> None:
        super().__init__(default_action="")
        self._specs = specs
        self.name = "scripted"

    async def plan(self, goal, state):
        from agency.planner import Plan

        del state
        return Plan(steps=list(self._specs), rationale="scripted test plan")


class FlakyPlanner(Planner):
    """Planner that raises — the loop must survive planner crashes."""

    name = "flaky"

    async def plan(self, goal, state):
        del goal, state
        raise RuntimeError("planner exploded")


class SadEvaluator(DeterministicEvaluator):
    """Evaluator whose run evaluation raises — must not crash the loop."""

    name = "sad"

    async def evaluate_run(self, state):
        raise RuntimeError("evaluator exploded")


# ---------------------------------------------------------------- happy path


@pytest.mark.asyncio
async def test_happy_path_echo_run_completes(memory_settings) -> None:
    core = make_core(memory_settings)
    summary = await core.run_goal(
        GoalSpec(title="Say hello", params={"steps": [{"type": "echo", "params": {"text": "hello"}}]})
    )
    assert summary.ok is True
    assert summary.status == "completed"
    assert summary.steps_planned == 1 and summary.steps_executed == 1
    assert summary.actions[0]["type"] == "echo"
    assert summary.actions[0]["success"] is True
    assert summary.evaluation is not None and summary.evaluation.passed is True
    assert summary.observations == 0  # null perception
    assert summary.events >= 8  # RUN_STARTED..RUN_COMPLETED all present
    assert summary.experiences >= 3  # action + evaluation + summary experiences
    assert summary.error is None
    assert summary.final_state["status"] == "completed"
    assert summary.result["summary"].startswith("Run completed")


@pytest.mark.asyncio
async def test_goal_id_flow_and_summary_fields(memory_settings) -> None:
    core = make_core(memory_settings)
    summary = await core.run_goal(GoalSpec(title="Field check"))
    assert summary.goal_id is not None
    assert summary.goal_title == "Field check"
    assert summary.planner == "deterministic"
    assert summary.perception == "none"
    assert summary.started_at <= summary.finished_at


@pytest.mark.asyncio
async def test_events_cover_full_lifecycle(memory_settings) -> None:
    sink = MemoryTraceSink()
    core = make_core(memory_settings, extra_trace_sinks=(sink,))
    await core.run_goal(
        GoalSpec(title="Trace me", params={"steps": [{"type": "echo", "params": {"text": "x"}}]})
    )
    types = sink.types
    assert types[0] == "RUN_STARTED"
    assert "GOAL_CREATED" in types
    assert "PLAN_CREATED" in types
    assert "ACTION_SELECTED" in types
    assert "ACTION_STARTED" in types
    assert "ACTION_COMPLETED" in types
    assert "EVALUATION_COMPLETED" in types
    assert types[-1] == "RUN_COMPLETED"
    # seq is strictly increasing
    seqs = [event.seq for event in sink.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


# ------------------------------------------------------------ failure paths


@pytest.mark.asyncio
async def test_unknown_action_type_fails_gracefully(memory_settings) -> None:
    core = make_core(memory_settings, extra_actions=())
    # registry has echo only; request an unregistered action
    summary = await core.run_goal(
        GoalSpec(title="Ghost action", params={"steps": [{"type": "nonexistent_action"}]})
    )
    # Cycle integrity holds (no crash), quality fails.
    assert summary.status == "completed"
    assert summary.ok is False
    assert summary.actions[0]["success"] is False
    assert "unknown action type" in summary.actions[0]["error"]
    assert summary.evaluation is not None and summary.evaluation.passed is False


@pytest.mark.asyncio
async def test_allowlist_refusal(memory_settings) -> None:
    settings = make_settings(enabled_actions="")  # refuse everything
    core = make_core(settings)
    summary = await core.run_goal(
        GoalSpec(title="Forbidden", params={"steps": [{"type": "echo", "params": {"text": "hi"}}]})
    )
    assert summary.status == "completed"
    assert summary.ok is False
    assert "not permitted by configuration" in summary.actions[0]["error"]


@pytest.mark.asyncio
async def test_failing_action_yields_failed_evaluation_with_trace(memory_settings) -> None:
    sink = MemoryTraceSink()
    settings = memory_settings.model_copy(update={"enabled_actions": "echo,boom"})
    core = make_core(settings, extra_actions=(BoomAction(),), extra_trace_sinks=(sink,))
    summary = await core.run_goal(
        GoalSpec(title="Boom", params={"steps": [{"type": "boom"}]})
    )
    # Cycle completes; quality fails.
    assert summary.status == "completed"
    assert summary.ok is False
    assert summary.actions[0]["success"] is False
    assert "ActionError: kaboom" in summary.actions[0]["error"]
    assert "ACTION_FAILED" in sink.types
    state = summary.final_state
    assert state["errors"], "error must be recorded in agent state"
    assert state["actions"][0]["result"]["success"] is False


@pytest.mark.asyncio
async def test_planner_crash_is_caught_and_run_fails(memory_settings) -> None:
    core = make_core(memory_settings, planner=FlakyPlanner())
    summary = await core.run_goal(GoalSpec(title="Crash the planner"))
    assert summary.status == "failed"
    assert "planner exploded" in summary.error
    assert summary.final_state["status"] == "failed"
    assert summary.steps_planned == 0


@pytest.mark.asyncio
async def test_evaluator_crash_is_caught(memory_settings) -> None:
    core = make_core(
        memory_settings,
        planner=ScriptedPlanner([ActionSpec(type="echo", params={"text": "x"})]),
        evaluator=SadEvaluator(),
    )
    summary = await core.run_goal(GoalSpec(title="Sad evaluation"))
    # action succeeds; run evaluation crash -> generic failure path
    assert summary.status == "failed"
    assert "evaluator exploded" in summary.error


@pytest.mark.asyncio
async def test_step_budget_exhaustion(memory_settings) -> None:
    settings = make_settings(max_steps_per_run=2)
    core = make_core(
        settings,
        planner=ScriptedPlanner([ActionSpec(type="echo", params={"text": "1"}),
                                 ActionSpec(type="echo", params={"text": "2"}),
                                 ActionSpec(type="echo", params={"text": "3"})]),
    )
    summary = await core.run_goal(GoalSpec(title="Too many steps"))
    assert summary.status == "failed"
    assert "max_steps_per_run" in summary.error
    assert summary.steps_executed == 2
    assert summary.ok is False


@pytest.mark.asyncio
async def test_time_budget_exhaustion(memory_settings, monkeypatch) -> None:
    settings = memory_settings.model_copy(update={"max_run_seconds": 60})
    core = make_core(
        settings,
        planner=ScriptedPlanner([ActionSpec(type="echo", params={"text": "x"})]),
    )
    # The FIRST monotonic() call inside the loop computes the deadline; make
    # it land far in the past so every later (real) call is past-deadline.
    real_monotonic = time.monotonic
    calls = {"n": 0}

    def fake_monotonic() -> float:
        calls["n"] += 1
        return real_monotonic() - 3600 if calls["n"] == 1 else real_monotonic()

    monkeypatch.setattr(agency_loop.time, "monotonic", fake_monotonic)
    summary = await core.run_goal(GoalSpec(title="No time"))
    # The deadline check fires before the first step executes.
    assert summary.status == "failed"
    assert "max_run_seconds" in summary.error


@pytest.mark.asyncio
async def test_perception_failure_degrades_to_error_observation(memory_settings) -> None:
    class ExplodingPerception:
        name = "exploding"

        async def observe(self, state):
            raise RuntimeError("sensor explosion")

    core = make_core(memory_settings)
    core._perception = ExplodingPerception()
    summary = await core.run_goal(GoalSpec(title="Blind run"))
    # The run still completes: sensors failing must not kill the loop.
    assert summary.status == "completed"
    assert summary.observations == 0
    assert any("perception failed" in e for e in summary.final_state["errors"])


@pytest.mark.asyncio
async def test_experience_sink_failure_does_not_prevent_summary(memory_settings) -> None:
    class FailingSink:
        async def save(self, records):
            raise RuntimeError("experience sink down")

    from agency.experience import ExperienceRecorder

    core = make_core(memory_settings)
    core._experiences = ExperienceRecorder(FailingSink())
    summary = await core.run_goal(
        GoalSpec(title="Sink fire", params={"steps": [{"type": "echo", "params": {"text": "x"}}]})
    )
    assert summary.status == "completed"  # run survives; flush failure logged


@pytest.mark.asyncio
async def test_state_error_recorded_on_action_failure(memory_settings) -> None:
    core = make_core(memory_settings, extra_actions=(BoomAction(),))
    summary = await core.run_goal(GoalSpec(title="Err state", params={"steps": [{"type": "boom"}]}))
    assert any("action boom failed" in e for e in summary.final_state["errors"])


@pytest.mark.asyncio
async def test_storage_finish_failure_does_not_crash_loop(memory_settings) -> None:
    class FlakyStorage(MemoryStorage):
        async def finish_run(self, run_id, *, status, result, error, steps_executed):
            raise RuntimeError("storage down at the worst moment")

    core = make_core(memory_settings, storage=FlakyStorage())
    summary = await core.run_goal(
        GoalSpec(title="Storage fire", params={"steps": [{"type": "echo", "params": {"text": "x"}}]})
    )
    # Run completes and summarizes even though final persistence failed.
    assert summary.status == "completed"
    assert summary.run_id


# ------------------------------------------------------------ determinism


@pytest.mark.asyncio
async def test_deterministic_default_uses_gods_eye_plan(memory_settings) -> None:
    core = make_core(memory_settings)
    summary = await core.run_goal(GoalSpec(title="Default plan"))
    assert summary.steps_planned == 1
    # The default plan requests the God's Eye fetch, which the memory
    # backend never registered — proving the unknown-action refusal path.
    assert summary.actions[0]["type"] == "gods_eye_latest_events"
    assert summary.actions[0]["success"] is False
    assert "unknown action type" in summary.actions[0]["error"]
    # Goal is marked failed in storage even though the cycle completed:
    # goal status reflects outcome quality, run status reflects cycle integrity.
    goal = await core._goals.get(summary.goal_id)
    assert goal.status == "failed"


@pytest.mark.asyncio
async def test_echo_action_real_execution_path(memory_settings) -> None:
    core = make_core(memory_settings)
    summary = await core.run_goal(
        GoalSpec(title="Echo", params={"steps": [{"type": "echo", "params": {"text": "ping"}}]})
    )
    assert summary.result["actions"][0]["success"] is True
    # The experience recorder buffered + flushed through the in-memory sink.


def test_run_summary_is_json_serializable(memory_settings) -> None:
    summary = RunSummary(run_id="r", goal_title="t", status="completed")
    payload = json.dumps(summary.model_dump(mode="json"))
    assert "r" in payload
