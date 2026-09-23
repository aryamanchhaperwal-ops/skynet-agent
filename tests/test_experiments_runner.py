"""Experiments: sandbox isolation, strategy registry, runner lifecycle."""

from __future__ import annotations

import asyncio
import json

import pytest

from agency.experiments.models import (
    Experiment,
    ExperimentStatus,
    Hypothesis,
    MetricDefinition,
    SuccessCriterion,
    TargetRef,
    Verdict,
)
from agency.experiments.registry import STRATEGY_CAPABILITIES, StrategyRegistry
from agency.experiments.runner import ExperimentRunner
from agency.experiments.sandbox import Sandbox, SandboxError, TrialContext


def _make_proc(score: float, cost: float):
    """Deterministic procedure returning configured metrics."""

    async def proc(ctx: TrialContext) -> dict[str, float]:
        del ctx
        return {"score": score, "cost": cost}

    return proc


#: Baseline returns 1.0/0.5; candidate returns 2.0/0.5 (+100% score).
_proc_baseline = _make_proc(1.0, 0.5)
_proc_candidate = _make_proc(2.0, 0.5)


def _registry(**procs: object) -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(
        name="strat", version="v1", capability="research_strategy",
        procedure=procs.get("baseline", _proc_baseline),  # type: ignore[arg-type]
        config={},
    )
    registry.register(
        name="strat", version="v2", capability="research_strategy",
        procedure=procs.get("candidate", _proc_candidate),  # type: ignore[arg-type]
        config={},
    )
    return registry


def _experiment(**overrides: object) -> Experiment:
    base: dict[str, object] = {
        "name": "runner-test",
        "hypothesis": Hypothesis(
            statement="v2 beats v1",
            success_criteria=[
                SuccessCriterion(kind="improves", metric="score", threshold=0.5)
            ],
        ),
        "baseline": TargetRef(name="strat", version="v1"),
        "candidate": TargetRef(name="strat", version="v2"),
        "metrics": [
            MetricDefinition(name="score", direction="maximize"),
            MetricDefinition(name="cost", direction="minimize"),
        ],
        "trials": 3,
        "seed": 123,
    }
    base.update(overrides)
    return Experiment(**base)  # type: ignore[arg-type]


# -- sandbox ---------------------------------------------------------------------


class TestSandbox:
    def test_context_exposes_only_injected_services(self) -> None:
        with Sandbox(experiment_id="abc") as sandbox:
            ctx = sandbox.context_for(
                arm="baseline", trial_index=0, seed=1,
                params={"x": 1}, services={"corpus": {"pages": []}},
            )
            assert ctx.service("corpus") == {"pages": []}
            with pytest.raises(SandboxError):
                ctx.service("memory_manager")

    def test_scratch_dir_scoped_and_removable(self) -> None:
        with Sandbox(experiment_id="abc") as sandbox:
            ctx = sandbox.context_for(
                arm="baseline", trial_index=0, seed=None, params={}
            )
            scratch = ctx.scratch_dir
            probe = scratch / "probe.txt"
            probe.write_text("data")
            assert probe.exists()
        assert not probe.exists()  # cleaned up on close

    def test_sandbox_without_scratch_refuses_scratch_access(self) -> None:
        sandbox = Sandbox(experiment_id="abc", with_scratch=False)
        ctx = sandbox.context_for(arm="baseline", trial_index=0, seed=None, params={})
        with pytest.raises(SandboxError):
            _ = ctx.scratch_dir

    def test_trial_context_cannot_reach_host_state(self) -> None:
        """Structural boundary: no settings/storage/memory on the context."""
        ctx = TrialContext(
            experiment_id="abc", arm="baseline", trial_index=0, seed=None, params={}
        )
        forbidden = ("settings", "storage", "memory", "registry", "environ")
        for name in forbidden:
            assert not hasattr(ctx, name)


# -- registry ---------------------------------------------------------------------


class TestStrategyRegistry:
    def test_register_and_get(self) -> None:
        registry = _registry()
        assert registry.get("strat:v1").name == "strat"
        assert registry.get("strat:v2").version == "v2"
        assert len(registry) == 2

    def test_unknown_ref_raises_with_known_list(self) -> None:
        registry = _registry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("strat:v99")

    def test_duplicate_rejected(self) -> None:
        registry = _registry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                name="strat", version="v1", capability="research_strategy",
                procedure=_proc_baseline,
            )

    def test_unknown_capability_rejected(self) -> None:
        registry = StrategyRegistry()
        with pytest.raises(ValueError, match="unknown capability"):
            registry.register(
                name="x", version="v1", capability="self_modifying_code",
                procedure=_proc_baseline,
            )

    def test_capability_vocabulary_is_closed(self) -> None:
        assert "planner_policy" in STRATEGY_CAPABILITIES
        assert "research_strategy" in STRATEGY_CAPABILITIES

    def test_describe_excludes_procedure(self) -> None:
        registry = _registry()
        described = registry.get("strat:v1").describe()
        assert "procedure" not in json.dumps(described)


# -- runner ------------------------------------------------------------------------


class TestRunner:
    async def test_successful_experiment(self) -> None:
        runner = ExperimentRunner(_registry())
        experiment = await runner.run(_experiment())
        assert experiment.status is ExperimentStatus.COMPLETED
        assert experiment.evaluation is not None
        assert experiment.evaluation.verdict is Verdict.SUCCESS
        assert len(experiment.trials_results) == 6
        assert experiment.comparison is not None
        assert experiment.comparison.baseline.trials == 3
        assert experiment.comparison.significance_tested is False

    async def test_unregistered_target_fails_before_trials(self) -> None:
        runner = ExperimentRunner(_registry())
        experiment = _experiment(
            candidate=TargetRef(name="strat", version="v99")
        )
        finished = await runner.run(experiment)
        assert finished.status is ExperimentStatus.ERROR
        assert "not registered" in (finished.error or "")
        assert finished.trials_results == []

    async def test_procedure_failure_marks_trials(self) -> None:
        async def boom(ctx: TrialContext) -> dict[str, float]:
            raise RuntimeError("procedure exploded")

        runner = ExperimentRunner(
            _registry(baseline=boom, candidate=boom)
        )
        experiment = await runner.run(_experiment())
        # Every trial failed → the evaluator says ERROR → status ERROR.
        assert experiment.status is ExperimentStatus.ERROR
        assert experiment.evaluation.verdict is Verdict.ERROR
        assert all(not t.success for t in experiment.trials_results)
        assert all("procedure exploded" in (t.error or "") for t in experiment.trials_results)

    async def test_trial_timeout_becomes_failed_trial(self) -> None:
        async def slow(ctx: TrialContext) -> dict[str, float]:
            await asyncio.sleep(5.0)
            return {"score": 1.0}

        runner = ExperimentRunner(_registry(baseline=slow))
        experiment = _experiment(trials=1, per_trial_timeout_seconds=0.2)
        finished = await runner.run(experiment)
        baseline_trials = [t for t in finished.trials_results if t.arm == "baseline"]
        assert all(not t.success for t in baseline_trials)
        assert all("timed out" in (t.error or "") for t in baseline_trials)
        # Baseline produced no usable metrics; candidate is fine → the
        # criteria are unmeasurable → INCONCLUSIVE (never a guessed verdict).
        assert finished.evaluation.verdict is Verdict.INCONCLUSIVE

    async def test_total_budget_cancels_experiment(self) -> None:
        async def slow(ctx: TrialContext) -> dict[str, float]:
            await asyncio.sleep(0.3)
            return {"score": 1.0}

        # Candidate needs 5 × 0.3 s = 1.5 s; the 1 s total budget runs out
        # between trials and the experiment is CANCELLED, not faked.
        runner = ExperimentRunner(_registry(candidate=slow))
        experiment = _experiment(trials=5, max_total_seconds=1.0)
        finished = await runner.run(experiment)
        assert finished.status is ExperimentStatus.CANCELLED
        assert "budget" in (finished.error or "")

    async def test_external_cancellation_is_recorded(self) -> None:
        """Cancelling the runner task mid-flight marks the record CANCELLED.

        The runner catches CancelledError, closes the sandbox, records the
        cancelled status, and returns the record (asyncio semantics: an
        exception-swallowing coroutine completes its task cleanly).
        """
        async def slow(ctx: TrialContext) -> dict[str, float]:
            await asyncio.sleep(30)
            return {"score": 1.0}

        runner = ExperimentRunner(_registry(baseline=slow))
        experiment = _experiment(trials=1, per_trial_timeout_seconds=25)
        task = asyncio.get_running_loop().create_task(runner.run(experiment))
        await asyncio.sleep(0.05)  # runner is now inside the slow trial
        task.cancel()
        finished = await task
        assert isinstance(finished, Experiment)
        assert finished.status is ExperimentStatus.CANCELLED

    async def test_seed_determinism_same_result(self) -> None:
        def make_proc(base: float):
            async def proc(ctx: TrialContext) -> dict[str, float]:
                del ctx
                return {"score": base}

            return proc

        async def run_once() -> float:
            registry = _registry(
                baseline=make_proc(1.0), candidate=make_proc(2.0)
            )
            finished = await ExperimentRunner(registry).run(_experiment(trials=2))
            return finished.comparison.candidate.mean["score"]  # type: ignore[union-attr]

        assert await run_once() == await run_once()

    async def test_undeclared_metrics_still_reported(self) -> None:
        async def extra(ctx: TrialContext) -> dict[str, float]:
            del ctx
            return {"score": 1.0, "bonus_metric": 7.0}

        runner = ExperimentRunner(_registry(candidate=extra))
        finished = await runner.run(_experiment(trials=1))
        assert "bonus_metric" in finished.comparison.candidate.mean  # type: ignore[union-attr]

    async def test_rerun_terminal_experiment_refused(self) -> None:
        runner = ExperimentRunner(_registry())
        experiment = await runner.run(_experiment())
        with pytest.raises(ValueError, match="already terminal"):
            await runner.run(experiment)

    async def test_trace_events_emitted(self) -> None:
        events: list[str] = []

        async def emit(event: str, **payload: object) -> None:
            events.append(event)

        runner = ExperimentRunner(_registry(), emit=emit)
        await runner.run(_experiment(trials=1))
        assert "EXPERIMENT_CREATED" in events
        assert "TRIAL_STARTED" in events
        assert "METRIC_RECORDED" in events
        assert "EXPERIMENT_EVALUATED" in events
