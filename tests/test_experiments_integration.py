"""Experiments: actions, bootstrap gate, loop integration, CLI, security."""

from __future__ import annotations

import json

import pytest

from agency.actions.base import ActionContext, ActionError, ActionSpec
from agency.experiments.experiments_store import ExperimentRegistry
from agency.experiments.models import (
    Experiment,
    ExperimentStatus,
    Hypothesis,
    TargetRef,
    Verdict,
)
from agency.experiments.registry import StrategyRegistry
from agency.experiments.runner import ExperimentRunner
from agency.experiments.sandbox import TrialContext


async def _baseline(ctx: TrialContext) -> dict[str, float]:
    del ctx
    return {"score": 1.0}


async def _candidate(ctx: TrialContext) -> dict[str, float]:
    del ctx
    return {"score": 2.0}


def _registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(
        name="strat", version="v1", capability="research_strategy",
        procedure=_baseline,
    )
    registry.register(
        name="strat", version="v2", capability="research_strategy",
        procedure=_candidate,
    )
    return registry


def make_action_ctx(**overrides: object) -> ActionContext:
    from agency.state import AgentState

    defaults: dict[str, object] = {
        "run_id": "run-test-123",
        "goal_id": None,
        "state_snapshot": AgentState(run_id="run-test-123").model_dump(),
        "tracer": None,
    }
    defaults.update(overrides)
    return ActionContext(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def lab_env(tmp_path):
    from agency.experiments.actions import (
        ExperimentInspectAction,
        ExperimentListAction,
        ExperimentRunAction,
    )

    strategies = _registry()
    store = ExperimentRegistry(tmp_path / "experiments.jsonl")
    runner = ExperimentRunner(strategies)
    return {
        "run": ExperimentRunAction(runner, store),
        "list": ExperimentListAction(store),
        "inspect": ExperimentInspectAction(store),
        "store": store,
        "strategies": strategies,
    }


# -- actions -------------------------------------------------------------------


class TestExperimentRunAction:
    async def test_run_returns_full_report(self, lab_env) -> None:
        spec = ActionSpec(
            type="experiment_run",
            params={
                "name": "action-test",
                "hypothesis": "v2 doubles score",
                "baseline": "strat:v1",
                "candidate": "strat:v2",
                "metrics": [{"name": "score", "direction": "maximize"}],
                "success_criteria": [
                    {"kind": "improves", "metric": "score", "threshold": 0.5}
                ],
                "trials": 3,
                "seed": 7,
            },
        )
        output = await lab_env["run"].execute(spec, make_action_ctx())
        assert output["ok"] is True
        assert output["status"] == "completed"
        assert output["verdict"] == "success"
        assert output["significance_tested"] is False
        assert output["delta"]["score"] == 1.0
        # Persisted and inspectable.
        record = await lab_env["store"].get(output["experiment_id"])
        assert record is not None
        assert record.status is ExperimentStatus.COMPLETED

    async def test_run_persists_created_and_finished(self, lab_env) -> None:
        spec = ActionSpec(
            type="experiment_run",
            params={
                "name": "persist-check",
                "hypothesis": "v2 wins",
                "baseline": "strat:v1",
                "candidate": "strat:v2",
                "success_criteria": [
                    {"kind": "improves", "metric": "score", "threshold": 0.5}
                ],
                "trials": 1,
            },
        )
        await lab_env["run"].execute(spec, make_action_ctx())
        listed = await lab_env["list"].execute(
            ActionSpec(type="experiment_list", params={}), make_action_ctx()
        )
        assert listed["count"] >= 1

    async def test_missing_criteria_refused(self, lab_env) -> None:
        spec = ActionSpec(
            type="experiment_run",
            params={
                "name": "no-criteria",
                "hypothesis": "v2 wins",
                "baseline": "strat:v1",
                "candidate": "strat:v2",
            },
        )
        with pytest.raises(ActionError, match="success_criteria"):
            await lab_env["run"].execute(spec, make_action_ctx())

    async def test_unknown_strategy_refused(self, lab_env) -> None:
        spec = ActionSpec(
            type="experiment_run",
            params={
                "name": "bad-target",
                "hypothesis": "v9 wins",
                "baseline": "strat:v1",
                "candidate": "strat:v9",
                "success_criteria": [
                    {"kind": "improves", "metric": "score", "threshold": 0.5}
                ],
            },
        )
        output = await lab_env["run"].execute(spec, make_action_ctx())
        assert output["status"] == "error"
        assert "not registered" in output["explanation"]

    async def test_invalid_params_refused(self, lab_env) -> None:
        base = {
            "name": "invalid",
            "hypothesis": "x",
            "baseline": "strat:v1",
            "candidate": "strat:v2",
            "success_criteria": [{"kind": "improves", "metric": "s", "threshold": 0.1}],
        }
        bad_targets = [
            {**base, "baseline": "no-version"},
            {**base, "trials": 0},
            {**base, "trials": "three"},
            {**base, "seed": "not-an-int"},
            {**base, "procedure_params": "not-an-object"},
            {**base, "success_criteria": [{"kind": "bogus", "metric": "s"}]},
        ]
        for params in bad_targets:
            with pytest.raises(ActionError):
                await lab_env["run"].execute(
                    ActionSpec(type="experiment_run", params=params), make_action_ctx()
                )


class TestListInspectActions:
    async def test_inspect_unknown_id_is_action_error(self, lab_env) -> None:
        with pytest.raises(ActionError, match="unknown experiment"):
            await lab_env["inspect"].execute(
                ActionSpec(
                    type="experiment_inspect", params={"experiment_id": "nope"}
                ),
                make_action_ctx(),
            )

    async def test_inspect_returns_record_without_procedure(self, lab_env) -> None:
        spec = ActionSpec(
            type="experiment_run",
            params={
                "name": "inspect-me",
                "hypothesis": "v2 wins",
                "baseline": "strat:v1",
                "candidate": "strat:v2",
                "success_criteria": [
                    {"kind": "improves", "metric": "score", "threshold": 0.5}
                ],
                "trials": 1,
            },
        )
        output = await lab_env["run"].execute(spec, make_action_ctx())
        inspected = await lab_env["inspect"].execute(
            ActionSpec(
                type="experiment_inspect",
                params={"experiment_id": output["experiment_id"]},
            ),
            make_action_ctx(),
        )
        record = inspected["experiment"]
        assert record["hypothesis"]["statement"] == "v2 wins"
        assert record["comparison"]["baseline"]["ref"] == "strat:v1"
        # No executable procedure ever serializes ("procedure_params" — the
        # data params — is fine; a "procedure" code key must not exist).
        assert '"procedure":' not in json.dumps(inspected)
        assert '"procedure"' not in json.dumps(inspected).replace(
            '"procedure_params"', ''
        )


# -- bootstrap / gate -------------------------------------------------------------


class TestBootstrapGate:
    def test_lab_actions_dark_by_default(self, memory_settings) -> None:
        from agency.bootstrap import build_core

        core = build_core(memory_settings)
        assert "experiment_run" not in core._actions
        assert "experiment_list" not in core._actions

    def test_lab_actions_registered_when_flag_on(self, memory_settings) -> None:
        from agency.bootstrap import build_core

        settings = memory_settings.model_copy(
            update={"enable_self_improvement": True}
        )
        core = build_core(settings)
        assert "experiment_run" in core._actions
        assert "experiment_list" in core._actions
        assert "experiment_inspect" in core._actions

    def test_second_gate_still_blocks(self, memory_settings) -> None:
        """Registration ≠ permission: the allow-list still refuses at selection."""
        settings = memory_settings.model_copy(
            update={
                "enable_self_improvement": True,
                "enabled_actions": "echo",  # lab actions NOT allow-listed
            }
        )
        assert "experiment_run" not in settings.action_allowlist


# -- loop integration ----------------------------------------------------------------


class TestLoopIntegration:
    async def test_loop_runs_experiment_action_end_to_end(self, memory_settings, tmp_path) -> None:
        from agency.bootstrap import build_core
        from agency.goals import GoalSpec

        settings = memory_settings.model_copy(
            update={
                "enable_self_improvement": True,
                "enabled_actions": "echo,experiment_run",
                "max_steps_per_run": 2,
                # 2 trials/arm < MIN_TRIALS_FOR_VERDICT → SUCCESS downgrades
                # to INCONCLUSIVE by design; assert that contract instead.
                "experiments_registry_path": str(tmp_path / "experiments.jsonl"),
            }
        )
        strategies = _registry()
        core = build_core(settings)
        # Register the lab runner's strategy set into the core's action.
        action = core._actions.get("experiment_run")
        action._runner._registry = strategies
        spec = {
            "type": "experiment_run",
            "params": {
                "name": "loop-test",
                "hypothesis": "v2 doubles score",
                "baseline": "strat:v1",
                "candidate": "strat:v2",
                "success_criteria": [
                    {"kind": "improves", "metric": "score", "threshold": 0.5}
                ],
                "trials": 2,
            },
        }
        summary = await core.run_goal(
            GoalSpec(
                title="Run one experiment",
                params={"steps": [spec]},
            )
        )
        assert summary.status == "completed"
        records = [
            r for r in summary.final_state["actions"]
            if r["spec"]["type"] == "experiment_run"
        ]
        assert records and records[0]["result"]["success"]
        output = records[0]["result"]["output"]
        # 2 trials/arm: criteria pass but the verdict is (correctly)
        # downgraded to INCONCLUSIVE — a small sample is not evidence.
        assert output["verdict"] == "inconclusive"
        assert any(
            c["outcome"] == "passed" for c in output["criteria_results"]
        )


# -- CLI --------------------------------------------------------------------------


class TestCLI:
    def test_cli_demo_end_to_end(self, tmp_path, monkeypatch, capsys) -> None:
        from agency.cli import main

        monkeypatch.chdir(tmp_path)  # isolated experiments.jsonl + memory
        exit_code = main(["experiment", "demo", "--json"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "completed"
        assert payload["evaluation"]["verdict"] in {
            "success", "failure", "inconclusive"
        }
        assert payload["comparison"]["significance_tested"] is False
        assert payload["trials_results"]

    def test_cli_list_shows_history(self, tmp_path, monkeypatch, capsys) -> None:
        from agency.cli import main

        monkeypatch.chdir(tmp_path)
        assert main(["experiment", "demo", "--json"]) == 0
        capsys.readouterr()
        assert main(["experiment", "list", "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert len(listed) == 1
        assert listed[0]["name"] == "research-strategy-demo"

    def test_cli_run_with_custom_criteria(self, tmp_path, monkeypatch, capsys) -> None:
        from agency.cli import main

        monkeypatch.chdir(tmp_path)
        exit_code = main(
            [
                "experiment", "run",
                "--name", "cli-run",
                "--hypothesis", "multi-query finds more sources",
                "--baseline", "single_query:v1",
                "--candidate", "multi_query:v1",
                "--criterion",
                '{"kind": "improves", "metric": "sources_found", "threshold": 0.1}',
                "--metric", "sources_found:maximize",
                "--trials", "2",
                "--seed", "3",
                "--param", "goal=long-term memory for ai agents",
                "--json",
            ]
        )
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "cli-run"
        assert payload["evaluation"] is not None


# -- security ---------------------------------------------------------------------


class TestSecurityBoundary:
    def test_capability_vocabulary_excludes_code_execution(self) -> None:
        from agency.experiments.registry import STRATEGY_CAPABILITIES

        forbidden = {"code_edit", "source_modification", "shell", "filesystem"}
        assert forbidden & STRATEGY_CAPABILITIES == set()

    async def test_experiment_cannot_read_secrets_via_context(self) -> None:
        from agency.experiments.sandbox import SandboxError

        captured: dict[str, object] = {}

        async def snoop(ctx: TrialContext) -> dict[str, float]:
            captured["environ"] = hasattr(ctx, "environ")
            captured["settings"] = hasattr(ctx, "settings")
            captured["memory"] = hasattr(ctx, "memory")
            try:
                ctx.service("OPENAI_API_KEY")
                captured["service_key"] = "LEAKED"
            except SandboxError:
                captured["service_key"] = "refused"
            return {"score": 1.0}

        registry = StrategyRegistry()
        registry.register(
            name="snoop", version="v1",
            capability="tool_config", procedure=snoop,
        )
        registry.register(
            name="snoop", version="v2",
            capability="tool_config", procedure=snoop,
        )
        experiment = Experiment(
            name="security-probe",
            hypothesis=Hypothesis(statement="probe"),
            baseline=TargetRef(name="snoop", version="v1"),
            candidate=TargetRef(name="snoop", version="v2"),
            trials=1,
        )
        await ExperimentRunner(registry).run(experiment)
        assert captured["environ"] is False
        assert captured["settings"] is False
        assert captured["memory"] is False
        assert captured["service_key"] == "refused"

    async def test_procedure_returning_non_metrics_is_trial_failure(self) -> None:
        async def evil(ctx: TrialContext) -> dict[str, float]:
            del ctx
            return {"payload": "not-a-number"}  # type: ignore[dict-item]

        registry = StrategyRegistry()
        registry.register(
            name="evil", version="v1", capability="tool_config", procedure=evil
        )
        registry.register(
            name="evil", version="v2", capability="tool_config", procedure=evil
        )
        experiment = Experiment(
            name="type-probe",
            hypothesis=Hypothesis(statement="probe"),
            baseline=TargetRef(name="evil", version="v1"),
            candidate=TargetRef(name="evil", version="v2"),
            trials=1,
        )
        finished = await ExperimentRunner(registry).run(experiment)
        assert all(not t.success for t in finished.trials_results)
        assert finished.evaluation.verdict is Verdict.ERROR
