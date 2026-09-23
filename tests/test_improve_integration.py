"""Self-improvement integration: bootstrap gate, actions, CLI, demo."""

from __future__ import annotations

from typing import Any

import pytest

from agency.actions.base import ActionContext, ActionSpec
from agency.config import SkynetSettings


def make_action_ctx(run_id: str = "run-1", goal_id: str | None = None) -> ActionContext:
    return ActionContext(run_id=run_id, goal_id=goal_id)


def _lab_settings(**overrides: Any) -> SkynetSettings:
    updates = {
        "enable_self_improvement": True,
        "improvements_registry_path": "data/improvements.jsonl",
        **overrides,
    }
    return SkynetSettings(**updates)


# -- Bootstrap gate ---------------------------------------------------------------------


class TestBootstrapGate:
    def test_improve_actions_registered_when_enabled(self, memory_settings: Any) -> None:
        from agency.bootstrap import build_core

        settings = memory_settings.model_copy(update={"enable_self_improvement": True})
        core = build_core(settings)
        for expected in ("improve_propose", "improve_inspect", "improve_history"):
            assert expected in core._actions, f"{expected} not registered"
        assert "improve_apply" not in core._actions  # never an autonomous action

    def test_improve_actions_absent_when_dark(self, memory_settings: Any) -> None:
        from agency.bootstrap import build_core

        core = build_core(memory_settings)
        assert "improve_propose" not in core._actions
        assert "experiment_run" not in core._actions


# -- Actions --------------------------------------------------------------------------------


class TestImproveActions:
    def _pipeline(self, tmp_path: Any):
        from agency.experiments.registry import StrategyRegistry
        from agency.improve.actions import (
            ImproveHistoryAction,
            ImproveInspectAction,
            ImproveProposeAction,
        )
        from agency.improve.pipeline import ImprovementPipeline
        from agency.improve.store import ImprovementStore
        from tests.test_improve_pipeline import _FakeExperimentStore, _FakeRunner, _proc

        registry = StrategyRegistry()
        registry.register(
            name="research",
            version="v1",
            capability="research_strategy",
            procedure=_proc,
            config={"cap": 10},
        )
        experiments = _FakeExperimentStore()
        pipeline = ImprovementPipeline(
            strategies=registry,
            runner=_FakeRunner(registry, experiments),
            experiments=experiments,
            store=ImprovementStore(path=tmp_path / "imp.jsonl"),
        )
        return (
            pipeline,
            ImproveProposeAction(pipeline),
            ImproveInspectAction(pipeline),
            ImproveHistoryAction(pipeline),
        )

    @pytest.mark.asyncio
    async def test_propose_action_flow(self, tmp_path: Any) -> None:
        from agency.improve.models import EvidenceItem, Weakness, WeaknessCategory

        pipeline, propose, inspect, history = self._pipeline(tmp_path)
        weakness = Weakness(
            category=WeaknessCategory.DUPLICATE_SOURCES,
            description="d",
            evidence=[EvidenceItem(summary="s")],
            affected_strategy="research:v1",
        )
        await pipeline.record_weakness(weakness)

        result = await propose.execute(
            ActionSpec(
                type="improve_propose",
                params={
                    "weakness_id": weakness.id,
                    "origin": "external",
                    "candidate_config": {"cap": 5},
                },
            ),
            make_action_ctx(),
        )
        assert result["ok"] is True
        assert result["candidate_ref"].startswith("research:cand-")

        listing = await inspect.execute(
            ActionSpec(type="improve_inspect", params={"view": "proposals"}),
            make_action_ctx(),
        )
        assert listing["count"] == 1
        assert listing["items"][0]["status"] == "proposed"

        versions_view = await inspect.execute(
            ActionSpec(type="improve_inspect", params={"view": "versions", "name": "research"}),
            make_action_ctx(),
        )
        assert versions_view["items"] == []

        prior = await history.execute(
            ActionSpec(type="improve_history", params={"weakness_id": weakness.id}),
            make_action_ctx(),
        )
        assert prior["count"] == 1

    @pytest.mark.asyncio
    async def test_propose_rejects_bad_params(self, tmp_path: Any) -> None:
        from agency.actions.base import ActionError

        _pipeline, propose, _inspect, _history = self._pipeline(tmp_path)
        with pytest.raises(ActionError):
            await propose.execute(
                ActionSpec(type="improve_propose", params={}), make_action_ctx()
            )
        with pytest.raises((ActionError, KeyError)):
            await propose.execute(
                ActionSpec(
                    type="improve_propose",
                    params={"weakness_id": "nope"},
                ),
                make_action_ctx(),
            )

    @pytest.mark.asyncio
    async def test_inspect_rejects_unknown_view(self, tmp_path: Any) -> None:
        from agency.actions.base import ActionError

        _pipeline, _propose, inspect, _history = self._pipeline(tmp_path)
        with pytest.raises(ActionError):
            await inspect.execute(
                ActionSpec(type="improve_inspect", params={"view": "everything"}),
                make_action_ctx(),
            )


# -- CLI --------------------------------------------------------------------------------------


class TestImproveCli:
    def test_history_empty(self, capsys: Any, monkeypatch: Any, tmp_path: Any) -> None:
        from agency.cli import main

        monkeypatch.setenv("SKYNET_IMPROVEMENTS_REGISTRY_PATH", str(tmp_path / "imp.jsonl"))
        code = main(["improve", "history"])
        assert code == 0
        assert "no improvement proposals" in capsys.readouterr().out

    def test_detect_lists_weaknesses(
        self, capsys: Any, monkeypatch: Any, tmp_path: Any
    ) -> None:
        import asyncio

        from agency.cli import main
        from agency.improve.models import EvidenceItem, Weakness, WeaknessCategory
        from agency.improve.store import ImprovementStore

        path = tmp_path / "imp.jsonl"
        monkeypatch.setenv("SKYNET_IMPROVEMENTS_REGISTRY_PATH", str(path))
        store = ImprovementStore(path=path)
        asyncio.run(
            store.save_weakness(
                Weakness(
                    category=WeaknessCategory.DUPLICATE_SOURCES,
                    description="dup weakness from test",
                    evidence=[EvidenceItem(summary="rate=0.7")],
                    affected_strategy="research:v1",
                )
            )
        )
        code = main(["improve", "detect"])
        assert code == 0
        out = capsys.readouterr().out
        assert "duplicate_sources" in out

    def test_demo_runs_end_to_end(
        self, capsys: Any, monkeypatch: Any, tmp_path: Any
    ) -> None:
        from agency.cli import main

        monkeypatch.setenv("SKYNET_IMPROVEMENTS_REGISTRY_PATH", str(tmp_path / "imp.jsonl"))
        monkeypatch.setenv("SKYNET_EXPERIMENTS_REGISTRY_PATH", str(tmp_path / "exp.jsonl"))
        code = main(["improve", "demo"])
        assert code == 0
        out = capsys.readouterr().out
        assert "detected" in out
        assert "evaluated" in out

    def test_demo_rejection_is_recorded(
        self, capsys: Any, monkeypatch: Any, tmp_path: Any
    ) -> None:
        import json as _json

        from agency.cli import main
        from agency.improve.store import ImprovementStore

        base = tmp_path / "imp.jsonl"
        monkeypatch.setenv("SKYNET_IMPROVEMENTS_REGISTRY_PATH", str(base))
        monkeypatch.setenv("SKYNET_EXPERIMENTS_REGISTRY_PATH", str(tmp_path / "exp.jsonl"))
        main(["improve", "demo"])
        capsys.readouterr()
        # Demo history lands in the derived demo file (re-runnable, but the
        # same store format); the bad candidate is recorded there as rejected.
        demo_store = ImprovementStore(path=base.with_name("imp.demo.jsonl"))
        rejected = [p for p in demo_store.proposals() if p.status.value == "rejected"]
        assert rejected, "the deliberately bad candidate must be recorded"
        assert any(
            "cap each query at 1" in (p.proposed_change or "") for p in rejected
        )
        assert _json.dumps([p.decision for p in rejected]) is not None
        code = main(["improve", "history"])
        assert code == 0
        assert "no improvement proposals" in capsys.readouterr().out
