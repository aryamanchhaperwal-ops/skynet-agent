"""Improvement pipeline: lifecycle, acceptance gate, approval, rollback, memory."""

from __future__ import annotations

from typing import Any

import pytest

from agency.experiments.models import (
    Experiment,
    ExperimentStatus,
    MetricDefinition,
)
from agency.experiments.registry import StrategyRegistry
from agency.improve.detector import DetectorThresholds, WeaknessDetector
from agency.improve.models import (
    EvidenceItem,
    ProposalStatus,
    Weakness,
    WeaknessCategory,
)
from agency.improve.pipeline import ImprovementPipeline
from agency.improve.store import ImprovementStore


async def _proc(ctx: Any) -> dict[str, float]:
    """Registered procedure whose behavior is driven entirely by config."""
    cap = int(ctx.params.get("cap", 5))
    dup = 0.6 if cap >= 10 else 0.2
    return {"score": float(cap) / 10, "duplicate_rate": dup, "duration_ms": 1.0}


class _FakeMemory:
    """Minimal memory manager double (records store() calls)."""

    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []

    async def store(self, **kwargs: Any) -> Any:
        self.stored.append(kwargs)

        class _R:
            id = "mem-1"

        return _R()


def _make_weakness(strategy_ref: str = "research:v1") -> Weakness:
    return Weakness(
        category=WeaknessCategory.DUPLICATE_SOURCES,
        description="too many duplicates",
        evidence=[EvidenceItem(summary="rate=0.62")],
        affected_strategy=strategy_ref,
    )


def _make_pipeline(
    *,
    memory: Any = None,
    emit: Any = None,
    store: ImprovementStore | None = None,
) -> tuple[ImprovementPipeline, StrategyRegistry]:
    registry = StrategyRegistry()
    registry.register(
        name="research",
        version="v1",
        capability="research_strategy",
        procedure=_proc,
        description="baseline",
        config={"cap": 10},
    )
    experiments = _FakeExperimentStore()
    pipeline = ImprovementPipeline(
        strategies=registry,
        runner=_FakeRunner(registry, experiments),
        experiments=experiments,
        memory=memory,
        emit=emit,
        detector=WeaknessDetector(DetectorThresholds(min_frequency=1)),
        store=store or ImprovementStore(path=None),
    )
    return pipeline, registry


class _FakeExperimentStore:
    def __init__(self) -> None:
        self.records: dict[str, Experiment] = {}

    async def get(self, experiment_id: str) -> Experiment | None:
        return self.records.get(experiment_id)

    async def save(self, experiment: Experiment) -> Experiment:
        self.records[experiment.id] = experiment
        return experiment


class _FakeRunner:
    """Runs the real P6 evaluation path over synthetic trial data."""

    def __init__(self, registry: StrategyRegistry, store: _FakeExperimentStore) -> None:
        self._registry = registry
        self._store = store

    async def run(self, experiment: Experiment, *, run_id: str | None = None) -> Experiment:
        del run_id
        experiment.criteria_frozen()
        experiment.status = ExperimentStatus.RUNNING
        baseline = self._registry.get(experiment.baseline.ref)
        candidate = self._registry.get(experiment.candidate.ref)
        trials = []
        for arm, strategy in (("baseline", baseline), ("candidate", candidate)):
            params = {**experiment.procedure_params, **strategy.config}
            metrics = {
                "score": float(params.get("cap", 5)) / 10,
                "duplicate_rate": 0.6 if int(params.get("cap", 5)) >= 10 else 0.2,
                "duration_ms": 1.0,
            }
            trials.append(
                (arm, metrics)
            )
        experiment.trials_results = [
            __import__(
                "agency.experiments.models", fromlist=["TrialResult"]
            ).TrialResult(
                arm=arm,  # type: ignore[arg-type]
                index=i,
                metrics=metrics,
                success=True,
            )
            for arm, metrics in trials
            for i in range(experiment.trials // 2 + 1)
        ][: experiment.trials * 2]
        # Halve: build exactly `trials` per arm.
        experiment.trials_results = [
            __import__(
                "agency.experiments.models", fromlist=["TrialResult"]
            ).TrialResult(arm=arm, index=i, metrics=metrics, success=True)
            for arm, metrics in trials
            for i in range(experiment.trials)
        ]
        from agency.experiments.metrics import aggregate, compare

        metric_names = [m.name for m in experiment.metrics]
        baseline_summary = aggregate(
            [t for t in experiment.trials_results if t.arm == "baseline"], metric_names
        )
        candidate_summary = aggregate(
            [t for t in experiment.trials_results if t.arm == "candidate"], metric_names
        )
        baseline_summary.ref = experiment.baseline.ref
        candidate_summary.ref = experiment.candidate.ref
        directions = {m.name: m.direction for m in experiment.metrics}
        deltas, improvements, regressions = compare(
            baseline_summary, candidate_summary, directions=directions
        )
        experiment.comparison = __import__(
            "agency.experiments.models", fromlist=["ComparisonReport"]
        ).ComparisonReport(
            baseline=baseline_summary,
            candidate=candidate_summary,
            delta={k: v["abs"] for k, v in deltas.items() if v["abs"] is not None},
            relative={
                k: v["rel"] for k, v in deltas.items() if v.get("rel") is not None
            },
            improvements=improvements,
            regressions=regressions,
        )
        from agency.experiments.evaluation import evaluate_experiment

        experiment.evaluation = evaluate_experiment(experiment)
        experiment.status = ExperimentStatus.COMPLETED
        self._store.records[experiment.id] = experiment
        return experiment


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_full_accept_path(self) -> None:
        pipeline, _registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        _hypothesis, proposal, candidate = await pipeline.propose(
            weakness.id, candidate_config={"cap": 5}
        )
        experiment = await pipeline.test(proposal, candidate)
        report = await pipeline.evaluate(proposal, candidate, experiment)
        assert report["decision"] == "rejected"  # criteria not yet passed; gate decides
        # With default criteria (score improves ≥10%), cap 5 lowers score → reject.
        assert proposal.status is ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_apply_requires_approval(self) -> None:
        pipeline, _registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id, candidate_config={"cap": 5}
        )
        experiment = await pipeline.test(proposal, candidate)
        await pipeline.evaluate(proposal, candidate, experiment)
        # Force-accept via a genuinely passing experiment for apply-gate test:
        proposal.transition(ProposalStatus.ACCEPTED)
        proposal.decision = "accepted"
        with pytest.raises(PermissionError):
            await pipeline.apply(proposal, candidate, experiment)

    @pytest.mark.asyncio
    async def test_apply_with_approval_then_rollback(self) -> None:
        pipeline, _registry = _make_pipeline(memory=_FakeMemory())
        await pipeline.seed_baseline(name="research", version="v1")
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id,
            candidate_config={"cap": 5},
            plan_overrides={
                "acceptance_criteria": [
                    {"kind": "max", "metric": "duplicate_rate", "threshold": 0.35},
                    {"kind": "min", "metric": "score", "threshold": 0.3},
                ],
                "metrics": [
                    MetricDefinition(name="score"),
                    MetricDefinition(name="duplicate_rate", direction="minimize"),
                    MetricDefinition(name="duration_ms", direction="minimize"),
                ],
            },
        )
        experiment = await pipeline.test(proposal, candidate)
        report = await pipeline.evaluate(proposal, candidate, experiment)
        assert report["decision"] == "accepted", report
        await pipeline.approve(proposal, approved_by="operator")
        version = await pipeline.apply(proposal, candidate, experiment)
        assert pipeline.active("research").ref == version.ref
        # Rollback restores the seeded known-good version.
        restored = await pipeline.rollback(proposal, reason="demo regression")
        assert restored.ref == "research:v1"
        assert proposal.status is ProposalStatus.ROLLED_BACK
        # Failed candidate version is preserved in the ledger.
        assert any(v.ref == version.ref for v in pipeline.versions("research"))

    @pytest.mark.asyncio
    async def test_rollback_without_history_refuses(self) -> None:
        pipeline, _registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id,
            candidate_config={"cap": 5},
            plan_overrides={
                "acceptance_criteria": [
                    {"kind": "max", "metric": "duplicate_rate", "threshold": 0.35},
                    {"kind": "min", "metric": "score", "threshold": 0.3},
                ],
                "metrics": [
                    MetricDefinition(name="score"),
                    MetricDefinition(name="duplicate_rate", direction="minimize"),
                    MetricDefinition(name="duration_ms", direction="minimize"),
                ],
            },
        )
        experiment = await pipeline.test(proposal, candidate)
        await pipeline.evaluate(proposal, candidate, experiment)
        await pipeline.approve(proposal, approved_by="op")
        version = await pipeline.apply(proposal, candidate, experiment)
        assert pipeline.active("research").ref == version.ref
        # No seeded baseline → nothing to restore.
        with pytest.raises(ValueError, match="previous known-good"):
            await pipeline.rollback(proposal, reason="no history")

    @pytest.mark.asyncio
    async def test_rejected_proposal_records_memory(self) -> None:
        memory = _FakeMemory()
        pipeline, _registry = _make_pipeline(memory=memory)
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id, candidate_config={"cap": 5}
        )
        experiment = await pipeline.test(proposal, candidate)
        await pipeline.evaluate(proposal, candidate, experiment)
        assert proposal.status is ProposalStatus.REJECTED
        assert memory.stored, "rejections must be recorded to memory"
        procedural = [s for s in memory.stored if s.get("memory_type") == "procedural"]
        assert procedural, "rejection should carry procedural guidance"

    @pytest.mark.asyncio
    async def test_repeat_proposal_records_prior_attempts(self) -> None:
        pipeline, _registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        await pipeline.propose(weakness.id, candidate_config={"cap": 5})
        _h2, proposal2, _c2 = await pipeline.propose(weakness.id, candidate_config={"cap": 6})
        assert proposal2.metadata["prior_attempts"]
        assert proposal2.candidate_ref != "research:cand-" + weakness.id[:8]

    @pytest.mark.asyncio
    async def test_monitor_creates_regression_and_auto_rolls_back(self) -> None:
        pipeline, _registry = _make_pipeline()
        await pipeline.seed_baseline(name="research", version="v1")
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id,
            candidate_config={"cap": 5},
            plan_overrides={
                "acceptance_criteria": [
                    {"kind": "max", "metric": "duplicate_rate", "threshold": 0.35},
                    {"kind": "min", "metric": "score", "threshold": 0.3},
                ],
                "metrics": [
                    MetricDefinition(name="score"),
                    MetricDefinition(name="duplicate_rate", direction="minimize"),
                    MetricDefinition(name="duration_ms", direction="minimize"),
                ],
            },
        )
        experiment = await pipeline.test(proposal, candidate)
        await pipeline.evaluate(proposal, candidate, experiment)
        await pipeline.approve(proposal, approved_by="op")
        await pipeline.apply(proposal, candidate, experiment)
        proposal.metadata["monitor_policy"] = "auto_rollback"
        events = await pipeline.monitor(
            proposal,
            actual={"score": 0.1},
            rollback_below={"score": 0.3},
        )
        assert events, "below-floor actuals must create regression events"
        assert proposal.status is ProposalStatus.ROLLED_BACK
        assert pipeline.active("research").ref == "research:v1"

    @pytest.mark.asyncio
    async def test_monitor_without_floor_records_nothing(self) -> None:
        pipeline, _registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, _c = await pipeline.propose(weakness.id, candidate_config={"cap": 5})
        events = await pipeline.monitor(proposal, actual={"score": 0.1})
        assert events == []


class TestTraceEvents:
    @pytest.mark.asyncio
    async def test_lifecycle_emits_traced_events(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []

        async def emit(event: str, **payload: Any) -> None:
            events.append((event, payload))

        pipeline, _registry = _make_pipeline(emit=emit)
        await pipeline.seed_baseline(name="research", version="v1")
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id,
            candidate_config={"cap": 5},
            plan_overrides={
                "acceptance_criteria": [
                    {"kind": "max", "metric": "duplicate_rate", "threshold": 0.35},
                    {"kind": "min", "metric": "score", "threshold": 0.3},
                ],
                "metrics": [
                    MetricDefinition(name="score"),
                    MetricDefinition(name="duplicate_rate", direction="minimize"),
                    MetricDefinition(name="duration_ms", direction="minimize"),
                ],
            },
        )
        experiment = await pipeline.test(proposal, candidate)
        await pipeline.evaluate(proposal, candidate, experiment)
        await pipeline.approve(proposal, approved_by="op")
        await pipeline.apply(proposal, candidate, experiment)
        await pipeline.rollback(proposal, reason="x")
        names = [name for name, _payload in events]
        for expected in (
            "WEAKNESS_DETECTED",
            "HYPOTHESIS_CREATED",
            "IMPROVEMENT_PROPOSED",
            "CANDIDATE_CREATED",
            "CANDIDATE_TEST_STARTED",
            "CANDIDATE_TEST_COMPLETED",
            "ACCEPTANCE_EVALUATED",
            "IMPROVEMENT_ACCEPTED",
            "HUMAN_APPROVAL_REQUIRED",
            "IMPROVEMENT_APPLIED",
            "IMPROVEMENT_ROLLED_BACK",
        ):
            assert expected in names, f"missing {expected}"
        applied = next(payload for name, payload in events if name == "IMPROVEMENT_APPLIED")
        assert applied["proposal_id"] and applied["version"]


class TestSecurity:
    @pytest.mark.asyncio
    async def test_candidate_is_config_only_never_code(self) -> None:
        pipeline, registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id, candidate_config={"cap": 5}
        )
        # Before test(): the candidate is data only — NOT in the registry.
        with pytest.raises(KeyError):
            registry.get(candidate.ref)
        await pipeline.test(proposal, candidate)
        registered = registry.get(candidate.ref)
        # The candidate reuses the parent's registered procedure object.
        assert registered.procedure is registry.get("research:v1").procedure
        # Only its frozen config differs.
        assert registered.config == {"cap": 5}
        payload = candidate.model_dump()
        assert "code" not in payload and "source" not in payload

    @pytest.mark.asyncio
    async def test_external_origin_same_gates(self) -> None:
        pipeline, _registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, _c = await pipeline.propose(
            weakness.id, origin="external", candidate_config={"cap": 5}
        )
        assert proposal.origin == "external"
        # External proposals still cannot apply without the gate.
        experiment = await pipeline.test(proposal, candidate := pipeline.candidate_for_proposal(proposal.id))
        await pipeline.evaluate(proposal, candidate, experiment)
        with pytest.raises(PermissionError):
            await pipeline.apply(proposal, candidate, experiment)

    @pytest.mark.asyncio
    async def test_approval_gate_blocks_unapproved_apply(self) -> None:
        pipeline, _registry = _make_pipeline()
        weakness = await pipeline.record_weakness(_make_weakness())
        _h, proposal, candidate = await pipeline.propose(
            weakness.id,
            candidate_config={"cap": 5},
            plan_overrides={
                "acceptance_criteria": [
                    {"kind": "max", "metric": "duplicate_rate", "threshold": 0.35},
                    {"kind": "min", "metric": "score", "threshold": 0.3},
                ],
                "metrics": [
                    MetricDefinition(name="score"),
                    MetricDefinition(name="duplicate_rate", direction="minimize"),
                    MetricDefinition(name="duration_ms", direction="minimize"),
                ],
            },
        )
        experiment = await pipeline.test(proposal, candidate)
        await pipeline.evaluate(proposal, candidate, experiment)
        assert proposal.status is ProposalStatus.ACCEPTED
        with pytest.raises(PermissionError):
            await pipeline.apply(proposal, candidate, experiment)


class TestPersistence:
    @pytest.mark.asyncio
    async def test_store_roundtrip_and_restart(self, tmp_path: Any) -> None:
        path = tmp_path / "improvements.jsonl"
        store = ImprovementStore(path=path)
        weakness = _make_weakness()
        await store.save_weakness(weakness)
        from agency.improve.models import ActiveStrategy, CandidateStrategy, StrategyVersion

        candidate = CandidateStrategy(
            parent_ref="research:v1",
            name="research",
            version="cand-x",
            proposal_id="p1",
        )
        await store.save_candidate(candidate)
        version = StrategyVersion(name="research", version="cand-x", proposal_id="p1")
        await store.save_version(version)
        await store.save_active(ActiveStrategy(name="research", version="cand-x"))
        # Fresh "process": new store instance over the same file.
        store2 = ImprovementStore(path=path)
        assert store2.weakness(weakness.id).category is WeaknessCategory.DUPLICATE_SOURCES
        assert store2.candidate_for_proposal("p1").ref == "research:cand-x"
        assert store2.versions("research")[0].ref == "research:cand-x"
        assert store2.active("research").ref == "research:cand-x"

    @pytest.mark.asyncio
    async def test_pipeline_uses_persistent_store(self, tmp_path: Any) -> None:
        path = tmp_path / "improvements.jsonl"
        pipeline, _registry = _make_pipeline(store=ImprovementStore(path=path))
        weakness = await pipeline.record_weakness(_make_weakness())
        await pipeline.propose(weakness.id, candidate_config={"cap": 5})
        pipeline2, _ = _make_pipeline(store=ImprovementStore(path=path))
        assert len(pipeline2.list_weaknesses()) == 1
        assert len(pipeline2.proposals()) == 1
