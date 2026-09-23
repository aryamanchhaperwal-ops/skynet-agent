"""Self-improvement models + weakness detector (Phase P7)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agency.experience import ExperienceRecord
from agency.experiments.models import (
    ArmSummary,
    ComparisonReport,
    Experiment,
    ExperimentStatus,
    MetricDefinition,
    TargetRef,
    TrialResult,
)
from agency.experiments.models import (
    Hypothesis as ExperimentHypothesis,
)
from agency.experiments.registry import StrategyRegistry
from agency.improve.detector import DetectorInput, DetectorThresholds, WeaknessDetector
from agency.improve.models import (
    ActiveStrategy,
    CandidateStrategy,
    EvidenceItem,
    ExperimentPlan,
    ImprovementHypothesis,
    ImprovementProposal,
    ProposalStatus,
    StrategyVersion,
    Weakness,
    WeaknessCategory,
    WeaknessStatus,
)
from agency.improve.pipeline import build_proposal_artifacts

# -- Weakness model ------------------------------------------------------------------


class TestWeaknessModel:
    def test_creation_with_evidence(self) -> None:
        weakness = Weakness(
            category=WeaknessCategory.TASK_FAILURE,
            description="repeated failures",
            evidence=[EvidenceItem(summary="boom", source="experience:1")],
            frequency=3,
        )
        assert weakness.status is WeaknessStatus.OPEN
        assert weakness.evidence[0].summary == "boom"

    def test_evidence_required_no_fabrication(self) -> None:
        with pytest.raises(ValidationError):
            Weakness(category=WeaknessCategory.TASK_FAILURE, description="x", evidence=[])

    def test_severity_bounds(self) -> None:
        weakness = Weakness(
            category=WeaknessCategory.LOW_EVALUATION,
            description="d",
            evidence=[EvidenceItem(summary="s")],
            severity=0.42,
        )
        assert 0.0 <= weakness.severity <= 1.0

    def test_status_transition_updates_timestamp(self) -> None:
        weakness = Weakness(
            category=WeaknessCategory.LOW_EVALUATION,
            description="d",
            evidence=[EvidenceItem(summary="s")],
        )
        before = weakness.updated_at
        weakness.transition(WeaknessStatus.HYPOTHESIZED)
        assert weakness.status is WeaknessStatus.HYPOTHESIZED
        assert weakness.updated_at >= before

    def test_serialization_roundtrip(self) -> None:
        weakness = Weakness(
            category=WeaknessCategory.DUPLICATE_SOURCES,
            description="d",
            evidence=[EvidenceItem(summary="s", detail={"rate": 0.6})],
            related_experiments=["e1"],
        )
        parsed = Weakness.model_validate_json(weakness.model_dump_json())
        assert parsed.id == weakness.id
        assert parsed.category is WeaknessCategory.DUPLICATE_SOURCES


# -- Hypothesis / proposal / candidate / version models ---------------------------------


class TestImprovementModels:
    def test_hypothesis_creation(self) -> None:
        h = ImprovementHypothesis(
            weakness_id="w1",
            statement="tightening the cap reduces duplicates",
            confidence=0.6,
        )
        assert h.weakness_id == "w1"
        assert h.statement

    def test_proposal_lifecycle_statuses(self) -> None:
        proposal = ImprovementProposal(weakness_id="w", hypothesis_id="h", target="s:v1")
        assert proposal.status is ProposalStatus.PROPOSED
        proposal.transition(ProposalStatus.APPROVED_FOR_TEST)
        proposal.transition(ProposalStatus.TESTING)
        proposal.transition(ProposalStatus.ACCEPTED)
        assert proposal.status is ProposalStatus.ACCEPTED
        proposal.transition(ProposalStatus.ROLLED_BACK)
        assert proposal.status is ProposalStatus.ROLLED_BACK

    def test_proposal_origin_is_internal_or_external(self) -> None:
        proposal = ImprovementProposal(weakness_id="w", hypothesis_id="h", target="s:v1")
        assert proposal.origin == "internal"
        external = ImprovementProposal(
            weakness_id="w", hypothesis_id="h", target="s:v1", origin="external"
        )
        assert external.origin == "external"
        with pytest.raises(ValidationError):
            ImprovementProposal(
                weakness_id="w", hypothesis_id="h", target="s:v1", origin="hacker"
            )

    def test_candidate_is_config_only(self) -> None:
        candidate = CandidateStrategy(
            parent_ref="research_strategy:v1",
            name="research_strategy",
            version="cand-abc",
            config={"max_results": 5},
        )
        assert candidate.ref == "research_strategy:cand-abc"
        # The candidate carries data, never code references.
        assert "procedure" not in candidate.model_dump()

    def test_version_preserves_provenance(self) -> None:
        version = StrategyVersion(
            name="s",
            version="v2",
            parent_version="v1",
            proposal_id="p1",
            experiment_id="e1",
            config={"cap": 5},
            provenance={"weakness_id": "w1"},
        )
        assert version.parent_version == "v1"
        assert version.ref == "s:v2"

    def test_active_strategy_pointer(self) -> None:
        active = ActiveStrategy(name="s", version="v2", activated_from="s:v1")
        assert active.ref == "s:v2"
        assert active.activated_from == "s:v1"

    def test_experiment_plan_defaults(self) -> None:
        plan = ExperimentPlan()
        assert plan.trials == 3
        assert plan.benchmark == ""


# -- Detector -----------------------------------------------------------------------------


def _exp(
    kind: str = "error",
    outcome: str = "failure",
    summary: str = "boom",
    payload: dict | None = None,
    run_id: str | None = None,
) -> ExperienceRecord:
    return ExperienceRecord(
        kind=kind,  # type: ignore[arg-type]
        summary=summary,
        payload=payload or {},
        outcome=outcome,  # type: ignore[arg-type]
        run_id=run_id,
    )


def _experiment(
    *,
    name: str = "e",
    baseline_mean: dict[str, float] | None = None,
    candidate_mean: dict[str, float] | None = None,
    failed_trials: int = 0,
    total_trials: int = 3,
) -> Experiment:
    baseline_mean = baseline_mean or {}
    candidate_mean = candidate_mean or {}
    trials = []
    for i in range(total_trials):
        arm = "baseline" if i % 2 == 0 else "candidate"
        trials.append(
            TrialResult(
                arm=arm,  # type: ignore[arg-type]
                index=i,
                success=(i >= failed_trials),
                error=None if i >= failed_trials else "RuntimeError: broken",
                metrics=(baseline_mean if arm == "baseline" else candidate_mean),
            )
        )
    baseline = ArmSummary(ref="b:v1", trials=total_trials // 2, mean=baseline_mean)
    candidate = ArmSummary(ref="c:v1", trials=total_trials // 2, mean=candidate_mean)
    experiment = Experiment(
        name=name,
        hypothesis=ExperimentHypothesis(statement="s"),
        baseline=TargetRef(name="b", version="v1"),
        candidate=TargetRef(name="c", version="v1"),
        metrics=[MetricDefinition(name="duplicate_rate", direction="minimize")],
        trials_results=trials,
    )
    experiment.status = ExperimentStatus.COMPLETED
    experiment.comparison = ComparisonReport(
        baseline=baseline, candidate=candidate, delta={}, relative={}
    )
    experiment.evaluation = None
    return experiment


class TestDetector:
    def test_repeated_failures_detected(self) -> None:
        detector = WeaknessDetector(DetectorThresholds(min_frequency=1, min_task_failures=2))
        records = [
            _exp(payload={"error": "ValueError: bad", "action": "web_search"})
            for _ in range(2)
        ]
        weaknesses = detector.detect(DetectorInput(experiences=records))
        assert len(weaknesses) == 1
        assert weaknesses[0].category is WeaknessCategory.TASK_FAILURE
        assert weaknesses[0].frequency == 2
        assert weaknesses[0].evidence  # structured evidence present

    def test_insufficient_evidence_not_reported(self) -> None:
        detector = WeaknessDetector(DetectorThresholds(min_frequency=1, min_task_failures=2))
        records = [_exp(payload={"error": "ValueError: bad"})]  # only one
        assert detector.detect(DetectorInput(experiences=records)) == []

    def test_low_scores_detected(self) -> None:
        detector = WeaknessDetector(DetectorThresholds(min_frequency=2, low_score=0.4))
        records = [
            _exp(kind="evaluation", outcome="neutral", payload={"score": 0.2})
            for _ in range(3)
        ]
        weaknesses = detector.detect(DetectorInput(experiences=records))
        assert any(w.category is WeaknessCategory.LOW_EVALUATION for w in weaknesses)

    def test_excessive_steps_detected(self) -> None:
        detector = WeaknessDetector(DetectorThresholds(min_frequency=1, excessive_steps=2))
        records = [
            _exp(kind="action", outcome="neutral", summary="step", run_id=f"r{i}")
            for i in range(3)
        ] + [
            _exp(kind="action", outcome="neutral", summary="step", run_id="r0")
            for _ in range(2)  # r0 ends with 3 steps (> threshold 2)
        ]
        weaknesses = detector.detect(DetectorInput(experiences=records))
        assert any(w.category is WeaknessCategory.EXCESSIVE_TOOL_CALLS for w in weaknesses)

    def test_experiment_failures_detected(self) -> None:
        detector = WeaknessDetector(
            DetectorThresholds(min_frequency=1, experiment_failure_trials=2)
        )
        experiment = _experiment(failed_trials=2, total_trials=4)
        weaknesses = detector.detect(DetectorInput(experiments=[experiment]))
        assert any(w.category is WeaknessCategory.PLANNING_FAILURE for w in weaknesses)

    def test_duplicate_sources_detected(self) -> None:
        detector = WeaknessDetector(
            DetectorThresholds(min_frequency=1, duplicate_rate=0.5)
        )
        experiment = _experiment(baseline_mean={"duplicate_rate": 0.62})
        weaknesses = detector.detect(DetectorInput(experiments=[experiment]))
        assert any(w.category is WeaknessCategory.DUPLICATE_SOURCES for w in weaknesses)

    def test_healthy_data_yields_nothing(self) -> None:
        detector = WeaknessDetector()
        experiment = _experiment(baseline_mean={"duplicate_rate": 0.1})
        weaknesses = detector.detect(DetectorInput(experiments=[experiment]))
        assert weaknesses == []

    def test_severity_scales_with_frequency(self) -> None:
        detector = WeaknessDetector(DetectorThresholds(min_frequency=1, min_task_failures=2))
        few = detector.detect(
            DetectorInput(experiences=[_exp(payload={"error": "E: x"}) for _ in range(2)])
        )
        many = detector.detect(
            DetectorInput(experiences=[_exp(payload={"error": "E: x"}) for _ in range(10)])
        )
        assert many[0].severity > few[0].severity


# -- build_proposal_artifacts -----------------------------------------------------------------


class TestProposalBuilder:
    def _weakness(self) -> Weakness:
        return Weakness(
            category=WeaknessCategory.DUPLICATE_SOURCES,
            description="too many duplicates",
            evidence=[EvidenceItem(summary="rate=0.62")],
            affected_strategy="research_strategy:v1",
        )

    def test_builds_consistent_artifacts(self) -> None:
        registry = StrategyRegistry()

        async def proc(ctx):  # pragma: no cover - never executed here
            return {}

        registry.register(
            name="research_strategy",
            version="v1",
            capability="research_strategy",
            procedure=proc,
        )
        hypothesis, proposal, candidate = build_proposal_artifacts(
            self._weakness(), registry
        )
        assert hypothesis.weakness_id == proposal.weakness_id == candidate.weakness_id
        assert candidate.parent_ref == "research_strategy:v1"
        assert proposal.acceptance_criteria  # criteria are mandatory
        assert proposal.rollback_plan

    def test_refuses_unregistered_target(self) -> None:
        registry = StrategyRegistry()
        with pytest.raises(KeyError):
            build_proposal_artifacts(self._weakness(), registry)

    def test_refuses_weakness_without_strategy(self) -> None:
        registry = StrategyRegistry()
        weakness = Weakness(
            category=WeaknessCategory.TASK_FAILURE,
            description="d",
            evidence=[EvidenceItem(summary="s")],
        )
        with pytest.raises(KeyError):
            build_proposal_artifacts(weakness, registry)

    def test_repeat_attempts_get_fresh_versions(self) -> None:
        registry = StrategyRegistry()

        async def proc(ctx):  # pragma: no cover
            return {}

        registry.register(
            name="research_strategy",
            version="v1",
            capability="research_strategy",
            procedure=proc,
        )
        _, _, first = build_proposal_artifacts(self._weakness(), registry)
        _, _, second = build_proposal_artifacts(self._weakness(), registry, attempt=1)
        assert first.version != second.version
