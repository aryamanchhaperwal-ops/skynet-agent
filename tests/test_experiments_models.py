"""Experiments: models, metrics aggregation/comparison, evaluation verdicts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agency.experiments.evaluation import MIN_TRIALS_FOR_VERDICT, evaluate_experiment
from agency.experiments.metrics import aggregate, compare
from agency.experiments.models import (
    ArmSummary,
    ComparisonReport,
    Experiment,
    ExperimentStatus,
    Hypothesis,
    MetricDefinition,
    SuccessCriterion,
    TargetRef,
    TrialResult,
    Verdict,
)


def _trial(arm: str, index: int, metrics: dict[str, float], *, success: bool = True) -> TrialResult:
    return TrialResult(arm=arm, index=index, metrics=metrics, success=success)  # type: ignore[arg-type]


def _experiment(**overrides: object) -> Experiment:
    base: dict[str, object] = {
        "name": "unit-experiment",
        "hypothesis": Hypothesis(
            statement="candidate beats baseline",
            success_criteria=[
                SuccessCriterion(kind="improves", metric="score", threshold=0.1)
            ],
        ),
        "baseline": TargetRef(name="strat", version="v1"),
        "candidate": TargetRef(name="strat", version="v2"),
        "metrics": [MetricDefinition(name="score", direction="maximize")],
        "trials": 3,
    }
    base.update(overrides)
    return Experiment(**base)  # type: ignore[arg-type]


# -- models ---------------------------------------------------------------------


class TestExperimentModels:
    def test_target_ref_formatting(self) -> None:
        ref = TargetRef(name="planner_policy", version="v3")
        assert ref.ref == "planner_policy:v3"

    def test_target_ref_rejects_separators(self) -> None:
        with pytest.raises(ValidationError):
            TargetRef(name="bad:name", version="v1")

    def test_experiment_serialization_roundtrip(self) -> None:
        experiment = _experiment()
        payload = experiment.model_dump_json()
        restored = Experiment.model_validate_json(payload)
        assert restored.id == experiment.id
        assert restored.hypothesis.statement == "candidate beats baseline"
        assert restored.baseline.ref == "strat:v1"

    def test_criteria_frozen_copies_from_hypothesis_once(self) -> None:
        hypothesis = Hypothesis(
            statement="claim",
            success_criteria=[SuccessCriterion(kind="min", metric="x", threshold=1.0)],
        )
        experiment = _experiment(hypothesis=hypothesis)
        assert experiment.success_criteria == []
        experiment.criteria_frozen()
        assert len(experiment.success_criteria) == 1
        # A later hypothesis edit must NOT change the frozen criteria.
        experiment.hypothesis.success_criteria = [
            SuccessCriterion(kind="max", metric="y", threshold=9.0)
        ]
        experiment.criteria_frozen()
        assert experiment.success_criteria[0].metric == "x"

    def test_experiment_statuses_exist(self) -> None:
        for status in (
            "created", "running", "completed", "failed", "inconclusive",
            "error", "cancelled",
        ):
            assert ExperimentStatus(status)  # parses without error


# -- metrics ---------------------------------------------------------------------


class TestMetrics:
    def test_aggregate_mean_min_max_stddev(self) -> None:
        trials = [
            _trial("baseline", 0, {"score": 1.0}),
            _trial("baseline", 1, {"score": 2.0}),
            _trial("baseline", 2, {"score": 3.0}),
        ]
        summary = aggregate(trials, ["score"])
        assert summary.trials == 3
        assert summary.failed_trials == 0
        assert summary.mean["score"] == 2.0
        assert summary.min["score"] == 1.0
        assert summary.max["score"] == 3.0
        assert summary.stddev["score"] == pytest.approx(1.0)

    def test_aggregate_excludes_failed_trials_but_counts_them(self) -> None:
        trials = [
            _trial("baseline", 0, {"score": 4.0}),
            _trial("baseline", 1, {"score": 0.0}, success=False),
        ]
        summary = aggregate(trials, ["score"])
        assert summary.trials == 2
        assert summary.failed_trials == 1
        assert summary.mean["score"] == 4.0

    def test_compare_respects_direction(self) -> None:
        base = ArmSummary(ref="a:v1", mean={"latency": 10.0, "coverage": 0.5})
        cand = ArmSummary(ref="a:v2", mean={"latency": 8.0, "coverage": 0.6})
        deltas, improvements, regressions = compare(
            base, cand,
            directions={"latency": "minimize", "coverage": "maximize"},
        )
        assert improvements == ["latency", "coverage"]
        assert regressions == []
        assert deltas["latency"]["abs"] == -2.0
        assert deltas["coverage"]["rel"] == pytest.approx(0.2)

    def test_compare_flags_regressions(self) -> None:
        base = ArmSummary(ref="a:v1", mean={"coverage": 0.5})
        cand = ArmSummary(ref="a:v2", mean={"coverage": 0.4})
        _deltas, improvements, regressions = compare(
            base, cand, directions={"coverage": "maximize"}
        )
        assert improvements == []
        assert regressions == ["coverage"]

    def test_compare_handles_missing_metrics(self) -> None:
        base = ArmSummary(ref="a:v1", mean={})
        cand = ArmSummary(ref="a:v2", mean={"coverage": 0.4})
        deltas, _improvements, _regressions = compare(
            base, cand, directions={"coverage": "maximize"}
        )
        assert deltas["coverage"]["abs"] is None


# -- evaluation ---------------------------------------------------------------------


class TestEvaluation:
    def _finished(
        self,
        *,
        baseline_mean: dict[str, float],
        candidate_mean: dict[str, float],
        criteria: list[SuccessCriterion] | None = None,
        directions: dict[str, str] | None = None,
        trials: int = 5,
    ) -> Experiment:
        criteria = criteria if criteria is not None else [
            SuccessCriterion(kind="improves", metric="score", threshold=0.1)
        ]
        directions = directions or {"score": "maximize"}
        experiment = _experiment(
            hypothesis=Hypothesis(statement="claim", success_criteria=criteria),
            trials=trials,
        )
        experiment.criteria_frozen()
        experiment.trials_results = (
            [_trial("baseline", i, baseline_mean) for i in range(trials)]
            + [_trial("candidate", i, candidate_mean) for i in range(trials)]
        )
        baseline = aggregate(
            [t for t in experiment.trials_results if t.arm == "baseline"],
            list(directions),
        )
        candidate = aggregate(
            [t for t in experiment.trials_results if t.arm == "candidate"],
            list(directions),
        )
        deltas, improvements, regressions = compare(
            baseline, candidate, directions=directions
        )
        experiment.comparison = ComparisonReport(
            baseline=baseline,
            candidate=candidate,
            delta={k: v["abs"] for k, v in deltas.items() if v["abs"] is not None},
            relative={k: v["rel"] for k, v in deltas.items() if v.get("rel") is not None},
            improvements=improvements,
            regressions=regressions,
            significance_tested=False,
        )
        return experiment

    def test_success_when_all_criteria_pass(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 1.0}, candidate_mean={"score": 2.0}
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.SUCCESS
        assert evaluation.source == "deterministic"
        assert evaluation.criteria_results[0]["outcome"] == "passed"

    def test_failure_when_criterion_fails(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 2.0}, candidate_mean={"score": 2.0}
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.FAILURE

    def test_inconclusive_on_zero_baseline(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 0.0}, candidate_mean={"score": 5.0}
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.INCONCLUSIVE
        assert "baseline is 0" in evaluation.criteria_results[0]["detail"]

    def test_inconclusive_on_no_criteria(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 1.0},
            candidate_mean={"score": 2.0},
            criteria=[],
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.INCONCLUSIVE
        assert "no success criteria" in evaluation.explanation

    def test_inconclusive_on_mixed_evidence(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 1.0, "cost": 1.0},
            candidate_mean={"score": 2.0, "cost": 5.0},
            criteria=[
                SuccessCriterion(kind="improves", metric="score", threshold=0.1),
                SuccessCriterion(kind="max", metric="cost", threshold=2.0),
            ],
            directions={"score": "maximize", "cost": "minimize"},
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.INCONCLUSIVE
        assert "mixed evidence" in evaluation.explanation

    def test_error_when_every_trial_failed(self) -> None:
        experiment = self._finished(
            baseline_mean={}, candidate_mean={}
        )
        experiment.trials_results = [
            _trial("baseline", 0, {}, success=False),
            _trial("candidate", 0, {}, success=False),
        ]
        baseline = aggregate(experiment.trials_results, ["score"])
        candidate = aggregate(experiment.trials_results, ["score"])
        experiment.comparison = ComparisonReport(
            baseline=baseline, candidate=candidate, significance_tested=False
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.ERROR

    def test_few_trials_downgrade_success_to_inconclusive(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 1.0},
            candidate_mean={"score": 3.0},
            trials=MIN_TRIALS_FOR_VERDICT - 1,
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.INCONCLUSIVE
        assert "lucky run" in evaluation.explanation
        assert any("downgrading" in line for line in evaluation.evidence)

    def test_uncovered_regression_downgrades_success(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 1.0, "latency": 10.0},
            candidate_mean={"score": 2.0, "latency": 20.0},
            criteria=[SuccessCriterion(kind="improves", metric="score", threshold=0.1)],
            directions={"score": "maximize", "latency": "minimize"},
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.INCONCLUSIVE
        assert any("uncovered regression" in line for line in evaluation.evidence)

    def test_covered_regression_does_not_downgrade(self) -> None:
        # The latency regression is priced in via an absolute max criterion.
        experiment = self._finished(
            baseline_mean={"score": 1.0, "latency": 10.0},
            candidate_mean={"score": 2.0, "latency": 12.0},
            criteria=[
                SuccessCriterion(kind="improves", metric="score", threshold=0.1),
                SuccessCriterion(kind="max", metric="latency", threshold=15.0),
            ],
            directions={"score": "maximize", "latency": "minimize"},
        )
        evaluation = evaluate_experiment(experiment)
        assert evaluation.verdict is Verdict.SUCCESS

    def test_no_comparison_is_error(self) -> None:
        evaluation = evaluate_experiment(_experiment())
        assert evaluation.verdict is Verdict.ERROR

    def test_absolute_min_criterion(self) -> None:
        experiment = self._finished(
            baseline_mean={"score": 0.9},
            candidate_mean={"score": 0.95},
            criteria=[SuccessCriterion(kind="min", metric="score", threshold=0.9)],
        )
        assert evaluate_experiment(experiment).verdict is Verdict.SUCCESS
