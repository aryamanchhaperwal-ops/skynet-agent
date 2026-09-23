"""Experiment evaluation engine.

The evaluator turns measured metrics + frozen acceptance criteria into a
verdict. Design rules (phase spec §9–10):

- **Metrics decide where metrics exist.** Criteria are evaluated
  mechanically against aggregated trial data.
- **Uncertainty stays uncertainty.** Missing metrics, zero baselines that
  defeat relative deltas, mixed criterion outcomes, failed arms and absent
  criteria all produce INCONCLUSIVE — never a guessed SUCCESS/FAILURE.
- **The LLM never decides.** :class:`LLMExperimentAdvisor` may add
  supplementary commentary via the existing intelligence layer; its text is
  recorded as evidence with ``source="llm"`` and can flip no verdict.
"""

from __future__ import annotations

import logging
from typing import Any

from agency.experiments.models import (
    ComparisonReport,
    Experiment,
    ExperimentEvaluation,
    Verdict,
)

logger = logging.getLogger("skynet.experiments.evaluator")

#: Below this trial count per arm, criteria outcomes are reported but the
#: verdict degrades to INCONCLUSIVE (single lucky runs are not evidence).
MIN_TRIALS_FOR_VERDICT = 3


def evaluate_experiment(experiment: Experiment) -> ExperimentEvaluation:
    """Mechanical, deterministic evaluation of a finished comparison.

    Only :class:`Verdict.SUCCESS` requires *every* success criterion to
    pass; any failing criterion yields FAILURE; everything ambiguous
    (missing data, zero baselines, no criteria, mixed …) yields
    INCONCLUSIVE.
    """
    comparison = experiment.comparison
    if comparison is None:
        return ExperimentEvaluation(
            verdict=Verdict.ERROR,
            explanation="no comparison report (experiment did not complete its trials)",
        )

    evidence: list[str] = [f"trials: baseline={comparison.baseline.trials}, "
                           f"candidate={comparison.candidate.trials}"]
    criteria_results = _apply_criteria(experiment, comparison, evidence)

    failed_trials_total = comparison.baseline.failed_trials + comparison.candidate.failed_trials
    all_trials_failed = failed_trials_total and failed_trials_total >= (
        comparison.baseline.trials + comparison.candidate.trials
    )
    if not experiment.success_criteria:
        # A broken procedure is ERROR even without criteria — "nothing ran"
        # is more informative than "nothing was judged".
        if all_trials_failed:
            return ExperimentEvaluation(
                verdict=Verdict.ERROR,
                criteria_results=criteria_results,
                evidence=evidence,
                explanation="every trial failed and no criteria were frozen",
            )
        return ExperimentEvaluation(
            verdict=Verdict.INCONCLUSIVE,
            criteria_results=criteria_results,
            evidence=evidence,
            explanation="no success criteria were frozen; no verdict is possible",
        )

    if all_trials_failed:
        return ExperimentEvaluation(
            verdict=Verdict.ERROR,
            criteria_results=criteria_results,
            evidence=evidence,
            explanation="every trial failed on at least one arm; procedure is broken",
        )

    passed = sum(1 for c in criteria_results if c["outcome"] == "passed")
    failed = sum(1 for c in criteria_results if c["outcome"] == "failed")
    skipped = sum(1 for c in criteria_results if c["outcome"] == "skipped")

    if failed and passed:
        verdict = Verdict.INCONCLUSIVE
        explanation = (
            f"mixed evidence: {passed} criterion(a) passed, {failed} failed "
            "(no single verdict is justified)"
        )
    elif failed:
        verdict = Verdict.FAILURE
        explanation = f"{failed} acceptance criterion(a) failed"
    elif passed:
        verdict = Verdict.SUCCESS
        explanation = f"all {passed} acceptance criteria passed"
    else:  # everything skipped (unmeasurable data)
        verdict = Verdict.INCONCLUSIVE
        explanation = "no criterion could be evaluated on the measured data"
    if skipped and verdict is not Verdict.INCONCLUSIVE:
        evidence.append(f"{skipped} criterion(a) skipped (metric missing/zero baseline)")

    if verdict is Verdict.SUCCESS and (
        comparison.baseline.trials < MIN_TRIALS_FOR_VERDICT
        or comparison.candidate.trials < MIN_TRIALS_FOR_VERDICT
    ):
        evidence.append(
            f"fewer than {MIN_TRIALS_FOR_VERDICT} trials per arm; "
            "downgrading SUCCESS to INCONCLUSIVE"
        )
        verdict = Verdict.INCONCLUSIVE
        explanation = (
            f"{explanation} — but fewer than {MIN_TRIALS_FOR_VERDICT} trials per arm; "
            "a single lucky run is not evidence"
        )

    if comparison.regressions and verdict is Verdict.SUCCESS:
        # A regression the criteria did not explicitly cover still taints
        # success; one a criterion tolerates (e.g. ``max`` on that metric)
        # is already priced into the verdict.
        covered = {c.metric for c in experiment.success_criteria}
        uncovered = [m for m in comparison.regressions if m not in covered]
        if uncovered:
            evidence.append(
                "uncovered regression(s) detected: " + ", ".join(uncovered)
            )
            verdict = Verdict.INCONCLUSIVE
            explanation += "; uncovered regressions detected — verdict downgraded"

    return ExperimentEvaluation(
        verdict=verdict,
        score=_score(comparison),
        criteria_results=criteria_results,
        evidence=evidence,
        regressions=list(comparison.regressions),
        explanation=explanation,
        source="deterministic",
    )


def _apply_criteria(
    experiment: Experiment,
    comparison: ComparisonReport,
    evidence: list[str],
) -> list[dict[str, Any]]:
    """Evaluate each frozen criterion; returns outcome records."""
    results: list[dict[str, Any]] = []
    for criterion in experiment.success_criteria:
        record: dict[str, Any] = {"criterion": criterion.description, "outcome": "skipped"}
        base = comparison.baseline.mean.get(criterion.metric)
        cand = comparison.candidate.mean.get(criterion.metric)
        if base is None or cand is None:
            record["detail"] = f"metric {criterion.metric!r} missing on an arm"
            results.append(record)
            continue
        if criterion.kind == "improves":
            if base == 0:
                record["detail"] = "baseline is 0; relative improvement undefined"
                results.append(record)
                continue
            rel = (cand - base) / abs(base)
            record["measured"] = round(rel, 6)
            record["outcome"] = "passed" if rel >= criterion.threshold else "failed"
        elif criterion.kind == "not_regresses":
            if base == 0:
                record["detail"] = "baseline is 0; relative regression undefined"
                results.append(record)
                continue
            rel = (cand - base) / abs(base)
            worsened = rel < 0
            record["measured"] = round(rel, 6)
            record["outcome"] = (
                "failed" if worsened and abs(rel) > criterion.threshold else "passed"
            )
        elif criterion.kind == "max":
            record["measured"] = cand
            record["outcome"] = "passed" if cand <= criterion.threshold else "failed"
        else:  # min
            record["measured"] = cand
            record["outcome"] = "passed" if cand >= criterion.threshold else "failed"
        results.append(record)
    return results


def _score(comparison: ComparisonReport) -> float:
    """Coarse 0–1 score: share of improved vs moved metrics (informational)."""
    moved = len(comparison.improvements) + len(comparison.regressions)
    if not moved:
        return 0.5
    return round(len(comparison.improvements) / moved, 4)


class LLMExperimentAdvisor:
    """Optional LLM commentary on a finished experiment (never decisive).

    Wraps the existing :class:`~agency.intelligence.service.IntelligenceService`;
    any failure returns ``None`` text and the evaluation stands unchanged.
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    async def advise(self, experiment: Experiment) -> str | None:
        comparison = experiment.comparison
        evaluation = experiment.evaluation
        if comparison is None or evaluation is None:
            return None
        user = (
            f"Experiment: {experiment.name}\n"
            f"Hypothesis: {experiment.hypothesis.statement}\n"
            f"Baseline {comparison.baseline.ref}: mean={comparison.baseline.mean}\n"
            f"Candidate {comparison.candidate.ref}: mean={comparison.candidate.mean}\n"
            f"Deltas: {comparison.delta}\n"
            f"Verdict: {evaluation.verdict.value} — {evaluation.explanation}\n"
            "In 3 sentences: what does this evidence support, what is still "
            "uncertain, and what should be tried next? Do not invent numbers."
        )
        try:
            return await self._service.generate(
                "You are a cautious experiment analyst. Describe only what the "
                "measured data supports; never overstate certainty.",
                user,
                metadata={"purpose": "experiment_advisory"},
            )
        except Exception as exc:  # advisory must never break the pipeline
            logger.warning("experiment advisory failed: %s", exc)
            return None


__all__ = [
    "MIN_TRIALS_FOR_VERDICT",
    "LLMExperimentAdvisor",
    "evaluate_experiment",
]
