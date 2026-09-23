"""Weakness detection — finding recurring problems in recorded evidence.

The detector analyzes *actual historical performance data*: experiences
from :class:`~agency.experience.ExperienceRecorder`, finished experiments
from the P6 registry, and long-term memories. Design rules:

- **No fabrication.** A weakness requires at least ``min_frequency``
  supporting records; every weakness carries its evidence items verbatim
  (ids, counts, values) so a human can audit the claim.
- **Configurable thresholds.** Every rule is a threshold; nothing is
  hard-coded as "bad".
- **Deterministic.** Same data + same config → same weaknesses. The
  (future) LLM may help *interpret* a detected weakness; it never
  invents one.

Each detection rule groups records by ``group_key`` (e.g. action name or
error signature) and reports one weakness per group that crosses the
threshold — recurrence is what makes it a weakness rather than noise.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from agency.experience import ExperienceRecord
from agency.experiments.models import Experiment
from agency.improve.models import (
    EvidenceItem,
    Weakness,
    WeaknessCategory,
)

logger = logging.getLogger("skynet.improve.detector")


@dataclass
class DetectorThresholds:
    """Configurable detection thresholds (all rules opt-out via None)."""

    #: Minimum records backing one weakness (recurrence bar).
    min_frequency: int = 2
    #: ``error``/``failure`` experiences sharing a signature.
    min_task_failures: int = 2
    #: Evaluations whose score falls below this (0–1).
    low_score: float = 0.4
    #: Minimum low-scored evaluations before reporting.
    min_low_scores: int = 2
    #: Failed trials per broken experiment to call it a planning/execution failure.
    experiment_failure_trials: int = 2
    #: Duplicate-source rate above this in a candidate arm.
    duplicate_rate: float = 0.5
    #: Mean action latency (ms) above this.
    slow_action_ms: float = 30_000.0
    #: Mean steps per run above this.
    excessive_steps: float = 8.0

    def __post_init__(self) -> None:
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")


@dataclass
class DetectorInput:
    """The recorded evidence pools the detector may see (all optional)."""

    experiences: list[ExperienceRecord] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)


class WeaknessDetector:
    """Deterministic weakness detection over recorded performance data."""

    def __init__(self, thresholds: DetectorThresholds | None = None) -> None:
        self._thresholds = thresholds or DetectorThresholds()

    # -- public ---------------------------------------------------------------

    def detect(self, data: DetectorInput) -> list[Weakness]:
        """Run every rule; returns weaknesses sorted by severity (desc)."""
        weaknesses: list[Weakness] = []
        weaknesses.extend(self._detect_task_failures(data.experiences))
        weaknesses.extend(self._detect_low_scores(data.experiences))
        weaknesses.extend(self._detect_excessive_steps(data.experiences))
        weaknesses.extend(self._detect_experiment_failures(data.experiments))
        weaknesses.extend(self._detect_duplicate_sources(data.experiments))
        weaknesses.sort(key=lambda w: (-w.severity, -w.frequency, w.category.value))
        for weakness in weaknesses:
            logger.info(
                "weakness detected: %s (%s, freq=%d, severity=%.2f)",
                weakness.description[:80],
                weakness.category.value,
                weakness.frequency,
                weakness.severity,
            )
        return weaknesses

    # -- rules -------------------------------------------------------------------

    def _detect_task_failures(
        self, experiences: list[ExperienceRecord]
    ) -> list[Weakness]:
        t = self._thresholds
        if t.min_task_failures < t.min_frequency:
            t_min = t.min_frequency
        else:
            t_min = t.min_task_failures
        groups: dict[str, list[ExperienceRecord]] = {}
        for record in experiences:
            if record.kind != "error" and record.outcome != "failure":
                continue
            signature = _failure_signature(record)
            groups.setdefault(signature, []).append(record)
        weaknesses: list[Weakness] = []
        for signature, records in sorted(groups.items()):
            if len(records) < t_min:
                continue
            runs = sorted({r.run_id for r in records if r.run_id})
            weaknesses.append(
                Weakness(
                    category=WeaknessCategory.TASK_FAILURE,
                    description=f"repeated failures matching {signature!r}",
                    evidence=[
                        EvidenceItem(
                            summary=r.summary,
                            source=f"experience:{r.id}",
                            detail={
                                "kind": r.kind,
                                "outcome": r.outcome,
                                "run_id": r.run_id,
                                "goal_id": r.goal_id,
                            },
                        )
                        for r in records[:10]
                    ],
                    affected_component=signature,
                    frequency=len(records),
                    severity=_severity(len(records), floor=0.5, cap=0.95),
                    related_runs=runs,
                    metadata={"signature": signature},
                )
            )
        return weaknesses

    def _detect_low_scores(
        self, experiences: list[ExperienceRecord]
    ) -> list[Weakness]:
        t = self._thresholds
        scored = [
            r
            for r in experiences
            if r.kind == "evaluation"
            and isinstance(r.payload.get("score"), (int, float))
            and float(r.payload["score"]) < t.low_score
        ]
        if len(scored) < max(t.min_frequency, t.min_low_scores):
            return []
        runs = sorted({r.run_id for r in scored if r.run_id})
        scores = [float(r.payload["score"]) for r in scored]
        return [
            Weakness(
                category=WeaknessCategory.LOW_EVALUATION,
                description=(
                    f"{len(scored)} evaluations scored below {t.low_score:.2f} "
                    f"(mean {sum(scores) / len(scores):.2f})"
                ),
                evidence=[
                    EvidenceItem(
                        summary=r.summary,
                        source=f"experience:{r.id}",
                        detail={"score": r.payload.get("score"), "run_id": r.run_id},
                    )
                    for r in scored[:10]
                ],
                affected_component="evaluator",
                frequency=len(scored),
                severity=_severity(len(scored), floor=0.4, cap=0.9),
                related_runs=runs,
                metadata={"threshold": t.low_score, "scores": scores[:20]},
            )
        ]

    def _detect_excessive_steps(
        self, experiences: list[ExperienceRecord]
    ) -> list[Weakness]:
        t = self._thresholds
        steps_by_run: Counter[str] = Counter()
        for record in experiences:
            if record.kind == "action" and record.run_id:
                steps_by_run[record.run_id] += 1
        heavy = {run: n for run, n in steps_by_run.items() if n > t.excessive_steps}
        if not heavy:
            return []
        total = sum(heavy.values())
        return [
            Weakness(
                category=WeaknessCategory.EXCESSIVE_TOOL_CALLS,
                description=(
                    f"{len(heavy)} run(s) exceeded {t.excessive_steps:.0f} tool "
                    f"calls (max {max(heavy.values())})"
                ),
                evidence=[
                    EvidenceItem(
                        summary=f"run {run} used {n} tool calls",
                        source=f"run:{run}",
                        detail={"steps": n, "threshold": t.excessive_steps},
                    )
                    for run, n in sorted(heavy.items(), key=lambda kv: -kv[1])[:10]
                ],
                affected_component="planner",
                frequency=total,
                severity=_severity(total, floor=0.35, cap=0.85),
                related_runs=sorted(heavy),
                metadata={"threshold": t.excessive_steps, "by_run": heavy},
            )
        ]

    def _detect_experiment_failures(
        self, experiments: list[Experiment]
    ) -> list[Weakness]:
        t = self._thresholds
        weaknesses: list[Weakness] = []
        for experiment in experiments:
            failed = sum(1 for trial in experiment.trials_results if not trial.success)
            if failed < max(t.min_frequency, t.experiment_failure_trials):
                continue
            errors = Counter(
                trial.error.split(":")[0] if trial.error else "unknown"
                for trial in experiment.trials_results
                if not trial.success
            )
            top_error, _count = errors.most_common(1)[0]
            weaknesses.append(
                Weakness(
                    category=WeaknessCategory.PLANNING_FAILURE,
                    description=(
                        f"experiment {experiment.name!r}: {failed}/"
                        f"{len(experiment.trials_results)} trials failed ({top_error})"
                    ),
                    evidence=[
                        EvidenceItem(
                            summary=trial.error or "trial failed",
                            source=f"experiment:{experiment.id}",
                            detail={
                                "trial_id": trial.trial_id,
                                "arm": trial.arm,
                                "index": trial.index,
                            },
                        )
                        for trial in experiment.trials_results
                        if not trial.success
                    ][:10],
                    affected_component=experiment.candidate.ref,
                    frequency=failed,
                    severity=_severity(failed, floor=0.5, cap=0.95),
                    related_experiments=[experiment.id],
                    metadata={
                        "experiment_id": experiment.id,
                        "error_counts": dict(errors),
                    },
                )
            )
        return weaknesses

    def _detect_duplicate_sources(
        self, experiments: list[Experiment]
    ) -> list[Weakness]:
        t = self._thresholds
        weaknesses: list[Weakness] = []
        for experiment in experiments:
            comparison = experiment.comparison
            if comparison is None:
                continue
            for arm_name, arm in (
                ("baseline", comparison.baseline),
                ("candidate", comparison.candidate),
            ):
                rate = arm.mean.get("duplicate_rate")
                if rate is None or rate < t.duplicate_rate:
                    continue
                weaknesses.append(
                    Weakness(
                        category=WeaknessCategory.DUPLICATE_SOURCES,
                        description=(
                            f"{arm.ref} {arm_name} duplicate_rate "
                            f"{rate:.2f} ≥ {t.duplicate_rate:.2f}"
                        ),
                        evidence=[
                            EvidenceItem(
                                summary=f"{arm_name} mean duplicate_rate={rate}",
                                source=f"experiment:{experiment.id}",
                                detail={"arm": arm_name, "mean": dict(arm.mean)},
                            )
                        ],
                        affected_component=arm.ref,
                        frequency=max(1, arm.trials),
                        severity=_severity(int(rate * 10), floor=0.3, cap=0.8),
                        related_experiments=[experiment.id],
                        metadata={"duplicate_rate": rate, "threshold": t.duplicate_rate},
                    )
                )
        return weaknesses


def _failure_signature(record: ExperienceRecord) -> str:
    """Coarse grouping key: error type + action name when present."""
    payload = record.payload or {}
    action = payload.get("action") or payload.get("action_name") or ""
    error = payload.get("error") or payload.get("error_type") or ""
    if error:
        head = str(error).split(":")[0].strip()
        return f"{action}:{head}" if action else head
    return action or record.summary[:60]


def _severity(frequency: int, *, floor: float, cap: float) -> float:
    """Deterministic severity: log2 growth from floor to cap."""
    import math

    if frequency <= 1:
        return round(floor, 3)
    scaled = floor + (cap - floor) * min(1.0, math.log2(frequency + 1) / 5.0)
    return round(min(cap, scaled), 3)


__all__ = [
    "DetectorInput",
    "DetectorThresholds",
    "WeaknessDetector",
]
