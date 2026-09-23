"""Experiment and hypothesis models (blueprint Phase P6).

An experiment compares a **candidate strategy** against a **baseline
strategy** over the same procedure, measures named metrics per trial, and
records enough context to reproduce the run (blueprint §16). Strategies are
*registered artifacts* (see :mod:`agency.experiments.registry`) — the model
here never references code paths, which is the structural guarantee against
experiments becoming a code-modification channel.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class ExperimentStatus(StrEnum):
    """Lifecycle of one experiment (never force uncertain into success)."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    CANCELLED = "cancelled"


#: Statuses a runner can end an experiment with.
TERMINAL_STATUSES: frozenset[ExperimentStatus] = frozenset(
    {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.INCONCLUSIVE,
        ExperimentStatus.ERROR,
        ExperimentStatus.CANCELLED,
    }
)


class Verdict(StrEnum):
    """The evaluator's verdict on a comparison — evidence-based, never vibes."""

    SUCCESS = "success"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    CANCELLED = "cancelled"


class TargetRef(BaseModel):
    """A reference to a *registered* strategy artifact, not a code path.

    ``kind`` is a closed vocabulary mirroring the blueprint's artifact
    types (planner policy, research strategy, tool config…). Experiments
    compare registry entries like ``research_strategy:v3`` — never source
    files, never inline code.
    """

    name: str = Field(min_length=1, description="Artifact name, e.g. 'research_strategy'.")
    version: str = Field(min_length=1, description="Registry version, e.g. 'v1'.")

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"

    @field_validator("name", "version")
    @classmethod
    def _no_separators(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or ":" in stripped:
            raise ValueError("name/version must be non-empty and contain no ':'")
        return stripped


class MetricDefinition(BaseModel):
    """One metric an experiment measures (per trial, then aggregated)."""

    name: str = Field(min_length=1)
    direction: Literal["maximize", "minimize"] = "maximize"
    #: Human-readable explanation recorded in the report.
    description: str = ""


class SuccessCriterion(BaseModel):
    """One frozen acceptance rule evaluated on aggregated metrics.

    ``threshold`` is compared against the *candidate minus baseline* delta
    (for ``improves``/``not_regresses``) or against the candidate's absolute
    value (for ``max``/``min``). Criteria are frozen into the experiment
    record before the runner starts (blueprint §5: frozen-before-run).
    """

    kind: Literal["improves", "not_regresses", "max", "min"]
    metric: str = Field(min_length=1)
    #: Minimum delta for ``improves`` (candidate − baseline, relative share),
    #: maximum tolerated regression share for ``not_regresses``, or the
    #: absolute bound for ``max``/``min``.
    threshold: float = 0.0

    @property
    def description(self) -> str:
        if self.kind == "improves":
            return f"{self.metric} improves by ≥ {self.threshold:.1%} (relative)"
        if self.kind == "not_regresses":
            return f"{self.metric} regresses by ≤ {self.threshold:.1%} (relative)"
        if self.kind == "max":
            return f"{self.metric} ≤ {self.threshold}"
        return f"{self.metric} ≥ {self.threshold}"


class Hypothesis(BaseModel):
    """A testable claim with explicit success/failure criteria."""

    statement: str = Field(min_length=1)
    rationale: str = ""
    #: What is expected to move, and how.
    expected_effect: str = ""
    #: Frozen acceptance criteria; empty means the evaluator can only ever
    #  return INCONCLUSIVE (no criteria, no verdict).
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    #: Provenance: weakness evidence, prior observations, conversation ids…
    provenance: dict[str, Any] = Field(default_factory=dict)
    #: Self-declared confidence (0–1) at proposal time; recorded, never
    #: treated as evidence.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TrialResult(BaseModel):
    """One measured execution of one arm of the experiment."""

    trial_id: str = Field(default_factory=_new_id)
    arm: Literal["baseline", "candidate"]
    index: int = Field(ge=0, description="Trial number within the arm (0-based).")
    metrics: dict[str, float] = Field(default_factory=dict)
    #: True when the trial's procedure completed without raising.
    success: bool = True
    error: str | None = None
    duration_ms: int = 0
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArmSummary(BaseModel):
    """Aggregated metrics for one arm across its trials."""

    ref: str
    trials: int = 0
    failed_trials: int = 0
    mean: dict[str, float] = Field(default_factory=dict)
    stddev: dict[str, float] = Field(default_factory=dict)
    min: dict[str, float] = Field(default_factory=dict)
    max: dict[str, float] = Field(default_factory=dict)


class ComparisonReport(BaseModel):
    """Descriptive baseline-vs-candidate comparison.

    Deliberately **no significance claims**: differences are reported as
    measured deltas with counts. Claiming statistical significance requires
    an actual statistical test (future phase); until then the report says
    ``significance_tested: False`` and the evaluator treats small deltas
    against small trial counts as INCONCLUSIVE evidence, not proof.
    """

    baseline: ArmSummary
    candidate: ArmSummary
    #: candidate − baseline per metric (sign follows ``direction``-agnostic
    #: raw difference; interpretation belongs to the criteria).
    delta: dict[str, float] = Field(default_factory=dict)
    #: Relative change per metric: delta / |baseline| when baseline ≠ 0.
    relative: dict[str, float] = Field(default_factory=dict)
    #: Metrics where the candidate moved against the metric's direction.
    regressions: list[str] = Field(default_factory=list)
    #: Metrics where the candidate moved with the metric's direction.
    improvements: list[str] = Field(default_factory=list)
    significance_tested: bool = False
    notes: list[str] = Field(default_factory=list)


class ExperimentEvaluation(BaseModel):
    """Verdict + evidence from the evaluation engine."""

    verdict: Verdict
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Per-criterion outcome: description → passed/failed/skipped + detail.
    criteria_results: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    explanation: str = ""
    #: ``deterministic`` (metrics + criteria) or ``llm`` (supplementary
    #: judgment — recorded, never decisive).
    source: Literal["deterministic", "llm"] = "deterministic"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Experiment(BaseModel):
    """One controlled comparison, from creation to terminal status."""

    id: str = Field(default_factory=_new_id)
    name: str = Field(min_length=1)
    description: str = ""
    objective: str = ""
    hypothesis: Hypothesis
    baseline: TargetRef
    candidate: TargetRef
    #: Metric definitions the procedure must report per trial.
    metrics: list[MetricDefinition] = Field(default_factory=list)
    #: Frozen acceptance criteria (copied from the hypothesis at creation).
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    #: Repeatability context: trials, seed, budgets (blueprint §16).
    trials: int = Field(default=3, ge=1, le=100)
    seed: int | None = None
    per_trial_timeout_seconds: float = Field(default=60.0, ge=0.1, le=3600.0)
    max_total_seconds: float = Field(default=600.0, ge=1.0, le=86_400.0)
    #: Procedure parameters handed to each trial (data, never code).
    procedure_params: dict[str, Any] = Field(default_factory=dict)
    parent_goal_id: str | None = None
    run_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    trials_results: list[TrialResult] = Field(default_factory=list)
    comparison: ComparisonReport | None = None
    evaluation: ExperimentEvaluation | None = None
    #: Environment snapshot for reproducibility (python, os, package…).
    environment: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def criteria_frozen(self) -> None:
        """Freeze the hypothesis' criteria into the experiment (once).

        Called by the runner before the first trial; after freezing, later
        criteria edits cannot change the verdict (frozen-before-run).
        """
        if not self.success_criteria:
            self.success_criteria = list(self.hypothesis.success_criteria)


class ExperimentSummary(BaseModel):
    """Compact listing entry for ``experiment list``."""

    id: str
    name: str
    status: ExperimentStatus
    baseline: str
    candidate: str
    verdict: Verdict | None = None
    trials: int = 0
    created_at: datetime
    completed_at: datetime | None = None


__all__ = [
    "TERMINAL_STATUSES",
    "ArmSummary",
    "ComparisonReport",
    "Experiment",
    "ExperimentEvaluation",
    "ExperimentStatus",
    "ExperimentSummary",
    "Hypothesis",
    "MetricDefinition",
    "SuccessCriterion",
    "TargetRef",
    "TrialResult",
    "Verdict",
]
