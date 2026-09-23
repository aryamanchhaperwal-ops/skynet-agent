"""Self-improvement domain models (blueprint Phase P7).

Everything in this package is a **data artifact** with provenance. The
engine never touches production source code: a candidate strategy is the
*parent strategy's own registered procedure* plus new frozen configuration,
and an accepted candidate becomes a new version in the strategy registry —
never a code rewrite. External suggestions (web pages, other AI systems)
enter through the exact same pipeline as internal detections and are never
trusted instructions (see :mod:`agency.improve.security`).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agency.experiments.models import MetricDefinition, SuccessCriterion


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


# -- Weaknesses -----------------------------------------------------------------


class WeaknessCategory(StrEnum):
    """Closed vocabulary of what a detector may report (no invented kinds)."""

    TASK_FAILURE = "task_failure"
    LOW_EVALUATION = "low_evaluation"
    HIGH_LATENCY = "high_latency"
    EXCESSIVE_TOOL_CALLS = "excessive_tool_calls"
    POOR_RESEARCH_COVERAGE = "poor_research_coverage"
    DUPLICATE_SOURCES = "duplicate_sources"
    PLANNING_FAILURE = "planning_failure"
    MEMORY_RETRIEVAL_FAILURE = "memory_retrieval_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    PERFORMANCE_REGRESSION = "performance_regression"


class WeaknessStatus(StrEnum):
    OPEN = "open"
    HYPOTHESIZED = "hypothesized"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class EvidenceItem(BaseModel):
    """One piece of evidence backing a weakness (never a vibe)."""

    summary: str = Field(min_length=1)
    #: Where it came from: experience id, experiment id, run id, metric name…
    source: str = ""
    #: Any structured detail (counts, values, ids) that supports the claim.
    detail: dict[str, Any] = Field(default_factory=dict)


class Weakness(BaseModel):
    """A recurring problem detected in *actual recorded performance data*."""

    id: str = Field(default_factory=_new_id)
    category: WeaknessCategory
    description: str = Field(min_length=1)
    evidence: list[EvidenceItem] = Field(min_length=1)
    affected_component: str = ""
    affected_strategy: str = ""
    #: How many supporting records the detector actually saw.
    frequency: int = Field(default=1, ge=1)
    #: 0–1, deterministic from category + frequency (see detector).
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    related_runs: list[str] = Field(default_factory=list)
    related_experiments: list[str] = Field(default_factory=list)
    status: WeaknessStatus = WeaknessStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def transition(self, status: WeaknessStatus) -> None:
        self.status = status
        self.updated_at = _utcnow()


# -- Hypotheses -------------------------------------------------------------------


class ImprovementHypothesis(BaseModel):
    """A testable claim about how Skynet could improve.

    Created deterministically from a weakness by the builder in
    :mod:`agency.improve.hypothesis`; an LLM may *draft* the statement text
    (future phase), but the structured claim, criteria and provenance are
    built here and validated mechanically.
    """

    id: str = Field(default_factory=_new_id)
    weakness_id: str
    statement: str = Field(min_length=1)
    rationale: str = ""
    #: The metric expected to move, and how.
    expected_improvement: str = ""
    affected_component: str = ""
    #: Fully-formed P6 hypothesis: baseline vs candidate targets, frozen
    #: criteria, metrics — the pipeline hands it straight to the runner.
    experiment_spec: dict[str, Any] = Field(default_factory=dict)
    #: Deterministic acceptance rules the proposal inherits.
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    regression_criteria: list[SuccessCriterion] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# -- Proposals ----------------------------------------------------------------------


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED_FOR_TEST = "approved_for_test"
    TESTING = "testing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class ExperimentPlan(BaseModel):
    """The benchmark plan a proposal carries into the P6 runner."""

    metrics: list[MetricDefinition] = Field(default_factory=list)
    trials: int = Field(default=3, ge=1, le=100)
    seed: int | None = None
    per_trial_timeout_seconds: float = Field(default=60.0, ge=0.1, le=3600.0)
    max_total_seconds: float = Field(default=600.0, ge=1.0, le=86_400.0)
    #: Data-only parameters passed to both arms' procedures.
    procedure_params: dict[str, Any] = Field(default_factory=dict)
    #: Named benchmark description (informational; the plan *is* the spec).
    benchmark: str = ""


class ImprovementProposal(BaseModel):
    """The full audit record: weakness → hypothesis → change → plan → risk."""

    id: str = Field(default_factory=_new_id)
    weakness_id: str
    hypothesis_id: str
    #: ``name:version`` of the baseline strategy (registry artifact).
    target: str
    #: Data-only description of the change (the candidate's config payload).
    proposed_change: str = ""
    rationale: str = ""
    expected_benefit: str = ""
    risks: list[str] = Field(default_factory=list)
    experiment_plan: ExperimentPlan = Field(default_factory=ExperimentPlan)
    acceptance_criteria: list[SuccessCriterion] = Field(default_factory=list)
    rollback_plan: str = "reactivate the previous known-good version via the version ledger"
    #: ``internal`` (detector) or ``external`` (web/AI suggestion) — external
    #: suggestions are recorded, quarantined through the same gates, never
    #: trusted more than internal ones.
    origin: Literal["internal", "external"] = "internal"
    proposed_by: str = "skynet"
    status: ProposalStatus = ProposalStatus.PROPOSED
    #: Filled by the pipeline as the lifecycle advances.
    candidate_ref: str | None = None
    experiment_id: str | None = None
    version_id: str | None = None
    decision: str | None = None
    decision_reason: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def transition(self, status: ProposalStatus) -> None:
        self.status = status
        self.updated_at = _utcnow()


# -- Candidates ------------------------------------------------------------------------


class CandidateStrategy(BaseModel):
    """An alternative version of an existing strategy — **config, not code**.

    The candidate reuses the parent's registered procedure and runs under
    new frozen configuration inside the P6 sandbox. It never names source
    files, imports, or executables; ``changes`` is free-text *description*
    for the audit trail, and the effect lives entirely in ``config``.
    """

    id: str = Field(default_factory=_new_id)
    #: ``name:version`` of the parent (must be registered).
    parent_ref: str
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    #: Human-readable change description (data artifact, not executable).
    changes: str = ""
    #: The new frozen configuration both arms see in their own shape.
    config: dict[str, Any] = Field(default_factory=dict)
    proposal_id: str | None = None
    hypothesis_id: str | None = None
    weakness_id: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"


# -- Versions ----------------------------------------------------------------------------


class StrategyVersion(BaseModel):
    """One accepted strategy version in the ledger (append-only history)."""

    id: str = Field(default_factory=_new_id)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parent_version: str | None = None
    change_description: str = ""
    proposal_id: str | None = None
    experiment_id: str | None = None
    #: Measured evidence at acceptance time (aggregated means/deltas).
    benchmark_results: dict[str, Any] = Field(default_factory=dict)
    acceptance: dict[str, Any] = Field(default_factory=dict)
    #: Full config that defines this version (reproducibility).
    config: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"


class ActiveStrategy(BaseModel):
    """The currently-active version pointer for one strategy name."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    version_id: str | None = None
    activated_at: datetime = Field(default_factory=_utcnow)
    activated_from: str | None = None  # prior version ref, for audit

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"


class RegressionEvent(BaseModel):
    """Post-deployment monitoring finding: actual < expected."""

    id: str = Field(default_factory=_new_id)
    version_ref: str
    proposal_id: str | None = None
    metric: str = Field(min_length=1)
    expected: float
    actual: float
    detail: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


__all__ = [
    "ActiveStrategy",
    "CandidateStrategy",
    "EvidenceItem",
    "ExperimentPlan",
    "ImprovementHypothesis",
    "ImprovementProposal",
    "ProposalStatus",
    "RegressionEvent",
    "StrategyVersion",
    "Weakness",
    "WeaknessCategory",
    "WeaknessStatus",
]
