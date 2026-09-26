"""Autonomous-run domain models (Phase P8 integration).

A run is a **persistent, resumable artifact**: objective, stage, iteration
counter, budgets, pending decisions and provenance survive process
restarts. The orchestrator never trusts a "current" in-memory world —
every stage reads and writes the persisted record, which is what makes
pause/resume/cancel across processes correct rather than best-effort.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class RunStatus(StrEnum):
    """Lifecycle of one autonomous run (spec §3)."""

    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"  # human checkpoint (e.g. improvement approval)
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class RunStage(StrEnum):
    """The explicit autonomous state machine (spec §4)."""

    INITIALIZE = "initialize"
    OBSERVE = "observe"
    PLAN = "plan"
    RESEARCH = "research"
    COMMUNICATE = "communicate"
    ACT = "act"
    EVALUATE = "evaluate"
    LEARN = "learn"
    IMPROVE = "improve"
    COMPLETE = "complete"


class BudgetDecision(StrEnum):
    """What a budget check tells the loop to do (spec §14: pause, never run on)."""

    PROCEED = "proceed"
    PAUSE = "pause"
    STOP = "stop"


class Evidence(BaseModel):
    """One provenance-carrying observation (spec §9).

    Web pages, AI responses, memories and experiments all become
    Evidence — distinguished by ``source_type``, never collapsed into
    fact. Content is DATA: it can never carry instructions for the
    orchestrator (enforced by the security contract in the security
    module, tested in the suite).
    """

    id: str = Field(default_factory=_new_id)
    source: str = ""
    source_type: Literal["web", "ai", "memory", "experiment", "perception", "action"] = "perception"
    content: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=_utcnow)


class CoreRunSummary(BaseModel):
    """Outcome of one core agent cycle, kept on the run for evaluation."""

    run_id: str
    status: str = ""
    ok: bool = False
    recorded_at: datetime = Field(default_factory=_utcnow)


class IterationPlan(BaseModel):
    """The per-iteration working plan the orchestrator executes."""

    iteration: int = Field(default=0, ge=0)
    research_queries: list[str] = Field(default_factory=list)
    ai_questions: list[str] = Field(default_factory=list)
    #: Evidence that answers the objective, accumulated this iteration.
    focus: str = ""
    #: Explicit stop condition the evaluator checks.
    sufficient_evidence: bool = False


class StageDecision(BaseModel):
    """What the evaluation stage instructs next (spec §11)."""

    action: Literal[
        "continue",
        "replan",
        "research_more",
        "ask_external_ai",
        "run_experiment",
        "improvement_opportunity",
        "complete",
        "failed",
    ] = "continue"
    reason: str = ""
    #: Optional focus text feeding the next iteration's plan.
    next_focus: str = ""


class ApprovalRequest(BaseModel):
    """A recorded human checkpoint (spec §17)."""

    id: str = Field(default_factory=_new_id)
    run_id: str
    what: str
    why: str
    proposed_action: str = ""
    evidence: list[str] = Field(default_factory=list)
    status: Literal["pending", "approved", "rejected"] = "pending"
    requested_at: datetime = Field(default_factory=_utcnow)
    decided_at: datetime | None = None
    decided_by: str | None = None


class OrchestratorBudgets(BaseModel):
    """Hard limits for one autonomous run (spec §14)."""

    max_iterations: int = Field(default=3, ge=1, le=100)
    max_runtime_seconds: float = Field(default=600.0, ge=1.0, le=86_400.0)
    max_web_requests: int = Field(default=10, ge=0, le=1000)
    #: External AI (AI↔AI) request ceiling — the comms seam.
    max_ai_requests: int = Field(default=6, ge=0, le=1000)
    #: Provider LLM call ceiling (planning, evaluation, synthesis) — every
    #: call through the intelligence service, counted by the orchestrator.
    max_llm_calls: int = Field(default=200, ge=0, le=100_000)
    #: Optional token ceiling (prompt+completion) when the provider reports
    #: usage; None disables the check rather than guessing a number.
    max_tokens: int | None = Field(default=None, ge=0)
    max_experiments: int = Field(default=2, ge=0, le=100)
    #: Optional cost ceiling; adapters that report cost stop above it.
    max_cost_usd: float | None = None

    def check(
        self,
        *,
        iteration: int,
        runtime_seconds: float,
        web_requests: int,
        ai_requests: int,
        experiments: int,
        cost_usd: float = 0.0,
        llm_calls: int = 0,
        tokens: int = 0,
    ) -> BudgetDecision:
        """Return PROCEED/PAUSE/STOP from current usage."""
        if self.max_cost_usd is not None and cost_usd > self.max_cost_usd:
            return BudgetDecision.PAUSE
        if self.max_tokens is not None and tokens > self.max_tokens:
            return BudgetDecision.PAUSE
        if llm_calls > self.max_llm_calls:
            return BudgetDecision.PAUSE
        if experiments > self.max_experiments:
            return BudgetDecision.PAUSE
        if ai_requests > self.max_ai_requests:
            return BudgetDecision.PAUSE
        if web_requests > self.max_web_requests:
            return BudgetDecision.PAUSE
        if runtime_seconds > self.max_runtime_seconds:
            return BudgetDecision.PAUSE
        if iteration >= self.max_iterations:
            return BudgetDecision.STOP
        return BudgetDecision.PROCEED


class AutonomousRun(BaseModel):
    """One resumable autonomous run (spec §3)."""

    id: str = Field(default_factory=_new_id)
    objective: str = Field(min_length=1)
    current_state: str = "active"
    current_stage: RunStage = RunStage.INITIALIZE
    iteration: int = Field(default=0, ge=0)
    status: RunStatus = RunStatus.RUNNING
    parent_run_id: str | None = None
    #: Working plan + evidence accumulated across iterations.
    plan: IterationPlan = Field(default_factory=IterationPlan)
    evidence: list[Evidence] = Field(default_factory=list)
    #: Stage decisions trail (one per completed evaluation).
    decisions: list[StageDecision] = Field(default_factory=list)
    #: Pending human checkpoint, if any (status becomes WAITING).
    pending_approval: ApprovalRequest | None = None
    #: Persisted sub-system references for resume: core run ids,
    #: experiment ids, improvement proposal ids.
    core_run_ids: list[str] = Field(default_factory=list)
    #: Core-cycle outcome summaries (status + ok flag) for mechanical evaluation.
    core_summaries: list[CoreRunSummary] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    #: Active strategy versions at run start (audit; never mutated here).
    active_versions: dict[str, str] = Field(default_factory=dict)
    #: Security findings from scanning external content (web pages, AI
    #: responses) before they influenced the run. Directive-like fragments
    #: are recorded here as DATA — never executed, never acted upon.
    security_report: dict[str, Any] = Field(default_factory=dict)
    budgets: OrchestratorBudgets = Field(default_factory=OrchestratorBudgets)
    #: Actual usage counters the budgets check.
    usage: dict[str, float] = Field(
        default_factory=lambda: {
            "iterations": 0.0,
            "runtime_seconds": 0.0,
            "web_requests": 0.0,
            "ai_requests": 0.0,
            "llm_calls": 0.0,
            "tokens": 0.0,
            "experiments": 0.0,
            "cost_usd": 0.0,
        }
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    #: Cumulative error/notice log (failure recovery, spec §16).
    errors: list[str] = Field(default_factory=list)
    #: Final result payload (completed runs).
    result: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def touch(self) -> None:
        self.updated_at = _utcnow()

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.ROLLED_BACK,
        }

    def add_error(self, message: str) -> None:
        self.errors.append(f"{_utcnow().isoformat()} {message}")
        del self.errors[50:]  # bounded


__all__ = [
    "ApprovalRequest",
    "AutonomousRun",
    "BudgetDecision",
    "Evidence",
    "IterationPlan",
    "OrchestratorBudgets",
    "RunStage",
    "RunStatus",
    "StageDecision",
]
