"""Improvement pipeline — weakness → hypothesis → proposal → experiment → decision.

The pipeline composes the existing P6 experimentation engine; it owns **no
execution machinery of its own**: every benchmark is an
:class:`~agency.experiments.models.Experiment` run by the one-and-only
:class:`~agency.experiments.runner.ExperimentRunner` against the
:class:`~agency.experiments.registry.StrategyRegistry`.

Stages (each idempotent at its own step, each trace-emitting):

- :meth:`detect`    — recorded data → evidence-backed weaknesses
- :meth:`propose`   — weakness → hypothesis + proposal + candidate (data only)
- :meth:`test`      — register the candidate, run baseline vs candidate
- :meth:`evaluate`  — measured verdict → ``accept``/``reject``/``inconclusive``
- :meth:`approve`   — human approval to apply (required; nothing self-applies)
- :meth:`apply`     — persist the accepted version in the ledger + registry
- :meth:`rollback`  — reactivate the previous known-good version
- :meth:`monitor`   — expected vs actual performance; regression events

Safety invariants (enforced structurally, tested explicitly):

- candidates are parent procedure + frozen config — never code;
- no stage mutates production configuration or the loop;
- ``apply`` refuses proposals without an explicit ``approve`` record;
- rollback preserves the failed version (nothing is deleted, ever).
"""

from __future__ import annotations

import logging
from typing import Any

from agency.experiments.experiments_store import ExperimentRegistry
from agency.experiments.models import (
    Experiment,
    ExperimentStatus,
    MetricDefinition,
    SuccessCriterion,
    TargetRef,
    Verdict,
)
from agency.experiments.models import (
    Hypothesis as ExperimentHypothesis,
)
from agency.experiments.registry import StrategyRegistry
from agency.experiments.runner import ExperimentRunner
from agency.improve.detector import DetectorInput, WeaknessDetector
from agency.improve.memory_bridge import record_proposal
from agency.improve.models import (
    ActiveStrategy,
    CandidateStrategy,
    ExperimentPlan,
    ImprovementHypothesis,
    ImprovementProposal,
    ProposalStatus,
    RegressionEvent,
    StrategyVersion,
    Weakness,
    WeaknessStatus,
)
from agency.improve.store import ImprovementStore

logger = logging.getLogger("skynet.improve.pipeline")


class ImprovementPipeline:
    """The self-improvement loop's operator-facing engine."""

    def __init__(
        self,
        *,
        strategies: StrategyRegistry,
        runner: ExperimentRunner,
        experiments: ExperimentRegistry,
        memory: Any = None,
        emit: Any = None,
        detector: WeaknessDetector | None = None,
        store: ImprovementStore | None = None,
    ) -> None:
        self._strategies = strategies
        self._runner = runner
        self._experiments = experiments
        self._memory = memory
        self._emit = emit
        self._detector = detector or WeaknessDetector()
        # History: weaknesses, proposals, version ledger, active pointers,
        # regressions. Persisted when the store has a path (bootstrap/CLI);
        # in-process only when constructed with none (tests).
        self._store = store or ImprovementStore(path=None)

    # -- weak event emission (one stream, never a second logger) --------------

    async def _trace(self, event: str, **payload: Any) -> None:
        if self._emit is None:
            return
        try:
            await self._emit(event, **payload)
        except Exception:
            logger.exception("improvement trace emission failed for %s", event)

    # -- weakness detection ------------------------------------------------------

    async def detect(self, data: DetectorInput) -> list[Weakness]:
        """Detect weaknesses from recorded data; persists new ones."""
        weaknesses = self._detector.detect(data)
        known = {
            (w.category.value, w.affected_component, w.description)
            for w in self._store.weaknesses()
            if w.status is WeaknessStatus.OPEN
        }
        detected: list[Weakness] = []
        for weakness in weaknesses:
            key = (weakness.category.value, weakness.affected_component, weakness.description)
            if key in known:
                continue
            await self._store.save_weakness(weakness)
            detected.append(weakness)
            await self._trace(
                "WEAKNESS_DETECTED",
                weakness_id=weakness.id,
                category=weakness.category.value,
                severity=weakness.severity,
                frequency=weakness.frequency,
            )
        return detected

    async def record_weakness(self, weakness: Weakness) -> Weakness:
        """Record a weakness from any source (operator, external suggestion).

        This is the intake for suggestions arriving from web pages or other
        AI systems: they become ordinary weaknesses and pass through the
        exact same proposal → benchmark → approval pipeline as internal
        detections — recorded data, never trusted instructions.
        """
        await self._store.save_weakness(weakness)
        await self._trace(
            "WEAKNESS_DETECTED",
            weakness_id=weakness.id,
            category=weakness.category.value,
            severity=weakness.severity,
            frequency=weakness.frequency,
        )
        return weakness

    def weakness(self, weakness_id: str) -> Weakness:
        return self._store.weakness(weakness_id)

    def list_weaknesses(self, status: str | None = None) -> list[Weakness]:
        return self._store.weaknesses(status=status)

    # -- hypothesis + proposal + candidate (all data artifacts) --------------------

    async def propose(
        self,
        weakness_id: str,
        *,
        proposed_by: str = "skynet",
        origin: str = "internal",
        candidate_config: dict[str, Any] | None = None,
        candidate_changes: str = "",
        plan_overrides: dict[str, Any] | None = None,
    ) -> tuple[ImprovementHypothesis, ImprovementProposal, CandidateStrategy]:
        """Build the improvement artifacts for one weakness.

        ``origin='external'`` marks suggestions arriving from web pages or
        external AI systems. They go through **exactly the same** pipeline:
        recorded, gated, benchmarked — and never trusted as instructions.
        """
        weakness = self.weakness(weakness_id)
        priors = self._store.prior_proposals_for(weakness.id)
        hypothesis, proposal, candidate = build_proposal_artifacts(
            weakness,
            self._strategies,
            proposed_by=proposed_by,
            origin=origin,
            candidate_config=candidate_config,
            candidate_changes=candidate_changes,
            plan_overrides=plan_overrides,
            attempt=len(priors),
        )
        if priors:
            # §20: record, don't hide, prior attempts at this weakness. The
            # operator (or future engine) decides whether to proceed.
            proposal.metadata["prior_attempts"] = [
                {"proposal_id": p.id, "status": p.status.value, "decision": p.decision}
                for p in priors
            ]
        weakness.transition(WeaknessStatus.HYPOTHESIZED)
        await self._store.save_weakness(weakness)
        await self._store.save_proposal(proposal)
        await self._store.save_candidate(candidate)
        await self._trace(
            "HYPOTHESIS_CREATED",
            weakness_id=weakness.id,
            hypothesis_id=hypothesis.id,
            statement=hypothesis.statement,
        )
        await self._trace(
            "IMPROVEMENT_PROPOSED",
            proposal_id=proposal.id,
            weakness_id=weakness.id,
            hypothesis_id=hypothesis.id,
            target=proposal.target,
            origin=proposal.origin,
        )
        await self._trace(
            "CANDIDATE_CREATED",
            proposal_id=proposal.id,
            candidate_id=candidate.id,
            parent=candidate.parent_ref,
            candidate_ref=candidate.ref,
        )
        return hypothesis, proposal, candidate

    # -- candidate testing (the only benchmark path: the P6 runner) ---------------

    async def test(self, proposal: ImprovementProposal, candidate: CandidateStrategy) -> Experiment:
        """Register the candidate and run the planned experiment.

        Idempotent at its own step: a proposal already past TESTING keeps
        its experiment id and is not re-run.
        """
        if proposal.experiment_id:
            existing = await self._experiments.get(proposal.experiment_id)
            if existing is not None and existing.status in {
                ExperimentStatus.COMPLETED,
                ExperimentStatus.FAILED,
                ExperimentStatus.INCONCLUSIVE,
                ExperimentStatus.ERROR,
                ExperimentStatus.CANCELLED,
            }:
                return existing

        plan = proposal.experiment_plan
        parent = self._strategies.get(candidate.parent_ref)
        # Candidate = parent's own registered procedure + new frozen config.
        self._strategies.register(
            name=candidate.name,
            version=candidate.version,
            capability=parent.capability,
            procedure=parent.procedure,
            description=f"candidate of {candidate.parent_ref}: {candidate.changes}",
            config=dict(candidate.config),
        )
        proposal.candidate_ref = candidate.ref
        proposal.transition(ProposalStatus.APPROVED_FOR_TEST)
        proposal.transition(ProposalStatus.TESTING)

        experiment = Experiment(
            name=f"improvement:{proposal.id[:12]}",
            description=f"improvement test for proposal {proposal.id}",
            objective=proposal.expected_benefit,
            hypothesis=ExperimentHypothesis(
                statement=f"{_hypothesis_statement(proposal)}",
                rationale=proposal.rationale,
                success_criteria=[
                    SuccessCriterion.model_validate(c.model_dump())
                    for c in proposal.acceptance_criteria
                ],
                provenance={
                    "proposal_id": proposal.id,
                    "hypothesis_id": proposal.hypothesis_id,
                    "weakness_id": proposal.weakness_id,
                    "origin": proposal.origin,
                },
                confidence=None,
            ),
            baseline=TargetRef(
                name=candidate.parent_ref.split(":")[0],
                version=candidate.parent_ref.split(":")[1],
            ),
            candidate=TargetRef(name=candidate.name, version=candidate.version),
            metrics=plan.metrics,
            trials=plan.trials,
            seed=plan.seed,
            per_trial_timeout_seconds=plan.per_trial_timeout_seconds,
            max_total_seconds=plan.max_total_seconds,
            procedure_params=dict(plan.procedure_params),
        )
        proposal.experiment_id = experiment.id
        await self._store.save_proposal(proposal)
        await self._trace(
            "CANDIDATE_TEST_STARTED",
            proposal_id=proposal.id,
            candidate_id=candidate.id,
            experiment_id=experiment.id,
        )
        await self._trace(
            "BENCHMARK_STARTED",
            experiment_id=experiment.id,
            benchmark=plan.benchmark or "proposal plan",
        )
        finished = await self._runner.run(experiment)
        await self._experiments.save(finished)
        await self._trace(
            "BENCHMARK_COMPLETED",
            experiment_id=finished.id,
            status=finished.status.value,
            verdict=finished.evaluation.verdict.value if finished.evaluation else None,
        )
        await self._trace(
            "CANDIDATE_TEST_COMPLETED",
            proposal_id=proposal.id,
            candidate_id=candidate.id,
            experiment_id=finished.id,
            verdict=finished.evaluation.verdict.value if finished.evaluation else None,
        )
        await self._store.save_proposal(proposal)
        return finished

    # -- acceptance ------------------------------------------------------------------

    async def evaluate(
        self, proposal: ImprovementProposal, candidate: CandidateStrategy, experiment: Experiment
    ) -> dict[str, Any]:
        """Mechanical acceptance decision from measured evidence.

        Returns a report dict and records the decision on the proposal:
        ``accepted`` requires Verdict.SUCCESS from the P6 evaluator (all
        frozen criteria passed on real data). Anything else records the
        reason and moves the proposal to REJECTED — no forcing.
        """
        verdict = experiment.evaluation.verdict if experiment.evaluation else Verdict.ERROR
        verdict_value = verdict.value
        criteria_results = (
            experiment.evaluation.criteria_results if experiment.evaluation else []
        )
        evidence: list[str] = (
            list(experiment.evaluation.evidence) if experiment.evaluation else []
        )
        explanation = (
            experiment.evaluation.explanation if experiment.evaluation else "no evaluation"
        )
        await self._trace(
            "ACCEPTANCE_EVALUATED",
            proposal_id=proposal.id,
            experiment_id=experiment.id,
            verdict=verdict_value,
        )
        proposal.decision = "accepted" if verdict is Verdict.SUCCESS else "rejected"
        proposal.decision_reason = explanation
        if verdict is Verdict.SUCCESS:
            proposal.transition(ProposalStatus.ACCEPTED)
            await self._trace(
                "IMPROVEMENT_ACCEPTED",
                proposal_id=proposal.id,
                candidate_ref=candidate.ref,
                experiment_id=experiment.id,
            )
            await self._trace(
                "HUMAN_APPROVAL_REQUIRED",
                proposal_id=proposal.id,
                candidate_ref=candidate.ref,
                change=proposal.proposed_change,
                rollback_plan=proposal.rollback_plan,
            )
        else:
            proposal.transition(ProposalStatus.REJECTED)
            await self._trace(
                "IMPROVEMENT_REJECTED",
                proposal_id=proposal.id,
                candidate_ref=candidate.ref,
                experiment_id=experiment.id,
                reason=explanation,
            )
        await self._store.save_proposal(proposal)
        await record_proposal(
            proposal,
            weakness=self._store.weakness_or_none(proposal.weakness_id),
            candidate=candidate,
            experiment=experiment,
            memory=self._memory,
        )
        return {
            "proposal_id": proposal.id,
            "verdict": verdict_value,
            "decision": proposal.decision,
            "reason": explanation,
            "criteria_results": criteria_results,
            "evidence": evidence,
        }

    # -- human approval + application --------------------------------------------------

    async def approve(
        self,
        proposal: ImprovementProposal,
        *,
        approved_by: str,
        note: str = "",
    ) -> None:
        """Record explicit human approval (the only path to ``apply``).

        Nothing in Skynet calls this automatically — the operator does, via
        the CLI/API, after reading the evidence.
        """
        if proposal.status is not ProposalStatus.ACCEPTED:
            raise ValueError(
                f"only ACCEPTED proposals can be approved "
                f"(status={proposal.status.value}, decision={proposal.decision})"
            )
        proposal.metadata["approval"] = {
            "approved_by": approved_by,
            "note": note,
            "at": _utcnow_iso(),
        }
        # Approval is a gate record; the proposal stays ACCEPTED until apply.
        await self._store.save_proposal(proposal)

    async def apply(
        self,
        proposal: ImprovementProposal,
        candidate: CandidateStrategy,
        experiment: Experiment,
    ) -> StrategyVersion:
        """Persist the accepted candidate as a version (never a code change).

        Requires a recorded human approval. The version carries the full
        config, benchmark evidence and provenance; the strategy registry
        keeps running the same procedure under the new version ref.
        """
        approval = proposal.metadata.get("approval")
        if not approval:
            raise PermissionError(
                "apply requires recorded human approval (improve approve first)"
            )
        if proposal.status is not ProposalStatus.ACCEPTED:
            raise ValueError(f"proposal status is {proposal.status.value}, not accepted")

        self._strategies.get(candidate.parent_ref)  # parent must be registered
        version = StrategyVersion(
            name=candidate.name,
            version=candidate.version,
            parent_version=candidate.parent_ref.split(":")[-1],
            change_description=candidate.changes or proposal.proposed_change,
            proposal_id=proposal.id,
            experiment_id=experiment.id,
            benchmark_results={
                "baseline_mean": (
                    experiment.comparison.baseline.mean if experiment.comparison else {}
                ),
                "candidate_mean": (
                    experiment.comparison.candidate.mean if experiment.comparison else {}
                ),
                "delta": experiment.comparison.delta if experiment.comparison else {},
                "verdict": experiment.evaluation.verdict.value if experiment.evaluation else None,
            },
            acceptance={
                "decision": proposal.decision,
                "reason": proposal.decision_reason,
                **approval,
            },
            config=dict(candidate.config),
            provenance={
                "weakness_id": proposal.weakness_id,
                "hypothesis_id": proposal.hypothesis_id,
                "candidate_id": candidate.id,
                "parent_ref": candidate.parent_ref,
                "capability": self._strategies.get(candidate.parent_ref).capability,
                "procedure": "parent procedure reused (config-only change)",
            },
        )
        await self._store.save_version(version)
        active = ActiveStrategy(
            name=version.name,
            version=version.version,
            version_id=version.id,
            activated_from=self._store.active(version.name).ref
            if self._store.active(version.name)
            else None,
        )
        await self._store.save_active(active)
        proposal.version_id = version.id
        await self._store.save_proposal(proposal)
        await self._trace(
            "IMPROVEMENT_APPLIED",
            proposal_id=proposal.id,
            version_id=version.id,
            version=version.ref,
            previous=active.activated_from,
        )
        return version

    async def seed_baseline(
        self,
        *,
        name: str,
        version: str,
        config: dict[str, Any] | None = None,
        description: str = "initial known-good version",
    ) -> StrategyVersion:
        """Record the currently-active production version in the ledger.

        Operators call this once per strategy so rollback always has a
        known-good version to restore. It creates no proposal and runs
        nothing — it is bookkeeping for the version tree.
        """
        version_record = StrategyVersion(
            name=name,
            version=version,
            parent_version=None,
            change_description=description,
            config=dict(config or {}),
            provenance={"seeded": True, "by": "operator"},
        )
        await self._store.save_version(version_record)
        await self._store.save_active(
            ActiveStrategy(name=name, version=version, version_id=version_record.id)
        )
        return version_record

    # -- rollback ------------------------------------------------------------------------

    async def rollback(
        self,
        proposal: ImprovementProposal,
        *,
        reason: str,
        record_rollback: bool = True,
    ) -> ActiveStrategy:
        """Restore the previous known-good version for the proposal's strategy.

        The rolled-back candidate version stays in the ledger (history is
        never deleted); only the active pointer moves.
        """
        if not proposal.version_id:
            raise ValueError("proposal has no applied version to roll back")
        candidate_ref = proposal.candidate_ref or ""
        name = candidate_ref.split(":")[0] if candidate_ref else _name_from_ref(proposal.target)
        versions = self._store.versions(name)
        rolled_back = next((v for v in versions if v.id == proposal.version_id), None)
        if rolled_back is None:
            raise ValueError(
                f"version {proposal.version_id} not found in ledger for {name!r}"
            )
        history = [v for v in versions if v.id != proposal.version_id]
        if not history:
            raise ValueError(
                f"no previous known-good version exists for {name!r}; "
                "cannot roll back (nothing to restore)"
            )
        previous = history[-1]
        active = ActiveStrategy(
            name=name,
            version=previous.version,
            version_id=previous.id,
            activated_from=rolled_back.ref,
        )
        await self._store.save_active(active)
        proposal.transition(ProposalStatus.ROLLED_BACK)
        await self._store.save_proposal(proposal)
        if record_rollback:
            await self._trace(
                "IMPROVEMENT_ROLLED_BACK",
                proposal_id=proposal.id,
                version_id=proposal.version_id,
                restored=active.ref,
                reason=reason,
            )
        return active

    # -- post-deployment monitoring ---------------------------------------------------------

    async def monitor(
        self,
        proposal: ImprovementProposal,
        *,
        actual: dict[str, float],
        rollback_below: dict[str, float] | None = None,
    ) -> list[RegressionEvent]:
        """Compare expected (benchmark) vs actual production performance.

        ``actual`` maps metric → measured value from live runs. A metric in
        ``rollback_below`` that falls below its floor creates a regression
        event; with ``policy='auto_rollback'`` in metadata the pipeline
        also executes the rollback (explicit, traced, reversible).
        """
        experiment_id = proposal.experiment_id
        benchmark: dict[str, float] = {}
        if experiment_id:
            record = await self._experiments.get(experiment_id)
            if record is not None and record.comparison is not None:
                benchmark = record.comparison.candidate.mean
        events: list[RegressionEvent] = []
        for metric, value in actual.items():
            floor = (rollback_below or {}).get(metric)
            if floor is None:
                continue
            if value >= floor:
                continue
            event = RegressionEvent(
                version_ref=proposal.candidate_ref or proposal.target,
                proposal_id=proposal.id,
                metric=metric,
                expected=benchmark.get(metric, floor),
                actual=value,
                detail=f"actual {value} below rollback floor {floor}",
            )
            events.append(event)
            await self._store.save_regression(event)
            await self._trace(
                "IMPROVEMENT_MONITORED",
                proposal_id=proposal.id,
                metric=metric,
                expected=event.expected,
                actual=value,
                regression=True,
            )
            if proposal.metadata.get("monitor_policy") == "auto_rollback":
                await self.rollback(proposal, reason=event.detail)
        if not events:
            await self._trace(
                "IMPROVEMENT_MONITORED", proposal_id=proposal.id, regression=False
            )
        return events

    # -- introspection --------------------------------------------------------------------------

    def versions(self, name: str) -> list[StrategyVersion]:
        return self._store.versions(name)

    def active(self, name: str) -> ActiveStrategy | None:
        return self._store.active(name)

    def regressions(self) -> list[RegressionEvent]:
        return self._store.regressions()

    def proposals(self, status: str | None = None) -> list[ImprovementProposal]:
        return self._store.proposals(status=status)

    def candidate_for_proposal(self, proposal_id: str) -> CandidateStrategy:
        candidate = self._store.candidate_for_proposal(proposal_id)
        if candidate is None:
            raise KeyError(f"no candidate recorded for proposal {proposal_id!r}")
        return candidate

    async def experiment_record(self, experiment_id: str):
        """Fetch the experiment record behind a tested proposal."""
        return await self._experiments.get(experiment_id)

    @property
    def memory(self):
        """The memory manager improvement decisions are recorded into (may be None)."""
        return self._memory

    # -- persistence helpers (bootstrap restore) -----------------------------------------------

    def snapshot_registry_strategy(self, ref: str) -> dict[str, Any] | None:
        """Serializable record of a registered strategy (procedure excluded)."""
        try:
            strategy = self._strategies.get(ref)
        except KeyError:
            return None
        return strategy.describe()


# -- pure builders (no I/O) ------------------------------------------------------------------------


def build_proposal_artifacts(
    weakness: Weakness,
    strategies: StrategyRegistry,
    *,
    proposed_by: str = "skynet",
    origin: str = "internal",
    candidate_config: dict[str, Any] | None = None,
    candidate_changes: str = "",
    plan_overrides: dict[str, Any] | None = None,
    attempt: int = 0,
) -> tuple[ImprovementHypothesis, ImprovementProposal, CandidateStrategy]:
    """Deterministic weakness → (hypothesis, proposal, candidate).

    Reuses the P6 registry's registered baseline strategy: capability,
    description and procedure come from it; the candidate differs only in
    frozen configuration. Raises ``KeyError`` if the weakness names no
    registered strategy — the pipeline never invents targets.
    """
    target_ref = weakness.affected_strategy or weakness.affected_component
    if not target_ref or ":" not in target_ref:
        raise KeyError(
            f"weakness {weakness.id} does not name a registered strategy "
            "ref (affected_strategy 'name:version'); nothing to improve"
        )
    parent = strategies.get(target_ref)
    candidate_name = parent.name
    # Repeat attempts get fresh candidate versions (a rejected candidate
    # version stays registered — history is never overwritten).
    candidate_version = (
        f"cand-{weakness.id[:8]}" if attempt <= 0 else f"cand-{weakness.id[:8]}-{attempt + 1}"
    )
    config = dict(candidate_config or {})
    candidate = CandidateStrategy(
        parent_ref=parent.ref,
        name=candidate_name,
        version=candidate_version,
        changes=candidate_changes
        or f"configuration adjustment responding to: {weakness.description}",
        config=config,
        weakness_id=weakness.id,
        provenance={"origin": origin, "proposed_by": proposed_by},
    )
    statement = (
        f"Raising {candidate.ref} configuration ({_describe_config(config)}) will "
        f"improve {weakness.category.value} versus {parent.ref} without "
        "unacceptable regressions."
    )
    hypothesis = ImprovementHypothesis(
        weakness_id=weakness.id,
        statement=statement,
        rationale=(
            f"weakness {weakness.id} ({weakness.category.value}, severity "
            f"{weakness.severity:.2f}, frequency {weakness.frequency})"
        ),
        expected_improvement=weakness.category.value,
        affected_component=target_ref,
        confidence=None,
        evidence=list(weakness.evidence),
        provenance={"origin": origin, "proposed_by": proposed_by},
    )
    overrides = dict(plan_overrides or {})
    plan = ExperimentPlan(
        metrics=overrides.pop("metrics", None) or [
            MetricDefinition(name="score", direction="maximize")
        ],
        trials=overrides.pop("trials", 3),
        seed=overrides.pop("seed", 42),
        procedure_params=overrides.pop("procedure_params", {}),
        benchmark=overrides.pop("benchmark", "acceptance gate plan"),
    )
    for key, value in overrides.items():
        if hasattr(plan, key):
            setattr(plan, key, value)
    acceptance = overrides.get("acceptance_criteria") or [
        SuccessCriterion(kind="improves", metric="score", threshold=0.10),
        SuccessCriterion(kind="not_regresses", metric="duration_ms", threshold=0.5),
    ]
    proposal = ImprovementProposal(
        weakness_id=weakness.id,
        hypothesis_id=hypothesis.id,
        target=parent.ref,
        proposed_change=f"{candidate.ref}: {candidate.changes}",
        rationale=hypothesis.rationale,
        expected_benefit=statement,
        risks=[
            "candidate may regress metrics the criteria do not cover",
            "candidate may behave differently outside benchmark conditions",
        ],
        experiment_plan=plan,
        acceptance_criteria=[
            SuccessCriterion.model_validate(c.model_dump())
            if hasattr(c, "model_dump")
            else SuccessCriterion.model_validate(c)
            for c in acceptance
        ],
        origin=origin,  # type: ignore[arg-type]
        proposed_by=proposed_by,
        metadata={"weakness_category": weakness.category.value},
    )
    candidate.proposal_id = proposal.id
    candidate.hypothesis_id = hypothesis.id
    return hypothesis, proposal, candidate


def _hypothesis_statement(proposal: ImprovementProposal) -> str:
    return proposal.expected_benefit or proposal.proposed_change


def _describe_config(config: dict[str, Any]) -> str:
    if not config:
        return "no-op configuration"
    return ", ".join(f"{key}={value}" for key, value in sorted(config.items()))


def _name_from_ref(ref: str) -> str:
    return ref.split(":")[0]


def _utcnow_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


__all__ = [
    "ImprovementPipeline",
    "build_proposal_artifacts",
]
