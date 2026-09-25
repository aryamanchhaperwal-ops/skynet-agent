"""Autonomous orchestrator — coordinates the existing subsystems (Phase P8).

The orchestrator is a **stage driver**, not a second agent: every stage
delegates to a finished phase and records progress in the persisted run.

    INITIALIZE → OBSERVE → PLAN → RESEARCH → COMMUNICATE → ACT
        → EVALUATE → (LEARN → IMPROVE) → PLAN (next iteration) / COMPLETE

Design invariants:

- **Reuse, never duplicate.** The full agent cycle still runs inside
  :meth:`SkynetCore.run_goal` (ACT stage); research goes through the
  P4-backed adapter; comms through the P5-backed adapter; memory through
  the P2 manager; experiments through the P6 runner; improvement through
  the P7 pipeline.
- **Persisted state.** The run record on disk is the single source of
  truth; every stage write is a save, so pause/resume/cancel from another
  process observe a consistent world.
- **Budgets before work.** Every stage checks remaining budget first and
  PAUSEs (never silently continues) when a limit is hit.
- **Untrusted content stays data.** Evidence from web/AI adapters is
  stored verbatim but can only influence the loop as information; the
  security module guarantees directive-like content never becomes control
  flow.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from agency.orchestrator.adapters import (
    CommsAdapter,
    ResearchAdapter,
)
from agency.orchestrator.models import (
    ApprovalRequest,
    AutonomousRun,
    BudgetDecision,
    CoreRunSummary,
    Evidence,
    IterationPlan,
    RunStage,
    RunStatus,
    StageDecision,
)
from agency.orchestrator.state_machine import InvalidTransitionError, require_transition
from agency.orchestrator.store import RunStore

logger = logging.getLogger("skynet.orchestrator")


def _extract_search_query(answer: str) -> str:
    """Pull a usable web-search query out of a model's free-text proposal.

    Models routinely wrap the query in prose ("Web search query: \"…\"").
    A verbatim prose sentence is a poor search query — search engines
    return few/no hits for it — so extract a short quoted/backticked
    fragment or a labeled value when present, else fall back to the first
    sentence. Returns "" when nothing usable remains.
    """
    text = (answer or "").strip()
    if not text:
        return ""
    # Models emit smart quotes; normalize so quoted fragments match.
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    for pattern in (
        r'"([^"]{3,120}?)"',      # "..." fragment
        r"`([^`\n]{3,120}?)`",     # `...` fragment
        r"query\s*[:=]\s*(.+$)",   # "query: ..." label, case-insensitive
    ):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).strip().strip("\"'`").strip().strip(".,;:")
            if 3 <= len(candidate) <= 120:
                return candidate
    first_sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip()
    # Strip labels the model may have prefixed ("Proposed query: ...").
    first_sentence = re.sub(
        r"^(?:proposed\s+|search\s+|web\s+)*query\s*[:=]\s*",
        "",
        first_sentence,
        flags=re.IGNORECASE,
    ).strip("\"'` ")
    return first_sentence[:120]


class OrchestratorError(RuntimeError):
    """Raised for orchestrator contract violations (never external content)."""


class AutonomousOrchestrator:
    """Coordinates the finished Skynet phases through one resumable loop."""

    def __init__(
        self,
        *,
        core: Any,
        store: RunStore,
        research: ResearchAdapter,
        comms: CommsAdapter,
        memory: Any = None,
        detector: Any = None,
        pipeline: Any = None,
        emit: Any = None,
    ) -> None:
        self._core = core
        self._store = store
        self._research = research
        self._comms = comms
        self._memory = memory
        self._detector = detector
        self._pipeline = pipeline
        self._emit = emit

    # -- infrastructure ------------------------------------------------------------

    async def _trace(self, event: str, **payload: Any) -> None:
        if self._emit is None:
            return
        try:
            await self._emit(event, **payload)
        except Exception:
            logger.exception("orchestrator trace emission failed for %s", event)

    async def _save(self, run: AutonomousRun) -> None:
        run.touch()
        await self._store.save(run)

    def _collect_security_report(self, run: AutonomousRun) -> None:
        """Record directive-like fragments found in external content.

        Findings are DATA on the run record (and one trace event per batch):
        they never change control flow, configuration or policy. The
        orchestrator treats them as telemetry about untrusted input.
        """
        found = 0
        for item in run.evidence:
            flags = item.provenance.get("security_flags") or []
            analysis = item.provenance.get("analysis_directives") or []
            hits = list(flags) + list(analysis)
            if hits:
                found += len(hits)
                run.security_report.setdefault("findings", []).append(
                    {
                        "evidence_id": item.id,
                        "source_type": item.source_type,
                        "source": item.source,
                        "matches": hits,
                        "recorded_at": item.provenance.get("retrieved_at", ""),
                    }
                )
        if found:
            run.security_report["total_flags"] = int(
                run.security_report.get("total_flags", 0)
            ) + found
            run.security_report["status"] = "CONTENT_TREATED_AS_UNTRUSTED_DATA"

    async def _trace_llm_request(
        self,
        run: AutonomousRun,
        coro: Any,
        *,
        purpose: str,
    ) -> Any:
        """Await one LLM call, emitting request lifecycle events.

        The LLM is invoked through the existing planner/evaluator seam —
        this wrapper only adds observability around it (payloads carry
        provider/model/latency, never message bodies or credentials).
        """
        service = getattr(self._core, "_intelligence", None)
        info = service.info() if service is not None else {}
        await self._trace(
            "LLM_REQUEST_STARTED",
            run_id=run.id,
            purpose=purpose,
            provider=info.get("provider", ""),
            model=info.get("model", ""),
        )
        started = time.perf_counter()
        try:
            result = await coro
        except Exception as exc:
            await self._trace(
                "LLM_REQUEST_FAILED",
                run_id=run.id,
                purpose=purpose,
                error=f"{type(exc).__name__}: {exc}"[:200],
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            raise
        usage = run.usage
        usage["ai_requests"] = usage.get("ai_requests", 0.0) + 1
        await self._trace(
            "LLM_REQUEST_COMPLETED",
            run_id=run.id,
            purpose=purpose,
            provider=info.get("provider", ""),
            model=info.get("model", ""),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return result

    def _budget_decision(self, run: AutonomousRun) -> BudgetDecision:
        usage = run.usage
        started = run.created_at.timestamp()
        runtime = time.time() - started
        return run.budgets.check(
            iteration=run.iteration,
            runtime_seconds=runtime,
            web_requests=int(usage.get("web_requests", 0.0)),
            ai_requests=int(usage.get("ai_requests", 0.0)),
            experiments=int(usage.get("experiments", 0.0)),
            cost_usd=usage.get("cost_usd", 0.0),
        )

    # -- public API ---------------------------------------------------------------------

    async def start(
        self,
        objective: str,
        *,
        budgets: Any = None,
        parent_run_id: str | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> AutonomousRun:
        """Create and persist a new autonomous run in INITIALIZE."""
        from agency.orchestrator.models import OrchestratorBudgets

        run = AutonomousRun(
            objective=objective.strip(),
            budgets=budgets or OrchestratorBudgets(),
            parent_run_id=parent_run_id,
            configuration=dict(configuration or {}),
            provenance={"orchestrator": "agency.orchestrator", "phase": "P8"},
        )
        await self._save(run)
        await self._trace(
            "AUTONOMOUS_RUN_STARTED",
            run_id=run.id,
            objective=run.objective,
            budgets=run.budgets.model_dump(),
        )
        return run

    async def execute(self, run_id: str) -> AutonomousRun:
        """Drive the run from its current stage until a stop condition.

        Stop conditions: terminal status, budget pause/stop, a pending
        human approval (WAITING), or completion. Never raises for
        recoverable problems — everything is recorded on the run.
        """
        run = await self._store.get(run_id)
        if run is None:
            raise OrchestratorError(f"unknown run id {run_id!r}")
        if run.is_terminal:
            return run

        if run.status is RunStatus.PAUSED:
            run.status = RunStatus.RUNNING
            await self._trace("AUTONOMOUS_RUN_RESUMED", run_id=run.id)
        if run.status is RunStatus.WAITING:
            # A waiting run needs its approval resolved first; not an error.
            return run

        max_stage_visits = 64  # structural anti-livelock bound
        visits = 0
        while not run.is_terminal and visits < max_stage_visits:
            visits += 1
            fresh = await self._store.get(run.id)
            if fresh is not None:
                run = fresh  # picks up cross-process pause/cancel immediately

            if run.status is RunStatus.PAUSED:
                await self._trace("AUTONOMOUS_RUN_PAUSED", run_id=run.id)
                return run
            if run.status is RunStatus.CANCELLED:
                await self._trace("AUTONOMOUS_RUN_CANCELLED", run_id=run.id)
                return run
            if run.status is RunStatus.WAITING:
                await self._trace(
                    "AUTONOMOUS_APPROVAL_REQUIRED",
                    run_id=run.id,
                    what=run.pending_approval.what if run.pending_approval else "",
                )
                return run

            budget = self._budget_decision(run)
            # Once the evaluator has decided completion, the remaining
            # stages (LEARN → COMPLETE) are bookkeeping: they consume no
            # budget and a run that met its objective must not be failed
            # merely because the iteration counter hit the cap on the way
            # through them.
            already_complete = run.plan.sufficient_evidence
            if budget is BudgetDecision.PAUSE and not already_complete:
                run.status = RunStatus.PAUSED
                run.add_error("budget reached; run paused")
                await self._save(run)
                await self._trace(
                    "AUTONOMOUS_BUDGET_PAUSED",
                    run_id=run.id,
                    usage=run.usage,
                    budgets=run.budgets.model_dump(),
                )
                return run
            if budget is BudgetDecision.STOP and not already_complete:
                await self._finish(run, completed=False, reason="iteration budget exhausted")
                return run

            stage = run.current_stage
            await self._trace("AUTONOMOUS_STAGE_STARTED", run_id=run.id, stage=stage.value)
            try:
                handler = getattr(self, f"_stage_{stage.value}", None)
                if handler is None:  # pragma: no cover - all stages have handlers
                    raise OrchestratorError(f"no handler for stage {stage.value!r}")
                next_stage = await handler(run)
            except InvalidTransitionError as exc:
                run.add_error(f"state machine violation: {exc}")
                await self._finish(run, completed=False, reason=str(exc))
                return run
            except Exception as exc:
                # Recoverable failure: record, count it, replan or stop.
                run.add_error(f"stage {stage.value} failed: {type(exc).__name__}: {exc}")
                if run.current_state == "recovering" and len(run.errors) > 4:
                    await self._finish(
                        run, completed=False, reason="repeated unrecoverable stage failures"
                    )
                    return run
                run.current_state = "recovering"
                await self._save(run)
                next_stage = RunStage.PLAN
                continue

            if next_stage is None:
                return run  # handler paused/waited/finished
            try:
                require_transition(stage, next_stage)
            except InvalidTransitionError:
                run.add_error(
                    f"handler requested invalid transition {stage.value} -> {next_stage.value}"
                )
                await self._finish(run, completed=False, reason="state machine violation")
                return run
            run.current_stage = next_stage
            await self._save(run)
            await self._trace(
                "AUTONOMOUS_STAGE_COMPLETED", run_id=run.id, stage=stage.value,
                next=next_stage.value,
            )
        return run

    async def pause(self, run_id: str) -> AutonomousRun | None:
        return await self._store.pause(run_id)

    async def resume(self, run_id: str) -> AutonomousRun | None:
        resumed = await self._store.resume(run_id)
        if resumed is not None:
            await self._trace("AUTONOMOUS_RUN_RESUMED", run_id=run_id)
        return resumed

    async def cancel(self, run_id: str) -> AutonomousRun | None:
        cancelled = await self._store.cancel(run_id)
        if cancelled is not None:
            await self._trace("AUTONOMOUS_RUN_CANCELLED", run_id=run_id)
        return cancelled

    async def decide_approval(
        self, run_id: str, *, approve: bool, decided_by: str
    ) -> AutonomousRun | None:
        """Resolve a pending human checkpoint (spec §17)."""
        run = await self._store.get(run_id)
        if run is None or run.pending_approval is None:
            return run
        approval = run.pending_approval
        approval.status = "approved" if approve else "rejected"
        approval.decided_at = approval.requested_at.__class__.now(approval.requested_at.tzinfo)
        approval.decided_by = decided_by
        run.decisions.append(
            StageDecision(
                action="improvement_opportunity",
                reason=f"approval {'approved' if approve else 'rejected'} by {decided_by}",
            )
        )
        if approve and self._pipeline is not None:
            # An approved checkpoint releases the run back to the loop; the
            # actual application happens through the P7 pipeline's own
            # approve/apply path (already human-gated there).
            run.status = RunStatus.RUNNING
            run.current_state = "active"
        else:
            run.status = RunStatus.RUNNING
            run.current_state = "active"
        run.pending_approval = None
        await self._save(run)
        return run

    # -- stages ------------------------------------------------------------------------

    async def _stage_initialize(self, run: AutonomousRun) -> Any:
        run.current_state = "active"
        # Objective → plan scaffold; the planner refines it every iteration.
        run.plan = IterationPlan(
            iteration=run.iteration,
            focus=run.objective,
            research_queries=[run.objective],
        )
        return RunStage.OBSERVE

    async def _stage_observe(self, run: AutonomousRun) -> Any:
        # Observe = recall relevant memory before spending budget on the
        # external world (spec §10: retrieve before repeating expensive work).
        if self._memory is not None:
            try:
                recalled = await self._memory.recall(run.objective, max_results=5)
                for record in recalled:
                    run.evidence.append(
                        Evidence(
                            source=f"memory:{record.id[:12]}",
                            source_type="memory",
                            content=(record.summary or record.content)[:500],
                            confidence=record.confidence or 0.5,
                            provenance={
                                "memory_id": record.id,
                                "type": record.type.value,
                                "evidence_class": "MEMORY",
                            },
                        )
                    )
                    await self._trace(
                        "OBSERVATION_RECORDED",
                        run_id=run.id,
                        source=f"memory:{record.id[:12]}",
                        source_type="memory",
                        evidence_class="MEMORY",
                    )
                if recalled:
                    await self._trace(
                        "AUTONOMOUS_MEMORY_UPDATED",
                        run_id=run.id,
                        recalled=len(recalled),
                    )
            except Exception as exc:
                run.add_error(f"memory recall failed (ignored): {exc}")
        return RunStage.PLAN

    async def _stage_plan(self, run: AutonomousRun) -> Any:
        plan = run.plan
        plan.iteration = run.iteration
        plan.focus = plan.focus or run.objective
        prior_queries = list(plan.research_queries)
        # Deterministic refinement from accumulated evidence: queries target
        # what is not yet covered (spec §6: revise the plan on new evidence).
        if run.iteration > 0:
            base = plan.focus
            plan.research_queries = [f"{base} (iteration {run.iteration + 1} focus)"]
        # Ask external AI only when configured to (budget + config seam).
        configured_questions = run.configuration.get("ai_questions")
        if isinstance(configured_questions, list):
            plan.ai_questions = [str(q) for q in configured_questions if str(q).strip()]
        # Real-LLM plan proposal: when the core runs an LLM planner (and no
        # explicit steps/queries are configured — those always win), let the
        # model propose the query here, with the deterministic text as
        # fallback. Re-fires on replan iterations so a new iteration can
        # target what previous evidence left uncovered. One observable
        # request per plan stage, one failure-recovery path, no retries.
        intelligence = getattr(self._core, "_intelligence", None)
        configured_queries = run.configuration.get("research_queries")
        if isinstance(configured_queries, list) and configured_queries:
            plan.research_queries = [str(q) for q in configured_queries if str(q).strip()]
        elif (
            intelligence is not None
            and not run.configuration.get("core_goal_params")
            and not configured_queries
        ):
            prompt = (
                "Propose one concise web search query for this objective, then "
                "justify it in one sentence.\n"
                f"Objective: {run.objective}\n"
                f"Focus: {plan.focus}"
            )
            if prior_queries:
                prompt += (
                    "\nPrior queries already searched: "
                    + "; ".join(prior_queries[:5])
                    + "\nPropose a query covering a DIFFERENT aspect of the objective."
                )
            answer = await self._trace_llm_request(
                run,
                intelligence.generate(
                    "You are a research planner. Be terse and concrete.",
                    prompt,
                    max_tokens=260,
                    metadata={"purpose": "autonomous_plan"},
                ),
                purpose="plan",
            )
            if answer:
                query = _extract_search_query(answer)
                if query:
                    run.plan.research_queries = [query]
                else:
                    run.add_error(
                        "llm plan proposal unusable; using deterministic queries"
                    )
            else:
                run.add_error("llm plan proposal failed; using deterministic queries")
        await self._trace(
            "AUTONOMOUS_PLAN_CREATED",
            run_id=run.id,
            iteration=run.iteration,
            queries=len(plan.research_queries),
            questions=len(plan.ai_questions),
        )
        if plan.sufficient_evidence:
            return RunStage.COMPLETE
        if plan.ai_questions and run.budgets.max_ai_requests > 0:
            return RunStage.RESEARCH
        return RunStage.RESEARCH

    async def _stage_research(self, run: AutonomousRun) -> Any:
        remaining_web = max(
            0, run.budgets.max_web_requests - int(run.usage.get("web_requests", 0.0))
        )
        queries = run.plan.research_queries[:remaining_web]
        gathered = 0
        for query in queries:
            await self._trace(
                "RESEARCH_STARTED",
                run_id=run.id,
                query=query,
                adapter=type(self._research).__name__,
            )
            try:
                found = await self._research.search(
                    query, max_results=3, run_id=run.id
                )
            except Exception as exc:
                run.add_error(f"research failed for {query!r}: {exc}")
                continue
            run.usage["web_requests"] = run.usage.get("web_requests", 0.0) + 1
            # Some backends (e.g. Wikipedia's keyword search) return nothing
            # for keyword-poor, punctuation-heavy queries. One budget-counted
            # simplified retry — first search terms only, punctuation
            # stripped — keeps the loop moving without fabricating results.
            if not found and remaining_web - int(
                run.usage.get("web_requests", 0.0)
            ) > 0:
                simplified = re.sub(r"[^\w\s-]", " ", query)
                simplified = " ".join(simplified.split()[:5])
                if simplified and simplified.lower() != query.lower():
                    try:
                        found = await self._research.search(
                            simplified, max_results=3, run_id=run.id
                        )
                    except Exception as exc:
                        run.add_error(
                            f"research retry failed for {simplified!r}: {exc}"
                        )
                        found = []
                    run.usage["web_requests"] = run.usage.get("web_requests", 0.0) + 1
            for item in found:
                run.evidence.append(item)
                gathered += 1
                await self._trace(
                    "SOURCE_RETRIEVED",
                    run_id=run.id,
                    source=item.source,
                    source_type=item.source_type,
                    evidence_class=item.provenance.get("evidence_class", ""),
                    retrieved_at=item.provenance.get("retrieved_at", ""),
                    security_flags=item.provenance.get("security_flags", []),
                )
        await self._trace(
            "AUTONOMOUS_RESEARCH_COMPLETED",
            run_id=run.id,
            queries=len(queries),
            evidence=gathered,
        )
        self._collect_security_report(run)
        if run.plan.ai_questions and run.budgets.max_ai_requests > 0:
            return RunStage.COMMUNICATE
        return RunStage.ACT

    async def _stage_communicate(self, run: AutonomousRun) -> Any:
        remaining = run.budgets.max_ai_requests - int(run.usage.get("ai_requests", 0.0))
        asked = 0
        for question in run.plan.ai_questions[:remaining]:
            try:
                answer = await self._comms.ask(question, run_id=run.id, timeout_seconds=30.0)
            except Exception as exc:
                run.add_error(f"comms failed for {question!r}: {exc}")
                continue
            run.usage["ai_requests"] = run.usage.get("ai_requests", 0.0) + 1
            if answer is not None:
                run.evidence.append(answer)
                asked += 1
        self._collect_security_report(run)
        await self._trace(
            "AUTONOMOUS_COMMUNICATION_COMPLETED",
            run_id=run.id,
            questions=len(run.plan.ai_questions[:remaining]),
            answered=asked,
        )
        return RunStage.ACT

    async def _stage_act(self, run: AutonomousRun) -> Any:
        """Execute the full agent cycle through the existing SkynetCore.

        This is where reuse is absolute: the core loop does its own
        observe→plan→execute→evaluate cycle against the real action
        registry, with all its budgets, tracing and experience recording.
        """
        from agency.goals import GoalSpec

        # Core goal params: the operator/config seam for steering the core
        # planner (e.g. explicit steps). The orchestrator never invents
        # action choices itself — planning stays the core's responsibility.
        params: dict[str, Any] = {
            "autonomous_run_id": run.id,
            "iteration": run.iteration,
        }
        configured = run.configuration.get("core_goal_params")
        if isinstance(configured, dict):
            params.update(configured)
        spec = GoalSpec(
            title=f"[autonomous:{run.id[:8]}:{run.iteration}] {run.plan.focus}",
            description=run.objective,
            origin="self",
            params=params,
        )
        summary = await self._core.run_goal(spec)
        run.core_run_ids.append(summary.run_id)
        run.core_summaries.append(
            CoreRunSummary(
                run_id=summary.run_id,
                status=str(summary.status),
                ok=bool(summary.ok),
            )
        )
        run.usage["runtime_seconds"] += 0.0  # wall time is tracked globally
        await self._trace(
            "AUTONOMOUS_ACTION_COMPLETED",
            run_id=run.id,
            core_run_id=summary.run_id,
            status=summary.status,
            ok=summary.ok,
        )
        if not summary.ok:
            run.add_error(
                f"core run {summary.run_id[:12]} did not pass: {summary.status}"
            )
        self._collect_security_report(run)
        # Evidence from the core run's synthesized result.
        result_payload = summary.result or {}
        if result_payload.get("summary"):
            run.evidence.append(
                Evidence(
                    source=f"core_run:{summary.run_id[:12]}",
                    source_type="action",
                    content=str(result_payload["summary"])[:500],
                    confidence=0.7 if summary.ok else 0.3,
                    provenance={
                        "core_run_id": summary.run_id,
                        "actions": len(summary.actions),
                        "evidence_class": "EXPERIMENTAL_RESULT" if any(
                            "experiment" in str(getattr(a, "type", "")) for a in summary.actions
                        ) else "MODEL_GENERATED_CONTENT",
                    },
                )
            )
            await self._trace(
                "OBSERVATION_RECORDED",
                run_id=run.id,
                source=f"core_run:{summary.run_id[:12]}",
                source_type="action",
                evidence_class=run.evidence[-1].provenance["evidence_class"],
            )
        return RunStage.EVALUATE

    async def _stage_evaluate(self, run: AutonomousRun) -> Any:
        """Mechanical, evidence-based evaluation (spec §11).

        No LLM decides completion here: the decision is a deterministic
        function of gathered evidence vs the plan's sufficiency marker,
        with an LLM later able to *suggest* via `next_focus` only.
        """
        external = [e for e in run.evidence if e.source_type in {"web", "ai"}]
        enough = len(external) >= 3
        # Completion is mechanical: sufficient external evidence AND the
        # agent cycle actually ran to a passing result. A model's answer
        # alone never completes the objective (spec §11).
        core_ok = bool(run.core_summaries) and any(s.ok for s in run.core_summaries)
        if enough and core_ok:
            decision = StageDecision(
                action="complete",
                reason=(
                    f"{len(external)} external evidence item(s) and a passing "
                    "agent cycle; objective supported"
                ),
                next_focus=run.plan.focus,
            )
        elif enough:
            decision = StageDecision(
                action="research_more",
                reason=(
                    f"agent cycle did not pass; {len(external)} external "
                    "evidence item(s) gathered"
                ),
                next_focus=run.plan.focus,
            )
        else:
            decision = StageDecision(
                action="replan",
                reason="no external evidence yet; replanning",
                next_focus=run.objective,
            )
        run.decisions.append(decision)
        await self._trace(
            "AUTONOMOUS_EVALUATION_COMPLETED",
            run_id=run.id,
            action=decision.action,
            reason=decision.reason,
        )
        mapping = {
            "continue": RunStage.PLAN,
            "replan": RunStage.PLAN,
            "research_more": RunStage.RESEARCH,
            "ask_external_ai": RunStage.COMMUNICATE,
            "complete": RunStage.LEARN,  # learn (memory) before finishing
            "failed": RunStage.LEARN,
            "improvement_opportunity": RunStage.IMPROVE,
            "run_experiment": RunStage.PLAN,
        }
        run.iteration += 1
        if decision.action == "complete":
            run.plan.sufficient_evidence = True
            return RunStage.LEARN
        if decision.action == "failed":
            await self._finish(run, completed=False, reason=decision.reason)
            return None
        return mapping[decision.action]

    async def _stage_learn(self, run: AutonomousRun) -> Any:
        """Persist a run summary memory (spec §10)."""
        if self._memory is not None:
            try:
                record = await self._memory.store(
                    content=(
                        f"Autonomous run {run.id[:12]} objective: {run.objective}. "
                        f"Iteration {run.iteration}, {len(run.evidence)} evidence items, "
                        f"{len(run.decisions)} decisions. Last: "
                        f"{run.decisions[-1].reason if run.decisions else 'n/a'}"
                    )[:2000],
                    summary=f"autonomous run: {run.objective[:80]}",
                    memory_type="episodic",
                    source=f"autonomous_run:{run.id[:12]}",
                    origin="system",
                    importance=0.55,
                    tags=["autonomous", "run"],
                    run_id=run.core_run_ids[-1] if run.core_run_ids else None,
                    force=True,
                )
                if record is not None:
                    await self._trace(
                        "AUTONOMOUS_MEMORY_UPDATED", run_id=run.id, memory_id=record.id
                    )
            except Exception as exc:
                run.add_error(f"learn/memory store failed (ignored): {exc}")
        # After learning, finish cleanly if the objective was met; the
        # COMPLETE stage itself has no handler — execute() treats reaching
        # it as the signal to finalize via _finish.
        if run.plan.sufficient_evidence:
            await self._finish(
                run,
                completed=True,
                reason="objective supported by evidence and a passing agent cycle",
            )
            return None
        return RunStage.PLAN

    async def _stage_improve(self, run: AutonomousRun) -> Any:
        """Self-improvement checkpoint (spec §12).

        Detection runs on recorded data through the P7 pipeline. Proposals
        are created and *parked*: applying one requires the human approval
        gate the pipeline already enforces. The run enters WAITING when a
        checkpoint is raised, never silently continues into application.
        """
        detected: list[Any] = []
        if self._detector is not None and self._pipeline is not None:
            try:
                from agency.improve.detector import DetectorInput

                experiments = await _load_experiments(self)
                detected = await self._pipeline.detect(
                    DetectorInput(experiments=experiments)
                )
                # Pipeline detection persists its findings; keep the run's
                # view of the world in sync for the checkpoint below.
                if detected:
                    weakness = detected[0]
            except Exception as exc:
                run.add_error(f"weakness detection failed (ignored): {exc}")
        await self._trace(
            "AUTONOMOUS_IMPROVEMENT_CHECK",
            run_id=run.id,
            weaknesses=len(detected),
        )
        if detected and self._pipeline is not None:
            weakness = detected[0]
            try:
                _h, proposal, _c = await self._pipeline.propose(weakness.id)
                run.proposal_ids.append(proposal.id)
                run.pending_approval = ApprovalRequest(
                    run_id=run.id,
                    what=f"apply improvement {proposal.id[:12]} for weakness "
                    f"{weakness.category.value}",
                    why=weakness.description,
                    proposed_action=proposal.proposed_change,
                    evidence=[e.summary for e in weakness.evidence[:3]],
                )
                run.status = RunStatus.WAITING
                await self._save(run)
                await self._trace(
                    "AUTONOMOUS_APPROVAL_REQUIRED",
                    run_id=run.id,
                    proposal_id=proposal.id,
                    what=run.pending_approval.what,
                )
                return None  # execute() sees WAITING and returns
            except Exception as exc:
                run.add_error(f"improvement proposal failed (ignored): {exc}")
        return RunStage.PLAN

    # -- completion ----------------------------------------------------------------------

    async def _finish(
        self, run: AutonomousRun, *, completed: bool, reason: str
    ) -> None:
        run.status = RunStatus.COMPLETED if completed else RunStatus.FAILED
        run.current_stage = RunStage.COMPLETE
        run.result = {
            "completed": completed,
            "reason": reason,
            "iterations": run.iteration,
            "evidence_count": len(run.evidence),
            "decisions": [d.action for d in run.decisions],
            "core_runs": len(run.core_run_ids),
        }
        await self._save(run)
        await self._trace(
            "AUTONOMOUS_RUN_COMPLETED" if completed else "AUTONOMOUS_RUN_FAILED",
            run_id=run.id,
            reason=reason,
        )


async def _load_experiments(self: Any) -> list[Any]:
    """Load finished experiments for the detector.

    Reads the **pipeline's own** experiment registry when available (the
    integration seam), falling back to the configured default path so a
    standalone detector still sees recorded history.
    """
    from agency.config import SkynetSettings
    from agency.experiments.experiments_store import ExperimentRegistry

    pipeline_registry = getattr(self._pipeline, "_experiments", None) if self else None
    registry = pipeline_registry or ExperimentRegistry(
        SkynetSettings().experiments_registry_path
    )
    summaries = await registry.list(limit=20)
    records = []
    for summary in summaries:
        record = await registry.get(summary.id)
        if record is not None:
            records.append(record)
    return records


__all__ = ["AutonomousOrchestrator", "OrchestratorError"]
