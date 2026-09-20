"""The Skynet core loop.

Executes one controlled agent cycle:

    GOAL → OBSERVE → PLAN → SELECT ACTION → EXECUTE → RECORD EXPERIENCE
         → EVALUATE → UPDATE STATE → COMPLETE / CONTINUE

Design invariants:

- **Bounded.** Every run carries step and wall-clock budgets; exhaustion
  ends the run as a recorded failure, never an infinite tangent.
- **Crash-proof.** An action that raises, a dead perception source or a
  failing storage backend all degrade into recorded failures — the loop
  always produces a :class:`RunSummary`.
- **Fully traced.** Every stage emits typed events through
  :class:`agency.trace.Trace`; every stage records experiences.
- **Pluggable.** Planner, actions, evaluator, perception and storage are
  injected; this module contains no provider-specific or unsafe logic.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from agency.actions.base import (
    ActionContext,
    ActionRecord,
    ActionRegistry,
    ActionResult,
    ActionSpec,
    run_action,
)
from agency.config import SkynetSettings
from agency.evaluator import Evaluation, Evaluator
from agency.experience import ExperienceRecorder
from agency.goals import Goal, GoalManager, GoalSpec
from agency.perception import PerceptionAdapter
from agency.planner import Plan, Planner
from agency.state import AgentState
from agency.storage import RunRecord, Storage
from agency.trace import Trace, TraceEventType, TraceSink

logger = logging.getLogger("skynet.loop")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunSummary(BaseModel):
    """Structured, serializable summary of one run.

    Status semantics (stable contract for future phases):
    - ``status`` reflects **cycle integrity**: ``completed`` means the cycle
      ran to its end (planner, storage, budgets all held); ``failed`` means
      the cycle itself broke (crashed planner/evaluator, exhausted budgets,
      unexpected exception).
    - ``evaluation`` reflects **quality**: failed actions and dead sensors
      do not break the cycle — they fail the evaluation instead.
    """

    run_id: str
    goal_id: str | None = None
    goal_title: str = ""
    status: str = "failed"
    planner: str = ""
    perception: str = ""
    steps_planned: int = 0
    steps_executed: int = 0
    actions: list[dict[str, Any]] = Field(default_factory=list)
    observations: int = 0
    evaluation: Evaluation | None = None
    experiences: int = 0
    events: int = 0
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime = Field(default_factory=_utcnow)
    #: Full serializable final AgentState (proof of the serialization contract).
    final_state: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff the cycle completed AND the evaluation passed."""
        return self.status == "completed" and (
            self.evaluation is None or self.evaluation.passed
        )


class SkynetCore:
    """Executes goals through the controlled agent cycle.

    Assembled once (see :func:`agency.bootstrap.build_core`) and reused for
    every run; all per-run state lives in :class:`AgentState` and storage —
    no mutable globals.
    """

    def __init__(
        self,
        *,
        settings: SkynetSettings,
        storage: Storage,
        goal_manager: GoalManager,
        actions: ActionRegistry,
        planner: Planner,
        evaluator: Evaluator,
        perception: PerceptionAdapter,
        experiences: ExperienceRecorder,
        trace_sinks: Sequence[TraceSink],
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._goals = goal_manager
        self._actions = actions
        self._planner = planner
        self._evaluator = evaluator
        self._perception = perception
        self._experiences = experiences
        self._trace_sinks = list(trace_sinks)

    # ------------------------------------------------------------------ API

    async def run_goal(self, spec: GoalSpec) -> RunSummary:
        """Run the full agent cycle for one goal. Never raises: failures
        are recorded and reported through the summary."""
        run_id = uuid.uuid4().hex
        started_at = _utcnow()
        deadline = time.monotonic() + self._settings.max_run_seconds
        tracer = Trace(run_id, self._trace_sinks)
        state = AgentState(run_id=run_id, goal_title=spec.title, status="running")

        goal: Goal | None = None
        plan: Plan | None = None
        action_summaries: list[dict[str, Any]] = []
        run_evaluation: Evaluation | None = None
        experience_count = 0
        run_result: dict[str, Any] = {}
        error: str | None = None
        status = "failed"

        try:
            # GOAL ------------------------------------------------------------
            goal = await self._goals.create(spec)
            goal = await self._goals.mark_running(goal.id)
            # Persist the run row BEFORE any trace event: run_events carries an
            # FK to runs, so a persisted trace must never precede its run row.
            await self._storage.create_run(
                RunRecord(
                    id=run_id,
                    goal_id=goal.id,
                    status="running",
                    planner=self._planner.name,
                    perception=self._perception.name,
                    started_at=started_at,
                )
            )
            await tracer.emit(TraceEventType.RUN_STARTED, goal_title=spec.title)
            await tracer.emit(TraceEventType.GOAL_CREATED, goal_id=goal.id, title=goal.title)

            # OBSERVE ---------------------------------------------------------
            try:
                observations = await self._perception.observe(state)
            except Exception as exc:
                observations = []
                state.add_error(f"perception failed: {type(exc).__name__}: {exc}")
            for observation in observations:
                observation.run_id = run_id
                observation.goal_id = goal.id
                state.add_observation(observation)
                await tracer.emit(
                    TraceEventType.OBSERVATION_RECEIVED,
                    observation_id=observation.id,
                    source=observation.source,
                    kind=observation.kind,
                )
                self._experiences.record(
                    run_id=run_id,
                    goal_id=goal.id,
                    kind="observation",
                    summary=observation.summary or observation.source,
                    payload=observation.model_dump(mode="json"),
                    outcome="failure" if observation.kind == "error" else "neutral",
                )
                if observation.kind == "error":
                    state.add_error(f"observation error: {observation.summary}")
            experience_count += len(observations)
            await self._safe_flush()

            # PLAN ------------------------------------------------------------
            plan = await self._planner.plan(goal, state)
            await tracer.emit(
                TraceEventType.PLAN_CREATED, steps=plan.step_types, rationale=plan.rationale
            )
            await self._storage.update_run_plan(
                run_id, [step.model_dump(mode="json") for step in plan.steps]
            )

            # SELECT ACTION → EXECUTE → RECORD → EVALUATE ----------------------
            for index, step_spec in enumerate(plan.steps):
                state.current_step = index
                if index >= self._settings.max_steps_per_run:
                    error = (
                        f"max_steps_per_run ({self._settings.max_steps_per_run}) exhausted "
                        f"before step {index}"
                    )
                    break
                if time.monotonic() > deadline:
                    error = f"max_run_seconds ({self._settings.max_run_seconds}) exceeded"
                    break

                await tracer.emit(
                    TraceEventType.ACTION_SELECTED,
                    action_id=step_spec.id,
                    action_type=step_spec.type,
                    step=index,
                )
                result = await self._execute_step(step_spec, state, run_id, goal.id, tracer)

                state.add_action(ActionRecord(spec=step_spec, result=result))
                if result.success:
                    await tracer.emit(
                        TraceEventType.ACTION_COMPLETED,
                        action_id=result.action_id,
                        action_type=result.action_type,
                        duration_ms=result.duration_ms,
                    )
                else:
                    state.add_error(f"action {result.action_type} failed: {result.error}")
                    await tracer.emit(
                        TraceEventType.ACTION_FAILED,
                        action_id=result.action_id,
                        action_type=result.action_type,
                        error=result.error,
                    )
                action_summaries.append(
                    {
                        "type": result.action_type,
                        "success": result.success,
                        "duration_ms": result.duration_ms,
                        "error": result.error,
                    }
                )
                self._experiences.record(
                    run_id=run_id,
                    goal_id=goal.id,
                    kind="action",
                    summary=f"{result.action_type} -> {'ok' if result.success else 'failed'}",
                    payload=result.model_dump(mode="json"),
                    outcome="success" if result.success else "failure",
                )
                experience_count += 1
                await self._safe_flush()

                evaluation = await self._evaluator.evaluate_action(step_spec, result, state)
                await tracer.emit(
                    TraceEventType.EVALUATION_COMPLETED,
                    subject=evaluation.subject,
                    passed=evaluation.passed,
                    score=evaluation.score,
                )
                self._experiences.record(
                    run_id=run_id,
                    goal_id=goal.id,
                    kind="evaluation",
                    summary=evaluation.notes,
                    payload=evaluation.model_dump(mode="json"),
                    outcome="success" if evaluation.passed else "failure",
                )
                experience_count += 1
                await self._safe_flush()

            # EVALUATE + UPDATE STATE + COMPLETE -------------------------------
            if error is None:
                status = "completed"
                run_result = self._synthesize(state)
                run_result["actions"] = action_summaries
                # Evaluate BEFORE stamping terminal status so the evaluator
                # judges the work, not the verdict.
                run_evaluation = await self._evaluator.evaluate_run(state)
                state.status = "completed"
                await tracer.emit(
                    TraceEventType.EVALUATION_COMPLETED,
                    subject=run_evaluation.subject,
                    passed=run_evaluation.passed,
                    score=run_evaluation.score,
                )
                self._experiences.record(
                    run_id=run_id,
                    goal_id=goal.id,
                    kind="evaluation",
                    summary=run_evaluation.notes,
                    payload=run_evaluation.model_dump(mode="json"),
                    outcome="success" if run_evaluation.passed else "failure",
                )
                self._experiences.record(
                    run_id=run_id,
                    goal_id=goal.id,
                    kind="summary",
                    summary=str(run_result.get("summary", "run completed")),
                    payload=run_result,
                    outcome="success" if run_evaluation.passed else "failure",
                )
                experience_count += 2
                await self._safe_flush()
            else:
                run_result = {"actions": action_summaries, "error": error}
                self._experiences.record(
                    run_id=run_id,
                    goal_id=goal.id,
                    kind="error",
                    summary=error,
                    outcome="failure",
                )
                experience_count += 1
                await self._safe_flush()

        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            state.add_error(error)
            run_result = {"actions": action_summaries, "error": error}
            logger.exception("Skynet run %s failed unexpectedly", run_id)
            self._experiences.record(
                run_id=run_id,
                goal_id=goal.id if goal else None,
                kind="error",
                summary=error,
                outcome="failure",
            )
            experience_count += 1
            await self._safe_flush()

        finished_at = _utcnow()
        state.status = status  # type: ignore[assignment]
        if status != "completed":
            state.touch()

        # UPDATE STATE (persistence) — a storage outage must not mask the summary.
        try:
            if goal is not None:
                # Goal status reflects OUTCOME quality: a cycle that held
                # together but failed its evaluation did not achieve the goal.
                if status == "completed" and (
                    run_evaluation is None or run_evaluation.passed
                ):
                    goal = await self._goals.mark_completed(goal.id)
                else:
                    goal = await self._goals.mark_failed(goal.id)
            await self._storage.finish_run(
                run_id,
                status=status,
                result=run_result,
                error=error,
                steps_executed=len(action_summaries),
            )
        except Exception:
            logger.exception("Skynet run %s: failed to persist final state", run_id)

        if status == "completed":
            await tracer.emit(TraceEventType.RUN_COMPLETED, result=run_result)
        else:
            await tracer.emit(TraceEventType.RUN_FAILED, error=error)

        return RunSummary(
            run_id=run_id,
            goal_id=goal.id if goal else None,
            goal_title=spec.title,
            status=status,
            planner=self._planner.name,
            perception=self._perception.name,
            steps_planned=len(plan.steps) if plan else 0,
            steps_executed=len(action_summaries),
            actions=action_summaries,
            observations=len(state.observations),
            evaluation=run_evaluation,
            experiences=experience_count,
            events=tracer.event_count,
            result=run_result,
            error=error,
            started_at=started_at,
            finished_at=finished_at,
            final_state=state.model_dump(mode="json"),
        )

    # ------------------------------------------------------------- internals

    async def _safe_flush(self) -> None:
        """Flush buffered experiences; a broken sink degrades to a log line
        so the run can still complete and summarize."""
        try:
            await self._experiences.flush()
        except Exception:
            logger.exception("Skynet run: experience flush failed")

    async def _execute_step(
        self,
        spec: ActionSpec,
        state: AgentState,
        run_id: str,
        goal_id: str,
        tracer: Trace,
    ) -> ActionResult:
        """Resolve and run one planned step; policy failures (unknown or
        non-allow-listed actions) become failed results, not crashes."""
        action = self._actions.get(spec.type)
        if action is None:
            return ActionResult(
                action_id=spec.id,
                action_type=spec.type,
                success=False,
                error=f"unknown action type {spec.type!r} "
                f"(registered: {', '.join(self._actions.names()) or 'none'})",
            )
        if spec.type not in self._settings.action_allowlist:
            return ActionResult(
                action_id=spec.id,
                action_type=spec.type,
                success=False,
                error=f"action {spec.type!r} is not permitted by configuration "
                f"(SKYNET_ENABLED_ACTIONS)",
            )
        ctx = ActionContext(
            run_id=run_id,
            goal_id=goal_id,
            state_snapshot=state.model_dump(mode="json"),
        )
        await tracer.emit(TraceEventType.ACTION_STARTED, action_id=spec.id, action_type=spec.type)
        return await run_action(action, spec, ctx)

    @staticmethod
    def _synthesize(state: AgentState) -> dict[str, Any]:
        """Deterministic result synthesis for the foundation phase.

        Extracts the strongest God's Eye event from observations, if any.
        Deliberately trivial: replaced by the reasoning module in the
        intelligence phase; the loop only depends on the dict contract.
        """
        events: list[dict[str, Any]] = []
        for observation in state.observations:
            content = observation.content
            if isinstance(content, dict) and isinstance(content.get("events"), list):
                events.extend(item for item in content["events"] if isinstance(item, dict))

        strongest: dict[str, Any] | None = None
        with_magnitude = [
            item for item in events if isinstance(item.get("magnitude"), (int, float))
        ]
        if with_magnitude:
            strongest = max(with_magnitude, key=lambda item: item["magnitude"])

        summary = (
            f"Run completed: {len(state.observations)} observation(s), "
            f"{len(state.actions)} action(s), {len(events)} event(s) seen"
        )
        result: dict[str, Any] = {
            "summary": summary,
            "observations": len(state.observations),
            "events_seen": len(events),
        }
        if strongest is not None:
            result["strongest_event"] = strongest
            summary += (
                f"; strongest magnitude {strongest['magnitude']} at {strongest.get('place')}"
            )
            result["summary"] = summary
        return result
