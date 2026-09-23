"""Experiment runner — executes controlled comparisons end to end.

Lifecycle::

    CREATED → RUNNING → COMPLETED | FAILED | INCONCLUSIVE | ERROR | CANCELLED

The runner is the **only** executor of experiments. It:

1. resolves both arms from the :class:`StrategyRegistry` (unregistered
   targets are refused — no invented behavior),
2. freezes the acceptance criteria into the experiment record *before*
   the first trial (frozen-before-run, blueprint §5),
3. runs each arm for ``experiment.trials`` trials inside a
   :class:`~agency.experiments.sandbox.Sandbox`, with a per-trial timeout
   and a total wall-clock budget,
4. aggregates metrics, builds the :class:`ComparisonReport`,
5. runs the deterministic evaluation engine, and
6. emits every lifecycle event through the run's ``Trace`` facade —
   experiments never open a second logging path.

Cancellation and budget exhaustion land in ``CANCELLED``/``ERROR``, never
in a fabricated SUCCESS.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from agency.experiments.evaluation import evaluate_experiment
from agency.experiments.metrics import aggregate, compare
from agency.experiments.models import (
    Experiment,
    ExperimentStatus,
    TrialResult,
    Verdict,
)
from agency.experiments.registry import StrategyRegistry
from agency.experiments.sandbox import Sandbox

logger = logging.getLogger("skynet.experiments.runner")

#: Event names emitted through the run's tracer (same stream as the core).
EXPERIMENT_EVENTS = (
    "EXPERIMENT_CREATED",
    "EXPERIMENT_STARTED",
    "TRIAL_STARTED",
    "TRIAL_COMPLETED",
    "TRIAL_FAILED",
    "METRIC_RECORDED",
    "BASELINE_MEASURED",
    "CANDIDATE_MEASURED",
    "EXPERIMENT_COMPARED",
    "EXPERIMENT_EVALUATED",
    "EXPERIMENT_COMPLETED",
    "EXPERIMENT_FAILED",
    "EXPERIMENT_CANCELLED",
)


class ExperimentRunner:
    """Executes experiments against a strategy registry."""

    def __init__(
        self,
        registry: StrategyRegistry,
        *,
        services: dict[str, Any] | None = None,
        emit: Any = None,
    ) -> None:
        self._registry = registry
        self._services = dict(services or {})
        self._emit = emit

    async def _trace(self, event: str, **payload: Any) -> None:
        if self._emit is None:
            return
        try:
            await self._emit(event, **payload)
        except Exception:  # a broken trace hook must not kill experiments
            logger.exception("experiment trace emission failed for %s", event)

    async def run(self, experiment: Experiment, *, run_id: str | None = None) -> Experiment:
        """Execute one experiment to a terminal state and return it.

        The experiment record is mutated in place (it is the report) and
        always ends in a terminal status.
        """
        if experiment.is_terminal:
            raise ValueError(f"experiment {experiment.id} is already terminal")
        experiment.run_id = run_id or experiment.run_id
        experiment.status = ExperimentStatus.CREATED
        await self._trace(
            "EXPERIMENT_CREATED",
            experiment_id=experiment.id,
            name=experiment.name,
            baseline=experiment.baseline.ref,
            candidate=experiment.candidate.ref,
        )

        # Resolve both arms BEFORE starting: an unregistered target must
        # fail the experiment before any trial runs (no partial evidence).
        try:
            baseline_strategy = self._registry.get(experiment.baseline.ref)
            candidate_strategy = self._registry.get(experiment.candidate.ref)
        except KeyError as exc:
            return self._fail(experiment, f"unregistered strategy target: {exc}")

        experiment.criteria_frozen()  # frozen-before-run acceptance criteria
        experiment.environment = self._environment_snapshot()
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now(UTC)
        await self._trace(
            "EXPERIMENT_STARTED",
            experiment_id=experiment.id,
            trials=experiment.trials,
            criteria=[c.description for c in experiment.success_criteria],
        )

        deadline = time.monotonic() + experiment.max_total_seconds
        sandbox = Sandbox(experiment_id=experiment.id)
        try:
            baseline_trials = await self._run_arm(
                experiment, baseline_strategy, "baseline", deadline, sandbox
            )
            candidate_trials = await self._run_arm(
                experiment, candidate_strategy, "candidate", deadline, sandbox
            )
        except asyncio.CancelledError:
            sandbox.close()
            experiment.status = ExperimentStatus.CANCELLED
            experiment.completed_at = _utcnow()
            await self._trace(
                "EXPERIMENT_CANCELLED", experiment_id=experiment.id
            )
            return experiment

        if baseline_trials is None or candidate_trials is None:
            # Budget exhausted mid-experiment.
            sandbox.close()
            experiment.status = ExperimentStatus.CANCELLED
            experiment.completed_at = _utcnow()
            experiment.error = "total time budget exhausted before both arms finished"
            await self._trace(
                "EXPERIMENT_CANCELLED",
                experiment_id=experiment.id,
                reason="budget_exhausted",
            )
            return experiment

        experiment.trials_results = baseline_trials + candidate_trials
        experiment.comparison = self._build_comparison(experiment)
        await self._trace(
            "EXPERIMENT_COMPARED",
            experiment_id=experiment.id,
            improvements=experiment.comparison.improvements,
            regressions=experiment.comparison.regressions,
        )

        experiment.evaluation = evaluate_experiment(experiment)
        await self._trace(
            "EXPERIMENT_EVALUATED",
            experiment_id=experiment.id,
            verdict=experiment.evaluation.verdict.value,
            source=experiment.evaluation.source,
        )

        if experiment.evaluation.verdict is Verdict.ERROR:
            experiment.status = ExperimentStatus.ERROR
        else:
            experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = _utcnow()
        await self._trace(
            "EXPERIMENT_COMPLETED"
            if experiment.status is ExperimentStatus.COMPLETED
            else "EXPERIMENT_FAILED",
            experiment_id=experiment.id,
            status=experiment.status.value,
            verdict=experiment.evaluation.verdict.value,
        )
        sandbox.close()
        return experiment

    # -- arms ------------------------------------------------------------------

    async def _run_arm(
        self,
        experiment: Experiment,
        strategy: Any,
        arm: str,
        deadline: float,
        sandbox: Sandbox,
    ) -> list[TrialResult] | None:
        """Run one arm's trials; ``None`` means the budget ran out."""
        results: list[TrialResult] = []
        for index in range(experiment.trials):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "experiment %s: budget exhausted before %s trial %d",
                    experiment.id, arm, index,
                )
                return None
            trial_timeout = min(experiment.per_trial_timeout_seconds, remaining)
            result = await self._run_trial(experiment, strategy, arm, index, trial_timeout, sandbox)
            results.append(result)
            await self._trace(
                "TRIAL_COMPLETED" if result.success else "TRIAL_FAILED",
                experiment_id=experiment.id,
                trial_id=result.trial_id,
                arm=arm,
                index=index,
                duration_ms=result.duration_ms,
            )
            for metric, value in result.metrics.items():
                await self._trace(
                    "METRIC_RECORDED",
                    experiment_id=experiment.id,
                    trial_id=result.trial_id,
                    arm=arm,
                    metric=metric,
                    value=value,
                )
        await self._trace(
            "BASELINE_MEASURED" if arm == "baseline" else "CANDIDATE_MEASURED",
            experiment_id=experiment.id,
            trials=len(results),
            failed=sum(1 for r in results if not r.success),
        )
        return results

    async def _run_trial(
        self,
        experiment: Experiment,
        strategy: Any,
        arm: str,
        index: int,
        trial_timeout: float,
        sandbox: Sandbox,
    ) -> TrialResult:
        trial = TrialResult(
            arm=arm,  # type: ignore[arg-type]
            index=index,
        )
        await self._trace(
            "TRIAL_STARTED", experiment_id=experiment.id, trial_id=trial.trial_id,
            arm=arm, index=index,
        )
        seed = None
        if experiment.seed is not None:
            seed = experiment.seed + (0 if arm == "baseline" else 10_000) + index
        context = sandbox.context_for(
            arm=arm,
            trial_index=index,
            seed=seed,
            params={**experiment.procedure_params, **strategy.config},
            services=dict(self._services),
        )
        started = time.perf_counter()
        try:
            if strategy.procedure is None:
                raise RuntimeError(f"strategy {strategy.ref} has no executable procedure")
            metrics = await asyncio.wait_for(
                strategy.procedure(context), timeout=trial_timeout
            )
            if not isinstance(metrics, dict) or not all(
                isinstance(k, str) and isinstance(v, (int, float))
                for k, v in metrics.items()
            ):
                raise TypeError("procedure must return dict[str, float] metrics")
            trial.metrics = {k: float(v) for k, v in metrics.items()}
            trial.success = True
        except TimeoutError:
            trial.success = False
            trial.error = f"trial timed out after {trial_timeout:.1f}s"
        except Exception as exc:
            trial.success = False
            trial.error = f"{type(exc).__name__}: {exc}"
        trial.duration_ms = int((time.perf_counter() - started) * 1000)
        trial.finished_at = _utcnow()
        return trial

    # -- reporting ---------------------------------------------------------------

    def _build_comparison(self, experiment: Experiment) -> Any:
        metric_names = [m.name for m in experiment.metrics]
        # Also include any metric a procedure reported that was not declared:
        # the data exists, hiding it would be worse than undeclared.
        for trial in experiment.trials_results:
            for name in trial.metrics:
                if name not in metric_names:
                    metric_names.append(name)
        directions = {m.name: m.direction for m in experiment.metrics}
        baseline = aggregate(
            [t for t in experiment.trials_results if t.arm == "baseline"], metric_names
        )
        candidate = aggregate(
            [t for t in experiment.trials_results if t.arm == "candidate"], metric_names
        )
        baseline.ref = experiment.baseline.ref
        candidate.ref = experiment.candidate.ref
        deltas, improvements, regressions = compare(
            baseline, candidate, directions=directions
        )
        from agency.experiments.models import ComparisonReport

        return ComparisonReport(
            baseline=baseline,
            candidate=candidate,
            delta={k: v["abs"] for k, v in deltas.items() if v["abs"] is not None},
            relative={
                k: v["rel"] for k, v in deltas.items() if v.get("rel") is not None
            },
            improvements=improvements,
            regressions=regressions,
            significance_tested=False,
            notes=[
                "descriptive comparison only; no statistical significance test performed"
            ],
        )

    def _fail(self, experiment: Experiment, message: str) -> Experiment:
        experiment.status = ExperimentStatus.ERROR
        experiment.error = message
        experiment.completed_at = _utcnow()
        logger.error("experiment %s failed: %s", experiment.id, message)
        return experiment

    @staticmethod
    def _environment_snapshot() -> dict[str, Any]:
        import os
        import platform

        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "timestamp": _utcnow().isoformat(),
        }


def _utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = ["EXPERIMENT_EVENTS", "ExperimentRunner"]
