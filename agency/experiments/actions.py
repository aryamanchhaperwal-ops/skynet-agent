"""Experiment actions — the lab category inside the existing action system.

Three actions, registered under category ``lab`` (gated by the dark
``SKYNET_ENABLE_SELF_IMPROVEMENT`` flag):

- ``experiment_run``     — create + execute one experiment to a verdict
- ``experiment_list``    — prior experiment history (the "have I tried
  this before?" seam; call before proposing a new experiment)
- ``experiment_inspect`` — full record of one prior experiment

These are *the only* path the loop has into the experimentation engine:
no second execution framework, and experiments run through the same
tracer as everything else.
"""

from __future__ import annotations

import logging
from typing import Any

from agency.actions.base import Action, ActionContext, ActionError, ActionSpec
from agency.experiments.experiments_store import ExperimentRegistry
from agency.experiments.memory_bridge import record_experiment
from agency.experiments.models import (
    Experiment,
    ExperimentStatus,
    Hypothesis,
    MetricDefinition,
    SuccessCriterion,
    TargetRef,
)
from agency.experiments.runner import ExperimentRunner

logger = logging.getLogger("skynet.experiments.actions")


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"missing or empty string parameter {key!r}")
    return value.strip()


def _parse_target(params: dict[str, Any], key: str) -> TargetRef:
    raw = _require_str(params, key)
    if ":" not in raw:
        raise ActionError(f"parameter {key!r} must be 'name:version', got {raw!r}")
    name, _, version = raw.partition(":")
    try:
        return TargetRef(name=name, version=version)
    except ValueError as exc:
        raise ActionError(f"invalid target {raw!r}: {exc}") from exc


def _parse_criteria(params: dict[str, Any]) -> list[SuccessCriterion]:
    raw = params.get("success_criteria")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ActionError("parameter 'success_criteria' must be a list")
    criteria: list[SuccessCriterion] = []
    for item in raw:
        if not isinstance(item, dict) or "kind" not in item or "metric" not in item:
            raise ActionError(
                "each success criterion needs 'kind' and 'metric' keys"
            )
        try:
            criteria.append(
                SuccessCriterion(
                    kind=item["kind"],
                    metric=item["metric"],
                    threshold=float(item.get("threshold", 0.0)),
                )
            )
        except (ValueError, TypeError) as exc:
            raise ActionError(f"invalid success criterion {item!r}: {exc}") from exc
    return criteria


def _parse_metrics(params: dict[str, Any]) -> list[MetricDefinition]:
    raw = params.get("metrics")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ActionError("parameter 'metrics' must be a list")
    metrics: list[MetricDefinition] = []
    for item in raw:
        if isinstance(item, str):
            metrics.append(MetricDefinition(name=item))
            continue
        if isinstance(item, dict) and item.get("name"):
            metrics.append(
                MetricDefinition(
                    name=str(item["name"]),
                    direction=item.get("direction", "maximize"),
                    description=str(item.get("description", "")),
                )
            )
            continue
        raise ActionError(f"invalid metric definition {item!r}")
    return metrics


class ExperimentRunAction(Action):
    """Create and execute one experiment; returns the full report."""

    name = "experiment_run"
    category = "lab"
    description = (
        "Run one controlled experiment: baseline vs candidate strategy "
        "(registered artifacts) over N trials with frozen acceptance "
        "criteria. Experiments operate on strategies/data only — never code."
    )

    def __init__(
        self,
        runner: ExperimentRunner,
        registry: ExperimentRegistry,
        *,
        memory: Any = None,
    ) -> None:
        self._runner = runner
        self._store = registry
        self._memory = memory

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        name = _require_str(spec.params, "name")
        hypothesis_raw = spec.params.get("hypothesis")
        if not isinstance(hypothesis_raw, str) or not hypothesis_raw.strip():
            raise ActionError("missing or empty string parameter 'hypothesis'")
        baseline = _parse_target(spec.params, "baseline")
        candidate = _parse_target(spec.params, "candidate")
        criteria = _parse_criteria(spec.params)
        if not criteria:
            raise ActionError(
                "an experiment without success_criteria can only be "
                "INCONCLUSIVE; pass at least one criterion"
            )
        metrics = _parse_metrics(spec.params)
        trials = spec.params.get("trials", 3)
        if not isinstance(trials, int) or not 1 <= trials <= 100:
            raise ActionError("parameter 'trials' must be an integer in [1, 100]")
        seed = spec.params.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ActionError("parameter 'seed' must be an integer")
        procedure_params = spec.params.get("procedure_params")
        if procedure_params is not None and not isinstance(procedure_params, dict):
            raise ActionError("parameter 'procedure_params' must be an object")

        experiment = Experiment(
            name=name,
            description=str(spec.params.get("description", "")),
            objective=str(spec.params.get("objective", "")),
            hypothesis=Hypothesis(
                statement=hypothesis_raw.strip(),
                rationale=str(spec.params.get("rationale", "")),
                expected_effect=str(spec.params.get("expected_effect", "")),
                success_criteria=criteria,
                provenance={"proposed_by": "skynet", "run_id": ctx.run_id},
            ),
            baseline=baseline,
            candidate=candidate,
            metrics=metrics,
            trials=trials,
            seed=seed,
            procedure_params=dict(procedure_params or {}),
            parent_goal_id=ctx.goal_id,
        )
        await self._store.save(experiment)
        await ctx.emit_trace(
            "EXPERIMENT_CREATED",
            experiment_id=experiment.id,
            name=experiment.name,
            baseline=baseline.ref,
            candidate=candidate.ref,
        )

        finished = await self._runner.run(experiment, run_id=ctx.run_id)
        await self._store.save(finished)
        memories = await record_experiment(finished, self._memory)
        evaluation = finished.evaluation
        return {
            "ok": True,
            "experiment_id": finished.id,
            "status": finished.status.value,
            "verdict": evaluation.verdict.value if evaluation else None,
            "explanation": evaluation.explanation if evaluation else finished.error,
            "criteria_results": evaluation.criteria_results if evaluation else [],
            "baseline_mean": finished.comparison.baseline.mean if finished.comparison else {},
            "candidate_mean": finished.comparison.candidate.mean if finished.comparison else {},
            "delta": finished.comparison.delta if finished.comparison else {},
            "improvements": finished.comparison.improvements if finished.comparison else [],
            "regressions": finished.comparison.regressions if finished.comparison else [],
            "significance_tested": False,
            "memories_stored": [m.id for m in memories if m is not None],
            "trials": len(finished.trials_results),
        }


class ExperimentListAction(Action):
    """List prior experiments (query before creating new ones)."""

    name = "experiment_list"
    category = "lab"
    description = (
        "List previously run experiments (status/verdict summaries). Check "
        "here before proposing a new experiment — prior evidence exists."
    )

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._store = registry

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        del ctx
        status = spec.params.get("status")
        if status is not None and status not in {s.value for s in ExperimentStatus}:
            raise ActionError(
                f"parameter 'status' must be one of: {sorted(s.value for s in ExperimentStatus)}"
            )
        summaries = await self._store.list(status=status)
        return {
            "ok": True,
            "count": len(summaries),
            "experiments": [s.model_dump(mode="json") for s in summaries],
        }


class ExperimentInspectAction(Action):
    """Full record of one experiment (reproducibility view)."""

    name = "experiment_inspect"
    category = "lab"
    description = (
        "Inspect one experiment's full record: hypothesis, criteria, trial "
        "metrics, comparison and verdict (provenance-complete)."
    )

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._store = registry

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        del ctx
        experiment_id = _require_str(spec.params, "experiment_id")
        record = await self._store.get(experiment_id)
        if record is None:
            raise ActionError(f"unknown experiment id {experiment_id!r}")
        payload = record.model_dump(mode="json")
        # Procedures are code; the registry never serializes them anyway.
        payload.pop("procedure", None)
        return {"ok": True, "experiment": payload}


__all__ = [
    "ExperimentInspectAction",
    "ExperimentListAction",
    "ExperimentRunAction",
]
