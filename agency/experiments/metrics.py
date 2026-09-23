"""Generic metrics for experiments (aggregation + comparison math).

The framework is metric-agnostic: any ``dict[str, float]`` per trial is
valid; :class:`MetricDefinition` only adds direction semantics for
regression detection. These helpers never fabricate values — missing
metrics surface as ``None`` at the call site.
"""

from __future__ import annotations

import math
from typing import Any

from agency.experiments.models import ArmSummary, TrialResult


def aggregate(trials: list[TrialResult], metric_names: list[str]) -> ArmSummary:
    """Aggregate one arm's trials into mean/stddev/min/max per metric.

    Failed trials are counted but excluded from the aggregates (their
    absence is visible via ``failed_trials``).
    """
    summary = ArmSummary(ref="")
    summary.trials = len(trials)
    summary.failed_trials = sum(1 for trial in trials if not trial.success)
    ok = [trial for trial in trials if trial.success]
    for name in metric_names:
        values = [trial.metrics[name] for trial in ok if name in trial.metrics]
        if not values:
            continue
        summary.mean[name] = _round(math.fsum(values) / len(values))
        summary.min[name] = min(values)
        summary.max[name] = max(values)
        if len(values) > 1:
            summary.stddev[name] = _round(_stddev(values))
        else:
            summary.stddev[name] = 0.0
    return summary


def _stddev(values: list[float]) -> float:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _round(value: float, digits: int = 6) -> float:
    return round(value, digits)


def compare(
    baseline: ArmSummary,
    candidate: ArmSummary,
    *,
    directions: dict[str, str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Descriptive comparison of two arms.

    Returns ``(deltas, improvements, regressions)`` where ``deltas`` maps
    ``metric -> {abs, rel}``. ``improvements``/``regressions`` respect the
    metric's declared direction. No significance testing happens here.
    """
    deltas: dict[str, Any] = {}
    improvements: list[str] = []
    regressions: list[str] = []
    for metric, direction in directions.items():
        base = baseline.mean.get(metric)
        cand = candidate.mean.get(metric)
        if base is None or cand is None:
            deltas[metric] = {"abs": None, "rel": None, "note": "missing on one arm"}
            continue
        abs_delta = _round(cand - base)
        rel_delta = _round(abs_delta / abs(base)) if base != 0 else None
        good = abs_delta > 0 if direction == "maximize" else abs_delta < 0
        if abs_delta == 0:
            pass
        elif good:
            improvements.append(metric)
        else:
            regressions.append(metric)
        deltas[metric] = {"abs": abs_delta, "rel": rel_delta}
    return deltas, improvements, regressions


__all__ = ["aggregate", "compare"]
