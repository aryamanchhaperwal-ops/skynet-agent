"""Experiment → memory bridge (phase spec §14/§15/§20).

A finished experiment becomes:

1. an **episodic memory** (what happened, with verdict + metrics), and
2. on SUCCESS, a **procedural memory** candidate (the candidate strategy
   worked better *under these conditions*) — recorded with confidence
   bounds, never as permanent truth.

Provenance is mandatory: every memory carries ``experiment_id``,
hypothesis, both target refs, and the goal/run linkage so Skynet can
always answer "why do I believe this strategy works?". A broken memory
bridge must never fail an experiment: :meth:`record_experiment` returns
``None`` records on failure and logs.
"""

from __future__ import annotations

import logging
from typing import Any

from agency.experiments.models import Experiment, Verdict
from agency.memory.manager import MemoryManager
from agency.memory.models import MemoryType

logger = logging.getLogger("skynet.experiments.memory")

#: Origin label for experiment-derived memories (first provenance hop).
ORIGIN_EXPERIMENT = "experiment"


def _provenance(experiment: Experiment) -> dict[str, Any]:
    return {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "objective": experiment.objective,
        "hypothesis": experiment.hypothesis.statement,
        "baseline": experiment.baseline.ref,
        "candidate": experiment.candidate.ref,
        "trials": len(experiment.trials_results),
        "run_id": experiment.run_id,
        "goal_id": experiment.parent_goal_id,
        "status": experiment.status.value,
        "verdict": experiment.evaluation.verdict.value if experiment.evaluation else None,
    }


def _verdict_confidence(verdict: Verdict, trials: int) -> float:
    """Confidence in the *recording* (not the truth of the claim)."""
    base = {
        Verdict.SUCCESS: 0.7,
        Verdict.FAILURE: 0.7,
        Verdict.INCONCLUSIVE: 0.3,
        Verdict.ERROR: 0.1,
        Verdict.CANCELLED: 0.1,
    }[verdict]
    # More trials → slightly more confidence in the observation.
    bump = min(0.2, 0.05 * max(0, trials - 3))
    return round(min(1.0, base + bump), 2)


def _episodic_content(experiment: Experiment) -> str:
    comparison = experiment.comparison
    evaluation = experiment.evaluation
    lines = [
        f"Experiment {experiment.name!r} ({experiment.id[:12]}) finished: "
        f"status={experiment.status.value}",
        f"hypothesis: {experiment.hypothesis.statement}",
        f"compared {experiment.baseline.ref} (baseline) vs "
        f"{experiment.candidate.ref} (candidate) over "
        f"{len(experiment.trials_results)} trials",
    ]
    if comparison is not None:
        lines.append(f"baseline mean: {comparison.baseline.mean}")
        lines.append(f"candidate mean: {comparison.candidate.mean}")
        if comparison.improvements:
            lines.append(f"improved: {', '.join(comparison.improvements)}")
        if comparison.regressions:
            lines.append(f"regressed: {', '.join(comparison.regressions)}")
        lines.append(
            "descriptive comparison only; no statistical significance tested"
        )
    if evaluation is not None:
        lines.append(f"verdict: {evaluation.verdict.value} — {evaluation.explanation}")
    return "\n".join(lines)


def _candidate_summary(experiment: Experiment) -> str:
    evaluation = experiment.evaluation
    assert evaluation is not None
    return (
        f"{experiment.candidate.ref} beat {experiment.baseline.ref} "
        f"({evaluation.verdict.value}) on: "
        f"{', '.join(experiment.comparison.improvements) if experiment.comparison else 'n/a'}"
    )


async def record_experiment(
    experiment: Experiment,
    memory: MemoryManager | None,
) -> list[Any]:
    """Store a finished experiment as memories; returns stored records.

    Importance: SUCCESS 0.8, FAILURE 0.6 (failures are valuable),
    INCONCLUSIVE 0.5, ERROR/CANCELLED 0.45. Everything carries the full
    provenance chain; nothing is stored when ``memory`` is ``None``.
    """
    if memory is None or not experiment.is_terminal:
        return []
    evaluation = experiment.evaluation
    verdict = evaluation.verdict if evaluation else Verdict.ERROR
    importance = {
        Verdict.SUCCESS: 0.8,
        Verdict.FAILURE: 0.6,
        Verdict.INCONCLUSIVE: 0.5,
        Verdict.ERROR: 0.45,
        Verdict.CANCELLED: 0.45,
    }[verdict]
    provenance = _provenance(experiment)
    stored: list[Any] = []

    try:
        episodic = await memory.store(
            content=_episodic_content(experiment)[:4000],
            summary=f"experiment {experiment.name}: {verdict.value}",
            memory_type=MemoryType.EPISODIC,
            source=f"experiment:{experiment.id[:12]}",
            origin=ORIGIN_EXPERIMENT,
            provenance=provenance,
            importance=importance,
            confidence=_verdict_confidence(verdict, len(experiment.trials_results)),
            tags=["experiment", experiment.status.value, verdict.value,
                  experiment.candidate.ref.split(":")[0]],
            goal_id=experiment.parent_goal_id,
            run_id=experiment.run_id,
            force=True,
        )
        if episodic is not None:
            stored.append(episodic)

        if verdict is Verdict.SUCCESS and experiment.comparison is not None:
            procedural = await memory.store(
                content=(
                    f"Strategy result: {_candidate_summary(experiment)}. "
                    f"Conditions: {experiment.objective or experiment.name}; "
                    f"{len(experiment.trials_results)} trials; metrics "
                    f"{experiment.comparison.candidate.mean}. "
                    "This is measured evidence under these conditions, not a "
                    "guarantee — re-verify before relying on it."
                )[:4000],
                summary=f"strategy: prefer {experiment.candidate.ref} "
                        f"over {experiment.baseline.ref}",
                memory_type=MemoryType.PROCEDURAL,
                source=f"experiment:{experiment.id[:12]}",
                origin=ORIGIN_EXPERIMENT,
                provenance=provenance,
                importance=0.85,
                confidence=_verdict_confidence(verdict, len(experiment.trials_results)),
                tags=["strategy", "experiment", experiment.candidate.ref.split(":")[0]],
                goal_id=experiment.parent_goal_id,
                run_id=experiment.run_id,
                force=True,
            )
            if procedural is not None:
                stored.append(procedural)
    except Exception:
        # Memory is an amplifier, not a dependency of experiments.
        logger.exception(
            "failed to record experiment %s into memory", experiment.id
        )
    return stored


__all__ = ["ORIGIN_EXPERIMENT", "record_experiment"]
