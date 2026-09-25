"""Run state machine — explicit, validated stage transitions (spec §4).

The orchestrator is a driver over this machine, not an enormous loop
function: every stage is independently traceable and resumable because
the machine records the current stage in the persisted run record.
"""

from __future__ import annotations

from agency.orchestrator.models import RunStage


class InvalidTransitionError(ValueError):
    """Raised when a stage transition is not allowed by the machine."""


_ALLOWED: dict[RunStage, frozenset[RunStage]] = {
    RunStage.INITIALIZE: frozenset({RunStage.OBSERVE, RunStage.COMPLETE}),
    RunStage.OBSERVE: frozenset({RunStage.PLAN}),
    # PLAN may skip straight to ACT when nothing needs gathering.
    RunStage.PLAN: frozenset(
        {RunStage.RESEARCH, RunStage.COMMUNICATE, RunStage.ACT, RunStage.COMPLETE}
    ),
    # RESEARCH may loop back to PLAN (revised plan on new evidence).
    RunStage.RESEARCH: frozenset(
        {RunStage.COMMUNICATE, RunStage.ACT, RunStage.EVALUATE, RunStage.PLAN}
    ),
    RunStage.COMMUNICATE: frozenset({RunStage.ACT, RunStage.EVALUATE, RunStage.PLAN}),
    RunStage.ACT: frozenset({RunStage.EVALUATE}),
    # EVALUATE is the decision point: continue (PLAN), gather more
    # (RESEARCH/COMMUNICATE), improve, learn, or finish.
    RunStage.EVALUATE: frozenset(
        {
            RunStage.PLAN,
            RunStage.RESEARCH,
            RunStage.COMMUNICATE,
            RunStage.IMPROVE,
            RunStage.LEARN,
            RunStage.COMPLETE,
        }
    ),
    RunStage.LEARN: frozenset({RunStage.PLAN, RunStage.COMPLETE}),
    RunStage.IMPROVE: frozenset({RunStage.PLAN, RunStage.COMPLETE, RunStage.EVALUATE}),
    RunStage.COMPLETE: frozenset(),
}


def can_transition(current: RunStage, target: RunStage) -> bool:
    return target in _ALLOWED.get(current, frozenset())


def require_transition(current: RunStage, target: RunStage) -> None:
    """Validate a stage move; raises :class:`InvalidTransitionError`."""
    if not can_transition(current, target):
        raise InvalidTransitionError(
            f"invalid stage transition {current.value} -> {target.value}; "
            f"allowed: {sorted(s.value for s in _ALLOWED.get(current, frozenset()))}"
        )


def allowed_stages(current: RunStage) -> frozenset[RunStage]:
    return _ALLOWED.get(current, frozenset())


__all__ = [
    "InvalidTransitionError",
    "allowed_stages",
    "can_transition",
    "require_transition",
]
