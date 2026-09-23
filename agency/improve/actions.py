"""Improvement actions — the operator-facing entry points in the action system.

Registered under category ``lab`` (gated by the dark
``SKYNET_ENABLE_SELF_IMPROVEMENT`` flag), alongside the P6 experiment
actions. Three actions:

- ``improve_propose`` — record an external suggestion or run detection +
  proposal for a weakness, producing the full audit artifact set
- ``improve_inspect`` — weaknesses / proposals / versions / regressions
- ``improve_history`` — prior attempts (the "have I tried this?" seam)

Testing, applying and rolling back stay **CLI/operator** operations: they
change persistent system state and require the explicit human gates the
pipeline enforces. External suggestions (§19) enter through
``improve_propose`` with ``origin="external"`` and pass the exact same
gates as internal detections — they are recorded data, never instructions.
"""

from __future__ import annotations

import logging
from typing import Any

from agency.actions.base import Action, ActionContext, ActionError, ActionSpec
from agency.improve.models import (
    CandidateStrategy,
    ImprovementProposal,
)
from agency.improve.pipeline import ImprovementPipeline

logger = logging.getLogger("skynet.improve.actions")


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"missing or empty string parameter {key!r}")
    return value.strip()


class ImproveProposeAction(Action):
    """Propose an improvement from a known weakness (or record external input)."""

    name = "improve_propose"
    category = "lab"
    description = (
        "Create an improvement hypothesis + proposal + candidate (config-only, "
        "no code changes) for a detected weakness. External suggestions enter "
        "here with origin='external' and go through the same evaluation gates."
    )

    def __init__(self, pipeline: ImprovementPipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        weakness_id = _require_str(spec.params, "weakness_id")
        origin = spec.params.get("origin", "internal")
        if origin not in {"internal", "external"}:
            raise ActionError("parameter 'origin' must be 'internal' or 'external'")
        candidate_config = spec.params.get("candidate_config")
        if candidate_config is not None and not isinstance(candidate_config, dict):
            raise ActionError("parameter 'candidate_config' must be an object")
        hypothesis, proposal, candidate = await self._pipeline.propose(
            weakness_id,
            origin=origin,
            proposed_by=str(spec.params.get("proposed_by", "skynet")),
            candidate_config=dict(candidate_config or {}),
            candidate_changes=str(spec.params.get("changes", "")),
        )
        return {
            "ok": True,
            "hypothesis_id": hypothesis.id,
            "proposal_id": proposal.id,
            "candidate_ref": candidate.ref,
            "statement": hypothesis.statement,
            "acceptance_criteria": [
                c.description for c in proposal.acceptance_criteria
            ],
            "prior_attempts": proposal.metadata.get("prior_attempts", []),
        }


class ImproveInspectAction(Action):
    """Inspect the improvement state: weaknesses, proposals, versions."""

    name = "improve_inspect"
    category = "lab"
    description = (
        "Inspect self-improvement state: list weaknesses, proposals, version "
        "ledger entries, the active pointer, or recorded regressions."
    )

    def __init__(self, pipeline: ImprovementPipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        del ctx
        view = spec.params.get("view", "weaknesses")
        if view == "weaknesses":
            status = spec.params.get("status")
            items = [
                {
                    "id": w.id,
                    "category": w.category.value,
                    "description": w.description,
                    "severity": w.severity,
                    "frequency": w.frequency,
                    "status": w.status.value,
                    "affected_strategy": w.affected_strategy,
                }
                for w in self._pipeline.list_weaknesses(status)
            ]
            return {"ok": True, "view": view, "count": len(items), "items": items}
        if view == "proposals":
            status = spec.params.get("status")
            items = [
                {
                    "id": p.id,
                    "status": p.status.value,
                    "target": p.target,
                    "candidate_ref": p.candidate_ref,
                    "decision": p.decision,
                    "reason": p.decision_reason,
                    "experiment_id": p.experiment_id,
                }
                for p in self._pipeline.proposals(status)
            ]
            return {"ok": True, "view": view, "count": len(items), "items": items}
        if view == "versions":
            name = _require_str(spec.params, "name")
            items = [
                {
                    "ref": v.ref,
                    "id": v.id,
                    "parent_version": v.parent_version,
                    "proposal_id": v.proposal_id,
                    "change_description": v.change_description,
                    "created_at": v.created_at.isoformat(),
                }
                for v in self._pipeline.versions(name)
            ]
            return {
                "ok": True,
                "view": view,
                "name": name,
                "count": len(items),
                "items": items,
            }
        if view == "active":
            name = _require_str(spec.params, "name")
            active = self._pipeline.active(name)
            return {
                "ok": True,
                "view": view,
                "active": active.model_dump(mode="json") if active else None,
            }
        if view == "regressions":
            items = [e.model_dump(mode="json") for e in self._pipeline.regressions()]
            return {"ok": True, "view": view, "count": len(items), "items": items}
        raise ActionError(
            "parameter 'view' must be one of: weaknesses, proposals, "
            "versions, active, regressions"
        )


class ImproveHistoryAction(Action):
    """Prior attempts for a weakness (dedupe before re-proposing)."""

    name = "improve_history"
    category = "lab"
    description = (
        "Retrieve previous improvement attempts for a weakness — what was "
        "tried, what happened, why it was rejected. Check before proposing."
    )

    def __init__(self, pipeline: ImprovementPipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        del ctx
        weakness_id = _require_str(spec.params, "weakness_id")
        prior = self._pipeline.proposals()
        items = [
            {
                "proposal_id": p.id,
                "status": p.status.value,
                "decision": p.decision,
                "reason": p.decision_reason,
                "candidate_ref": p.candidate_ref,
                "experiment_id": p.experiment_id,
                "created_at": p.created_at.isoformat(),
            }
            for p in prior
            if p.weakness_id == weakness_id
        ]
        return {"ok": True, "weakness_id": weakness_id, "count": len(items), "items": items}


__all__ = [
    "CandidateStrategy",  # re-export for typing convenience
    "ImproveHistoryAction",
    "ImproveInspectAction",
    "ImproveProposeAction",
    "ImprovementProposal",
]
