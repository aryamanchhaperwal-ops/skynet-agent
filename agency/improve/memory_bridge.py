"""Improvement → memory bridge (phase spec §14/§15).

Every decision in the improvement loop becomes a memory so a future run
can answer: *have I attempted this improvement before? what happened? why
was it rejected? which version is active?* Provenance is mandatory:
weakness → hypothesis → proposal → candidate → experiment → verdict chain
rides along on every record.

The raw artifacts stay in the improvement store and experiment registry;
memories are the *derived, recallable* layer — never a replacement for the
audit trail. A broken memory layer must never break an improvement: all
recording failures are logged and swallowed here.
"""

from __future__ import annotations

import logging
from typing import Any

from agency.experiments.models import Experiment
from agency.improve.models import (
    CandidateStrategy,
    ImprovementHypothesis,
    ImprovementProposal,
    Weakness,
)
from agency.memory.manager import MemoryManager
from agency.memory.models import MemoryType

logger = logging.getLogger("skynet.improve.memory")

#: Origin label for improvement-derived memories (first provenance hop).
ORIGIN_IMPROVEMENT = "improvement"


def _proposal_provenance(
    weakness: Weakness | None,
    hypothesis: ImprovementHypothesis | None,
    proposal: ImprovementProposal,
    candidate: CandidateStrategy | None,
    experiment: Experiment | None,
) -> dict[str, Any]:
    return {
        "weakness_id": proposal.weakness_id,
        "weakness_category": weakness.category.value if weakness else None,
        "hypothesis_id": proposal.hypothesis_id,
        "proposal_id": proposal.id,
        "proposal_status": proposal.status.value,
        "target": proposal.target,
        "candidate_ref": candidate.ref if candidate else proposal.candidate_ref,
        "candidate_id": candidate.id if candidate else None,
        "experiment_id": proposal.experiment_id,
        "origin": proposal.origin,
        "proposed_by": proposal.proposed_by,
    }


async def record_proposal(
    proposal: ImprovementProposal,
    *,
    weakness: Weakness | None = None,
    hypothesis: ImprovementHypothesis | None = None,
    candidate: CandidateStrategy | None = None,
    experiment: Experiment | None = None,
    memory: MemoryManager | None,
) -> list[Any]:
    """Record a proposal's decision into long-term memory.

    Called at the decision points of the lifecycle: after testing (with
    the experiment's verdict) and after rejection/acceptance/rollback.
    Importance scales with how much the outcome teaches:

    - accepted + approved + applied: 0.9 (a worked improvement)
    - rejected: 0.65 (failures are valuable — do not retry blindly)
    - inconclusive: 0.5 (recorded, low salience)
    """
    if memory is None:
        return []
    stored: list[Any] = []
    importance = {
        "accepted": 0.9,
        "rejected": 0.65,
        "rolled_back": 0.8,
        "proposed": 0.4,
        "approved_for_test": 0.4,
        "testing": 0.4,
    }.get(proposal.status.value, 0.4)
    provenance = _proposal_provenance(
        weakness, hypothesis, proposal, candidate, experiment
    )
    summary = proposal.decision_reason or proposal.proposed_change
    try:
        record = await memory.store(
            content=(
                f"Improvement proposal {proposal.id[:12]} ({proposal.status.value}): "
                f"{proposal.proposed_change}. "
                f"Weakness: {weakness.description if weakness else proposal.weakness_id}. "
                f"Outcome: {summary}"
            )[:4000],
            summary=f"improvement {proposal.status.value}: {proposal.target}",
            memory_type=MemoryType.EPISODIC,
            source=f"proposal:{proposal.id[:12]}",
            origin=ORIGIN_IMPROVEMENT,
            provenance=provenance,
            importance=importance,
            confidence=None,
            tags=["improvement", proposal.status.value, "proposal"],
            force=True,
        )
        if record is not None:
            stored.append(record)

        # Rejected proposals become procedural guidance: what does NOT work.
        if proposal.status.value == "rejected":
            record = await memory.store(
                content=(
                    f"Strategy result: {proposal.candidate_ref or proposal.target} did NOT "
                    f"improve on {proposal.target} ({summary}). "
                    "Do not re-propose the same configuration without new evidence; "
                    "check the experiment record for measured numbers."
                )[:4000],
                summary=(
                    f"strategy: {proposal.candidate_ref or 'candidate'} failed to "
                    f"beat {proposal.target}"
                ),
                memory_type=MemoryType.PROCEDURAL,
                source=f"proposal:{proposal.id[:12]}",
                origin=ORIGIN_IMPROVEMENT,
                provenance=provenance,
                importance=0.7,
                confidence=None,
                tags=["improvement", "rejected", "strategy"],
                force=True,
            )
            if record is not None:
                stored.append(record)
    except Exception:
        # Memory amplifies the improvement loop; it must never gate it.
        logger.exception(
            "failed to record improvement proposal %s into memory", proposal.id
        )
    return stored


__all__ = ["ORIGIN_IMPROVEMENT", "record_proposal"]
