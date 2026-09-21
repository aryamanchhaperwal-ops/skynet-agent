"""Deriving memory candidates from Skynet's existing data streams.

Three deterministic extractors bridge the phases already built into memory:

- observations (web research, tool outputs) → semantic/episodic memories
- AI conversations → episodic transcript reference + semantic claim notes
- experiences (actions, errors, run summaries) → episodic memories

Every candidate carries full provenance and an importance score; the
:class:`agency.memory.manager.MemoryManager` threshold decides what is
actually stored. Extraction is deliberately *not* an LLM call yet — the
same function signatures accept an optional intelligence service later
(importance scoring and content distillation become model calls; the
provenance plumbing does not change).

The raw stream is never replaced: extraction adds derived memories, the
originals stay in experiences/trace/conversations.
"""

from __future__ import annotations

import logging
from typing import Any

from agency.memory.models import MemoryType
from agency.observation import Observation

logger = logging.getLogger("skynet.memory.extraction")

#: Content cap for extracted memory bodies (stores cap harder at 20k).
EXTRACT_CONTENT_MAX_CHARS = 2000


def _clip(text: str, limit: int = EXTRACT_CONTENT_MAX_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class MemoryCandidate:
    """A proposed memory awaiting the manager's importance gate.

    Kept as a plain dict-shaped dataclass rather than reusing
    :class:`MemoryRecord` directly: candidates may be filtered, merged or
    discarded before storage, and the manager performs the gate + record
    construction in one place.
    """

    __slots__ = (
        "confidence",
        "content",
        "goal_id",
        "importance",
        "metadata",
        "origin",
        "provenance",
        "run_id",
        "source",
        "summary",
        "tags",
        "type",
    )

    def __init__(
        self,
        *,
        content: str,
        memory_type: MemoryType,
        summary: str = "",
        source: str = "",
        origin: str = "observation",
        provenance: dict[str, Any] | None = None,
        importance: float = 0.5,
        confidence: float | None = None,
        tags: list[str] | None = None,
        goal_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.type = memory_type
        self.summary = summary
        self.source = source
        self.origin = origin
        self.provenance = dict(provenance or {})
        self.importance = importance
        self.confidence = confidence
        self.tags = list(tags or [])
        self.goal_id = goal_id
        self.run_id = run_id
        self.metadata = dict(metadata or {})


def memories_from_observations(
    observations: list[Observation],
    *,
    goal_id: str | None = None,
    run_id: str | None = None,
) -> list[MemoryCandidate]:
    """Research/tool observations → memory candidates.

    Error observations are kept as episodic memories with raised importance
    (failures are the most valuable learning material); successful research
    observations become semantic memories whose confidence comes from the
    observation itself when present.
    """
    candidates: list[MemoryCandidate] = []
    for observation in observations:
        content = _clip(
            observation.content if isinstance(observation.content, str)
            else repr(observation.content)
        )
        if not content:
            continue
        is_error = observation.kind == "error"
        candidates.append(
            MemoryCandidate(
                content=content,
                memory_type=MemoryType.EPISODIC if is_error else MemoryType.SEMANTIC,
                summary=_clip(observation.summary or f"Observation from {observation.source}", 300),
                source=observation.source,
                origin="observation",
                provenance={
                    "observation_id": observation.id,
                    "observation_kind": observation.kind,
                    "observed_at": observation.observed_at.isoformat(),
                    **(observation.metadata.get("provenance", {}) or {}),
                },
                importance=0.7 if is_error else (observation.confidence or 0.5),
                confidence=observation.confidence,
                tags=["observation", observation.source.split(":", 1)[0]]
                + (["error"] if is_error else []),
                goal_id=goal_id or observation.goal_id,
                run_id=run_id or observation.run_id,
                metadata={"kind": observation.kind},
            )
        )
    return candidates


def memories_from_conversation(
    outcome: Any,
    *,
    goal_id: str | None = None,
    run_id: str | None = None,
) -> list[MemoryCandidate]:
    """AI conversation outcome → memory candidates.

    Produces one episodic memory per conversation (the transcript itself
    stays in the conversation record — this is a pointer plus digest, never
    a replacement) and one semantic candidate per turn whose analysis
    surfaced claim candidates (flagged, unverified, provenance-complete).
    """
    conversation = outcome.conversation
    participants = conversation.participant_id
    turns = list(outcome.turns)
    candidates: list[MemoryCandidate] = []

    transcript_digest = _clip(
        " | ".join(
            f"Q: {turn.question[:120]} → A: {turn.answer[:180]}" for turn in turns
        )
    )
    candidates.append(
        MemoryCandidate(
            content=transcript_digest,
            memory_type=MemoryType.EPISODIC,
            summary=f"Conversation with {participants} ({len(turns)} turn(s))",
            source=f"ai:{participants}",
            origin="conversation",
            provenance={
                "conversation_id": conversation.id,
                "participant_id": participants,
                "goal_id": conversation.goal_id or goal_id,
                "run_id": conversation.run_id or run_id,
                "message_ids": [mid for turn in turns for mid in turn.message_ids],
            },
            importance=0.6,
            tags=["ai-conversation", "episodic"],
            goal_id=goal_id,
            run_id=run_id,
            metadata={
                "turns": len(turns),
                "status": outcome.status.value
                if hasattr(outcome.status, "value")
                else str(outcome.status),
            },
        )
    )

    for index, turn in enumerate(turns):
        for claim in turn.analysis_claims:
            candidates.append(
                MemoryCandidate(
                    content=_clip(claim),
                    memory_type=MemoryType.SEMANTIC,
                    summary=f"Claim from {participants} (turn {index + 1})",
                    source=f"ai:{participants}",
                    origin="conversation",
                    provenance={
                        "conversation_id": conversation.id,
                        "participant_id": participants,
                        "message_id": turn.message_ids[1],
                        "turn": index + 1,
                        "verified": False,
                    },
                    importance=0.55,
                    tags=["ai-claim", "unverified"],
                    goal_id=goal_id,
                    run_id=run_id,
                    metadata={"analysis": "directive-claim-extraction"},
                )
            )
    return candidates


def memories_from_experiences(
    records: list[Any],
    *,
    goal_id: str | None = None,
    run_id: str | None = None,
) -> list[MemoryCandidate]:
    """Important experiences → episodic memory candidates.

    Deterministic importance policy:
    - ``error`` experiences → 0.75 (failures are high-value)
    - ``summary`` experiences (run results) → 0.65
    - ``action`` experiences that failed → 0.7
    - everything else → below default thresholds (not stored automatically;
      the full stream already lives in the experiences table)
    """
    candidates: list[MemoryCandidate] = []
    for record in records:
        kind = getattr(record, "kind", "")
        outcome = getattr(record, "outcome", "neutral")
        summary = getattr(record, "summary", "")
        if kind == "error":
            importance, tags = 0.75, ["experience", "error"]
        elif kind == "summary":
            importance, tags = 0.65, ["experience", "run-summary"]
        elif kind == "action" and outcome == "failure":
            importance, tags = 0.7, ["experience", "action-failure"]
        else:
            continue
        candidates.append(
            MemoryCandidate(
                content=_clip(summary),
                memory_type=MemoryType.EPISODIC,
                summary=_clip(summary, 300),
                source=f"experience:{kind}",
                origin="experience",
                provenance={
                    "experience_id": getattr(record, "id", ""),
                    "experience_kind": kind,
                    "experience_outcome": outcome,
                    "run_id": getattr(record, "run_id", None),
                },
                importance=importance,
                tags=tags,
                goal_id=goal_id or getattr(record, "goal_id", None),
                run_id=run_id or getattr(record, "run_id", None),
            )
        )
    return candidates


class Consolidator:
    """Future-ready consolidation with a deterministic first version.

    Blueprint §7.8: many experiences → recurring pattern → candidate
    knowledge → evaluation → semantic/procedural memory. This
    implementation detects the simplest deterministic pattern — episodic
    memories sharing two or more tags — and proposes one semantic summary
    per pattern group. The interface (``identify_candidates`` returning
    memory candidates with consolidated provenance) is what a future
    LLM-driven consolidation implements; nothing else changes.
    """

    MIN_GROUP_SIZE = 3

    def __init__(self, min_group_size: int = 3) -> None:
        self.min_group_size = max(2, min_group_size)

    async def identify_candidates(
        self,
        existing: list[tuple[Any, float]],
    ) -> list[MemoryCandidate]:
        """Group episodic memories by shared tag pairs; propose summaries.

        ``existing`` is ``(record, score)`` pairs as returned by
        ``MemoryManager.search``. Proposed candidates carry the source
        memory ids in provenance — full traceability of derived knowledge.
        """
        episodic = [record for record, _ in existing if record.type is MemoryType.EPISODIC]
        groups: dict[frozenset[str], list[Any]] = {}
        for record in episodic:
            tag_set = frozenset(record.tags)
            if len(tag_set) < 2:
                continue
            for other in episodic:
                if other.id == record.id:
                    continue
                shared = tag_set & frozenset(other.tags)
                if len(shared) >= 2:
                    key = frozenset(sorted(shared)[:2])
                    group = groups.setdefault(key, [])
                    if record.id not in {m.id for m in group}:
                        group.append(record)
                    if other.id not in {m.id for m in group}:
                        group.append(other)
        candidates: list[MemoryCandidate] = []
        for shared_tags, members in groups.items():
            if len(members) < self.min_group_size:
                continue
            member_ids = [m.id for m in members]
            candidates.append(
                MemoryCandidate(
                    content=(
                        f"Recurring pattern across {len(members)} episodic memories "
                        f"sharing tags {sorted(shared_tags)}: "
                        + " | ".join(_clip(m.summary or m.content, 120) for m in members[:5])
                    ),
                    memory_type=MemoryType.SEMANTIC,
                    summary=f"Consolidated pattern: {sorted(shared_tags)}",
                    source="consolidation",
                    origin="consolidation",
                    provenance={
                        "source_memory_ids": member_ids,
                        "shared_tags": sorted(shared_tags),
                        "method": "deterministic-tag-grouping",
                    },
                    importance=0.6,
                    tags=["consolidated", *sorted(shared_tags)[:2]],
                )
            )
        return candidates


__all__ = [
    "EXTRACT_CONTENT_MAX_CHARS",
    "Consolidator",
    "MemoryCandidate",
    "memories_from_conversation",
    "memories_from_experiences",
    "memories_from_observations",
]
