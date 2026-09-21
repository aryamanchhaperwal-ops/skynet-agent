"""Structured models for the Skynet AI ↔ AI communication layer.

Everything exchanged with an external AI system is data with **provenance**:
which participant said it, in which conversation, during which run/goal, and
when. A response is never stored as anonymous knowledge — Skynet can always
answer "which AI said this?" and "during which conversation, for what goal?".

Truthfulness is deliberately *not* decided here: analysis (Phase P5) records
observations *about* responses; assessment of evidence comes later.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class ParticipantStatus(StrEnum):
    """Lifecycle of a configured participant (declared, not inferred)."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"  # configured but failed identity/capability check
    DISABLED = "disabled"


class AIMessageRole(StrEnum):
    SKYNET = "skynet"  # message authored by Skynet
    EXTERNAL = "external"  # message authored by the external AI
    SYSTEM = "system"  # conversation metadata (no AI authored it)


class AICapability(StrEnum):
    """Declared capabilities a participant may advertise.

    These are **claims by the provider configuration**, not verified facts —
    capability-based selection trusts the declaration, and future evaluation
    may test it.
    """

    REASONING = "reasoning"
    CODING = "coding"
    RESEARCH = "research"
    VISION = "vision"
    MATHEMATICS = "mathematics"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    PLANNING = "planning"


class AIMessage(BaseModel):
    """One message in an AI conversation.

    ``role`` records authorship; ``content`` is always **data** — an external
    message is never interpreted as an instruction to Skynet (see
    :mod:`agency.comms.security`).
    """

    id: str = Field(default_factory=_new_id)
    conversation_id: str
    sender: str  # "skynet" or the participant id
    recipient: str  # participant id or "skynet"
    role: AIMessageRole
    content: str
    seq: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=_utcnow)
    #: Provider-reported usage/cost metadata when available (never credentials).
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIParticipant(BaseModel):
    """One external AI system Skynet can talk to.

    Constructed from provider configuration; capabilities are declared, not
    assumed from names.
    """

    participant_id: str
    provider: str  # provider implementation name, e.g. "mock", "anthropic"
    model: str = ""
    endpoint: str = ""
    capabilities: frozenset[AICapability] = Field(default_factory=frozenset)
    status: ParticipantStatus = ParticipantStatus.AVAILABLE
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    LIMIT_REACHED = "limit_reached"  # a send was rejected: caller wanted more turns than budgeted


class AIConversation(BaseModel):
    """A bounded multi-turn interaction with one external participant."""

    id: str = Field(default_factory=_new_id)
    participant_id: str
    #: Owning Skynet run/goal — the provenance root for every observation
    #: derived from this conversation.
    run_id: str | None = None
    goal_id: str | None = None
    goal_title: str = ""
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: list[AIMessage] = Field(default_factory=list)
    max_turns: int = Field(default=5, ge=1)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def turns_used(self) -> int:
        """Number of Skynet→external exchanges so far."""
        return sum(1 for m in self.messages if m.role is AIMessageRole.SKYNET)

    @property
    def turns_remaining(self) -> int:
        return max(0, self.max_turns - self.turns_used)

    def active(self) -> bool:
        return self.status is ConversationStatus.ACTIVE


class ResponseAnalysis(BaseModel):
    """What Skynet recorded *about* one external response.

    Deliberately basic for Phase P5: instruction-like fragments are flagged,
    not obeyed; claims are extracted textually; no truth engine.
    """

    message_id: str
    #: Instruction-like fragments detected in the response (untrusted data).
    directive_candidates: list[str] = Field(default_factory=list)
    #: Simple declarative sentences extracted as candidate claims.
    claim_candidates: list[str] = Field(default_factory=list)
    notes: str = ""


class ResponseComparison(BaseModel):
    """Structured comparison of responses to one prompt across participants.

    Agreement is recorded, not treated as truth: shared training data means
    different AIs can repeat the same mistake (blueprint §10).
    """

    prompt: str
    participant_ids: list[str]
    #: participant_id → normalized response text (full provenance retained).
    responses: dict[str, str] = Field(default_factory=dict)
    #: participant_ids that returned something usable.
    responded: list[str] = Field(default_factory=list)
    #: participant_ids that failed, with reasons.
    failures: dict[str, str] = Field(default_factory=dict)
    #: Terms appearing in a substantial majority (>50%) of usable responses.
    agreement_terms: list[str] = Field(default_factory=list)
    #: Terms appearing in only some responses — candidate disagreement markers.
    divergence_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
