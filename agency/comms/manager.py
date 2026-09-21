"""Conversation management for Skynet ↔ external AI interactions.

Turns Skynet questions into bounded, provenance-preserving conversations and
external responses into structured observations. Guarantees:

- **Turn budgets**: a conversation cannot exceed ``max_turns`` Skynet→AI
  exchanges; the budget is enforced here, not trusted to the provider.
- **Provenance**: every response keeps its participant, conversation, message,
  run and goal; observations carry the full chain in metadata.
- **Analysis boundary**: responses are analyzed (directives flagged, claims
  extracted) but never executed — see :mod:`agency.comms.security`.
- **One event stream**: all lifecycle events flow through the run's
  :class:`agency.trace.Trace` facade, sharing run_id and ordering.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agency.comms.models import (
    AIConversation,
    AIMessage,
    AIMessageRole,
    AIParticipant,
    ConversationStatus,
)
from agency.comms.providers import (
    AIProvider,
    AIProviderRegistry,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
)
from agency.comms.security import analyze_response, sanitize_metadata

logger = logging.getLogger("skynet.comms.manager")

#: Signature of the trace-emitting hook actions hand to the manager.
TraceEmitter = Callable[..., Awaitable[None]]


class ConversationTurn(BaseModel):
    """One Skynet→AI exchange within a conversation."""

    question: str
    answer: str
    message_ids: tuple[str, str]
    latency_ms: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)
    analysis_directives: list[str] = Field(default_factory=list)
    analysis_claims: list[str] = Field(default_factory=list)


class ConversationOutcome(BaseModel):
    """Structured result of a conversation, including provenance summary."""

    conversation: AIConversation
    turns: list[ConversationTurn] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    status: ConversationStatus = ConversationStatus.COMPLETED
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """A packaged outcome succeeded iff the conversation completed and
        produced observations (failures and rejected sends never package)."""
        return (
            self.status is ConversationStatus.COMPLETED and bool(self.observations)
        )


class ConversationManager:
    """Runs bounded conversations with external AI participants.

    The manager owns the conversation lifecycle (start → turns → close) and
    enforces every limit locally: turn budget, retries, request interval,
    and per-request timeout. Providers are never trusted to enforce them.
    """

    def __init__(
        self,
        registry: AIProviderRegistry,
        *,
        max_turns: int = 5,
        request_timeout: float = 30.0,
        max_retries: int = 1,
        request_interval: float = 1.0,
    ) -> None:
        self._registry = registry
        self._max_turns = max_turns
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._request_interval = request_interval
        self._last_request_at: dict[str, float] = {}

    # -- discovery ------------------------------------------------------------

    def participants(self) -> list[AIParticipant]:
        """Describe all registered participants (identity discovery)."""
        return [provider.identify() for provider in self._registry.list_providers()]

    def find_capable(self, capability: str) -> list[AIParticipant]:
        """Participants declaring ``capability`` (declared, not inferred)."""
        from agency.comms.models import AICapability

        try:
            cap = AICapability(capability)
        except ValueError:
            return []
        return [
            provider.identify()
            for provider in self._registry.find_capable(cap)
        ]

    # -- conversation lifecycle -----------------------------------------------

    async def start_conversation(
        self,
        participant_id: str,
        *,
        run_id: str | None = None,
        goal_id: str | None = None,
        goal_title: str = "",
        max_turns: int | None = None,
        emit: TraceEmitter | None = None,
    ) -> AIConversation:
        """Open a conversation with the named participant."""
        participant = self._require_participant(participant_id)
        conversation = AIConversation(
            participant_id=participant.participant_id,
            run_id=run_id,
            goal_id=goal_id,
            goal_title=goal_title,
            max_turns=max_turns if max_turns is not None else self._max_turns,
            metadata={"provider": participant.provider, "model": participant.model},
        )
        if emit is not None:
            await emit(
                "AI_CONVERSATION_STARTED",
                conversation_id=conversation.id,
                participant_id=participant.participant_id,
                max_turns=conversation.max_turns,
            )
        return conversation

    async def send_message(
        self,
        conversation: AIConversation,
        text: str,
        *,
        emit: TraceEmitter | None = None,
    ) -> ConversationTurn:
        """Send one Skynet message and collect the participant's response.

        Enforces: active status, turn budget, request interval, retries and
        timeout. Raises :class:`ProviderError` only when the conversation is
        in an unrecoverable state; provider failures close the conversation
        as FAILED and raise after recording the failure.
        """
        if not conversation.active():
            raise ProviderError(
                f"conversation {conversation.id} is {conversation.status.value}, not active"
            )
        if conversation.turns_remaining <= 0:
            # A *rejected* send: the caller wanted more turns than budgeted.
            # The conversation closes as LIMIT_REACHED — distinct from a clean
            # completion, and valuable signal for the future learning loop.
            conversation.status = ConversationStatus.LIMIT_REACHED
            conversation.error = f"turn budget exhausted ({conversation.max_turns} turns)"
            await self._close(conversation, emit=emit)
            raise ProviderError(
                f"conversation {conversation.id} exhausted its turn budget "
                f"({conversation.max_turns})"
            )
        participant = self._require_participant(conversation.participant_id)
        provider = self._registry.get(participant.participant_id)
        assert provider is not None  # registry guarantees participant presence

        seq = len(conversation.messages) + 1
        outbound = AIMessage(
            conversation_id=conversation.id,
            sender="skynet",
            recipient=participant.participant_id,
            role=AIMessageRole.SKYNET,
            content=text,
            seq=seq,
        )
        conversation.messages.append(outbound)
        if emit is not None:
            await emit(
                "AI_MESSAGE_SENT",
                conversation_id=conversation.id,
                participant_id=participant.participant_id,
                message_id=outbound.id,
                seq=outbound.seq,
            )

        history = [(m.sender, m.content) for m in conversation.messages[:-1]]
        response = await self._send_with_retry(
            provider,
            ProviderRequest(
                conversation_id=conversation.id,
                participant_id=participant.participant_id,
                message=text,
                history=history,
                timeout=self._request_timeout,
            ),
        )

        inbound_seq = len(conversation.messages) + 1
        if response.status is ConversationStatus.FAILED or response.error:
            conversation.status = ConversationStatus.FAILED
            conversation.error = response.error
            conversation.ended_at = _now()
            if emit is not None:
                await emit(
                    "AI_CONVERSATION_FAILED",
                    conversation_id=conversation.id,
                    participant_id=participant.participant_id,
                    error=response.error or "provider failure",
                )
            raise ProviderError(
                f"provider failure in conversation {conversation.id}: {response.error}"
            )

        inbound = AIMessage(
            conversation_id=conversation.id,
            sender=participant.participant_id,
            recipient="skynet",
            role=AIMessageRole.EXTERNAL,
            content=response.text,
            seq=inbound_seq,
            usage=sanitize_metadata(response.usage),
            metadata={"model": response.model, "finish_reason": response.finish_reason},
        )
        conversation.messages.append(inbound)
        if emit is not None:
            await emit(
                "AI_RESPONSE_RECEIVED",
                conversation_id=conversation.id,
                participant_id=participant.participant_id,
                message_id=inbound.id,
                latency_ms=response.latency_ms,
            )

        analysis = analyze_response(inbound.id, response.text)
        if emit is not None:
            await emit(
                "AI_RESPONSE_ANALYZED",
                conversation_id=conversation.id,
                message_id=inbound.id,
                directive_count=len(analysis.directive_candidates),
                claim_count=len(analysis.claim_candidates),
            )

        # Using the full budget is *not* auto-closure: a conversation that
        # consumed its planned turns cleanly is completed by its owner
        # (``research_session`` always closes). Only a rejected send marks
        # LIMIT_REACHED.
        if conversation.turns_remaining > 0 and emit is not None:
            await emit(
                "AI_CONVERSATION_CONTINUED",
                conversation_id=conversation.id,
                participant_id=participant.participant_id,
                turns_remaining=conversation.turns_remaining,
            )

        return ConversationTurn(
            question=text,
            answer=response.text,
            message_ids=(outbound.id, inbound.id),
            latency_ms=response.latency_ms,
            usage=dict(inbound.usage),
            analysis_directives=analysis.directive_candidates,
            analysis_claims=analysis.claim_candidates,
        )

    async def end_conversation(
        self,
        conversation: AIConversation,
        *,
        emit: TraceEmitter | None = None,
        error: str | None = None,
    ) -> AIConversation:
        """Close a conversation cleanly (idempotent; one terminal event)."""
        if error and not conversation.error:
            conversation.error = error
        await self._close(conversation, emit=emit)
        return conversation

    async def _close(self, conversation: AIConversation, *, emit: TraceEmitter | None) -> None:
        """Settle status/ended_at and emit exactly one terminal event.

        ``ended_at`` marks closure: the first ``_close`` call on a conversation
        emits its terminal event (COMPLETED, or FAILED if already marked), and
        every later call is a silent no-op. Statuses already terminal from
        elsewhere (failure in :meth:`send_message`, turn-budget exhaustion)
        still get their event on first close.
        """
        first_close = conversation.ended_at is None
        if conversation.ended_at is None:
            conversation.ended_at = _now()
        if conversation.status is ConversationStatus.ACTIVE:
            conversation.status = ConversationStatus.COMPLETED
        if emit is None or not first_close:
            return
        event = (
            "AI_CONVERSATION_FAILED"
            if conversation.status
            in {ConversationStatus.FAILED, ConversationStatus.LIMIT_REACHED}
            else "AI_CONVERSATION_COMPLETED"
        )
        payload: dict[str, Any] = {
            "conversation_id": conversation.id,
            "participant_id": conversation.participant_id,
            "turns_used": conversation.turns_used,
            "status": conversation.status.value,
        }
        if conversation.error:
            payload["error"] = conversation.error
        await emit(event, **payload)

    # -- research session -------------------------------------------------------

    async def research_session(
        self,
        participant_id: str,
        question: str,
        *,
        follow_ups: list[str] | None = None,
        run_id: str | None = None,
        goal_id: str | None = None,
        goal_title: str = "",
        emit: TraceEmitter | None = None,
    ) -> ConversationOutcome:
        """Ask one research question (plus optional follow-ups) and package
        the conversation, turns and observations with full provenance."""
        conversation = await self.start_conversation(
            participant_id,
            run_id=run_id,
            goal_id=goal_id,
            goal_title=goal_title,
            max_turns=1 + len(follow_ups or []),
            emit=emit,
        )
        outcome = ConversationOutcome(conversation=conversation)
        turns: list[ConversationTurn] = []
        try:
            turns.append(
                await self.send_message(conversation, question, emit=emit)
            )
            for follow_up in follow_ups or []:
                turns.append(
                    await self.send_message(conversation, follow_up, emit=emit)
                )
        except ProviderError as exc:
            outcome.status = ConversationStatus.FAILED
            outcome.error = str(exc)
            await self.end_conversation(conversation, emit=emit, error=str(exc))
            outcome.conversation = conversation
            outcome.turns = turns
            return outcome
        outcome.turns = turns
        outcome.observations = observations_from_conversation(
            conversation,
            turns,
            run_id=run_id,
            goal_id=goal_id,
        )
        await self.end_conversation(conversation, emit=emit)
        outcome.status = conversation.status
        outcome.conversation = conversation
        return outcome

    # -- internals ----------------------------------------------------------------

    async def _send_with_retry(
        self, provider: AIProvider, request: ProviderRequest
    ) -> ProviderResponse:
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            await self._respect_interval(provider.name)
            try:
                return await provider.send(request)
            except ProviderError:
                raise  # provider-signaled failure is final, not transient
            except (TimeoutError, ConnectionError) as exc:
                last_error = exc
                logger.warning(
                    "transient provider failure (%s/%s) for %s: %s",
                    attempt + 1,
                    attempts,
                    provider.name,
                    exc,
                )
        return ProviderResponse(
            text="",
            status=ConversationStatus.FAILED,
            error=f"provider {provider.name} failed after {attempts} attempt(s): {last_error}",
        )

    async def _respect_interval(self, provider_name: str) -> None:
        if self._request_interval <= 0:
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        last = self._last_request_at.get(provider_name)
        if last is not None:
            elapsed = now - last
            if elapsed < self._request_interval:
                await asyncio.sleep(self._request_interval - elapsed)
        self._last_request_at[provider_name] = loop.time()

    def _require_participant(self, participant_id: str) -> AIParticipant:
        provider = self._registry.get(participant_id)
        if provider is None:
            raise ProviderError(f"unknown participant {participant_id!r}")
        return provider.identify()


def _now() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)


def observations_from_conversation(
    conversation: AIConversation,
    turns: list[ConversationTurn],
    *,
    run_id: str | None = None,
    goal_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert conversation turns into Skynet observation payloads.

    Each observation preserves the full provenance chain:
    goal → conversation → participant → message → response.
    """
    observations: list[dict[str, Any]] = []
    participant_id = conversation.participant_id
    for turn in turns:
        external = next(
            (m for m in conversation.messages if m.id == turn.message_ids[1]), None
        )
        if external is None:  # pragma: no cover - internal invariant
            continue
        observations.append(
            {
                "source": f"ai:{participant_id}",
                "kind": "text",
                "content": external.content,
                "summary": f"Response from {participant_id} "
                f"(conversation {conversation.id[:8]}, turn {external.seq // 2 or 1})",
                "confidence": None,
                "metadata": {
                    "provenance": {
                        "goal_id": conversation.goal_id or goal_id,
                        "goal_title": conversation.goal_title,
                        "run_id": conversation.run_id or run_id,
                        "conversation_id": conversation.id,
                        "participant_id": participant_id,
                        "message_id": external.id,
                        "role": external.role.value,
                        "timestamp": external.timestamp.isoformat(),
                    },
                    "model": external.metadata.get("model", ""),
                    "analysis": {
                        "directive_candidates": turn.analysis_directives,
                        "claim_candidates": turn.analysis_claims,
                    },
                },
            }
        )
    return observations


__all__ = [
    "ConversationManager",
    "ConversationOutcome",
    "ConversationTurn",
    "observations_from_conversation",
]
