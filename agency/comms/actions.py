"""AI-communication actions: the only way the Skynet core talks to other AIs.

Three actions registered under category ``comms`` (gated by
``SKYNET_ENABLE_EXTERNAL_COMMS``):

- ``ai_list_participants`` — discovery + declared-capability query
- ``ai_ask``               — bounded research session with one participant
- ``ai_compare``           — same prompt to several participants, compared

All of them emit ``AI_*`` lifecycle events through the run's tracer, convert
responses into observations with full provenance, and treat every external
response as **data, never commands** (enforced in :mod:`agency.comms.security`;
nothing in an action consumes response text as an action spec).
"""

from __future__ import annotations

import logging
from typing import Any

from agency.actions.base import Action, ActionContext, ActionError, ActionSpec
from agency.comms.compare import ComparisonService
from agency.comms.manager import ConversationManager
from agency.comms.providers import ProviderError

logger = logging.getLogger("skynet.comms.actions")


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"missing or empty string parameter {key!r}")
    return value.strip()


def _optional_str_list(params: dict[str, Any], key: str) -> list[str] | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ActionError(f"parameter {key!r} must be a list of strings")
    return value


def _optional_int(params: dict[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ActionError(f"parameter {key!r} must be an integer")
    return value


class AIListParticipantsAction(Action):
    """Discover external AI participants and their declared capabilities."""

    name = "ai_list_participants"
    category = "comms"
    description = "List external AI participants and declared capabilities."

    def __init__(self, manager: ConversationManager) -> None:
        self._manager = manager

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        capability = spec.params.get("capability")
        participants = (
            self._manager.find_capable(str(capability))
            if isinstance(capability, str) and capability.strip()
            else self._manager.participants()
        )
        for participant in participants:
            await ctx.emit_trace(
                "AI_PARTICIPANT_DISCOVERED",
                participant_id=participant.participant_id,
                provider=participant.provider,
                capabilities=sorted(c.value for c in participant.capabilities),
            )
        return {
            "ok": True,
            "participants": [p.model_dump(mode="json") for p in participants],
        }


class AIAskAction(Action):
    """Bounded research session with one external AI participant."""

    name = "ai_ask"
    category = "comms"
    description = (
        "Ask one external AI a question (with optional follow-ups); returns "
        "the full conversation plus provenance-carrying observations."
    )

    def __init__(self, manager: ConversationManager) -> None:
        self._manager = manager

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        participant_id = _require_str(spec.params, "participant_id")
        question = _require_str(spec.params, "question")
        follow_ups = _optional_str_list(spec.params, "follow_ups") or []

        async def emit(event: str, **payload: Any) -> None:
            await ctx.emit_trace(event, **payload)

        try:
            outcome = await self._manager.research_session(
                participant_id,
                question,
                follow_ups=follow_ups,
                run_id=ctx.run_id,
                goal_id=ctx.goal_id,
                emit=emit,
            )
        except ProviderError as exc:
            await ctx.emit_trace(
                "AI_CONVERSATION_FAILED", participant_id=participant_id, error=str(exc)
            )
            raise ActionError(f"ai_ask failed: {exc}") from exc
        if not outcome.succeeded:
            raise ActionError(f"ai_ask produced no usable response: {outcome.error}")
        return {
            "ok": True,
            "conversation": outcome.conversation.model_dump(mode="json"),
            "turns": [turn.model_dump(mode="json") for turn in outcome.turns],
            "status": outcome.status.value,
            # The loop absorbs these into agent state; provenance lives in
            # each observation's metadata.
            "skynet_observations": outcome.observations,
        }


class AICompareAction(Action):
    """Ask several participants the same prompt and compare responses."""

    name = "ai_compare"
    category = "comms"
    description = (
        "Ask the same question to multiple external AIs; record agreement and "
        "divergence. Agreement is data, not truth."
    )

    def __init__(self, manager: ConversationManager, comparison: ComparisonService) -> None:
        self._manager = manager
        self._comparison = comparison

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        prompt = _require_str(spec.params, "prompt")
        participant_ids = _optional_str_list(spec.params, "participant_ids")
        if not participant_ids:
            available = [p.participant_id for p in self._manager.participants()]
            if not available:
                raise ActionError("no AI participants are registered")
            participant_ids = available

        async def emit(event: str, **payload: Any) -> None:
            await ctx.emit_trace(event, **payload)

        comparison = await self._comparison.compare(
            prompt,
            participant_ids,
            run_id=ctx.run_id,
            goal_id=ctx.goal_id,
            emit=emit,
        )
        await ctx.emit_trace(
            "AI_RESPONSE_ANALYZED",
            mode="comparison",
            responded=len(comparison.responded),
            failures=len(comparison.failures),
            agreement_terms=len(comparison.agreement_terms),
            divergence_terms=len(comparison.divergence_terms),
        )
        # Comparison responses also become observations (data with provenance).
        observations = [
            {
                "source": f"ai:{participant_id}",
                "kind": "text",
                "content": text,
                "summary": f"Comparison response from {participant_id} "
                f"({len(comparison.responded)} usable of "
                f"{len(comparison.participant_ids)} asked)",
                "metadata": {
                    "provenance": {
                        "run_id": ctx.run_id,
                        "goal_id": ctx.goal_id,
                        "comparison_prompt": prompt,
                        "participant_id": participant_id,
                    },
                    "comparison": {
                        "agreement_terms": comparison.agreement_terms[:20],
                        "divergence_terms": comparison.divergence_terms[:20],
                    },
                },
            }
            for participant_id, text in comparison.responses.items()
        ]
        return {
            "ok": True,
            "comparison": comparison.model_dump(mode="json"),
            "skynet_observations": observations,
        }


__all__ = [
    "AIAskAction",
    "AICompareAction",
    "AIListParticipantsAction",
]
