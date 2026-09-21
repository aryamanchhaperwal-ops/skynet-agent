"""Conversation manager tests: lifecycle, budgets, retries, provenance."""

from __future__ import annotations

import pytest

from agency.comms.manager import ConversationManager, observations_from_conversation
from agency.comms.models import ConversationStatus
from agency.comms.providers import (
    AIProviderRegistry,
    MockAIProvider,
    ProviderError,
)


def make_manager(
    scenario: str = "normal",
    *,
    max_turns: int = 5,
    max_retries: int = 1,
    interval: float = 0.0,
) -> ConversationManager:
    registry = AIProviderRegistry()
    registry.register(MockAIProvider(scenario=scenario))
    return ConversationManager(
        registry,
        max_turns=max_turns,
        request_timeout=5.0,
        max_retries=max_retries,
        request_interval=interval,
    )


class MemorySink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event: str, **payload) -> None:
        self.events.append((event, payload))

    def types(self) -> list[str]:
        return [name for name, _ in self.events]


class TestLifecycle:
    async def test_start_and_single_turn(self):
        manager = make_manager()
        sink = MemorySink()
        conversation = await manager.start_conversation(
            "mock:mock-agent-1", goal_title="g", emit=sink
        )
        turn = await manager.send_message(conversation, "What is memory?", emit=sink)
        assert conversation.status is ConversationStatus.ACTIVE
        assert turn.answer
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role.value == "skynet"
        assert conversation.messages[1].role.value == "external"
        assert "AI_CONVERSATION_STARTED" in sink.types()
        assert "AI_MESSAGE_SENT" in sink.types()
        assert "AI_RESPONSE_RECEIVED" in sink.types()

    async def test_end_conversation_is_idempotent(self):
        manager = make_manager()
        sink = MemorySink()
        conversation = await manager.start_conversation("mock:mock-agent-1", emit=sink)
        await manager.send_message(conversation, "q", emit=sink)
        await manager.end_conversation(conversation, emit=sink)
        await manager.end_conversation(conversation, emit=sink)
        assert conversation.status is ConversationStatus.COMPLETED
        assert conversation.ended_at is not None
        assert sink.types().count("AI_CONVERSATION_COMPLETED") == 1

    async def test_multi_turn_records_sequence(self):
        manager = make_manager(scenario="multi_turn")
        conversation = await manager.start_conversation(
            "mock:mock-agent-1", max_turns=3
        )
        await manager.send_message(conversation, "one")
        await manager.send_message(conversation, "two")
        seqs = [m.seq for m in conversation.messages]
        assert seqs == [1, 2, 3, 4]

    async def test_unknown_participant_rejected(self):
        manager = make_manager()
        with pytest.raises(ProviderError, match="unknown participant"):
            await manager.start_conversation("nobody:here")


class TestBudgets:
    async def test_full_budget_use_completes_cleanly(self):
        """Using every planned turn is completion, not a limit violation."""
        manager = make_manager(max_turns=2)
        conversation = await manager.start_conversation("mock:mock-agent-1")
        await manager.send_message(conversation, "one")
        await manager.send_message(conversation, "two")
        assert conversation.status is ConversationStatus.ACTIVE  # open until owner closes
        await manager.end_conversation(conversation)
        assert conversation.status is ConversationStatus.COMPLETED

    async def test_rejected_send_marks_limit_reached(self):
        """A send beyond the budget is rejected and closes as LIMIT_REACHED."""
        manager = make_manager(max_turns=2)
        sink = MemorySink()
        conversation = await manager.start_conversation("mock:mock-agent-1", emit=sink)
        await manager.send_message(conversation, "one", emit=sink)
        await manager.send_message(conversation, "two", emit=sink)
        with pytest.raises(ProviderError, match="turn budget"):
            await manager.send_message(conversation, "three", emit=sink)
        assert conversation.status is ConversationStatus.LIMIT_REACHED
        assert conversation.error is not None and "turn budget" in conversation.error
        assert "AI_CONVERSATION_FAILED" in sink.types()  # budget rejection closes it

    async def test_closed_conversation_rejects_messages(self):
        manager = make_manager()
        conversation = await manager.start_conversation("mock:mock-agent-1")
        await manager.end_conversation(conversation)
        with pytest.raises(ProviderError, match="not active"):
            await manager.send_message(conversation, "late")

    async def test_provider_failure_closes_conversation(self):
        manager = make_manager(scenario="failure")
        sink = MemorySink()
        conversation = await manager.start_conversation("mock:mock-agent-1", emit=sink)
        with pytest.raises(ProviderError):
            await manager.send_message(conversation, "q", emit=sink)
        assert conversation.status is ConversationStatus.FAILED
        assert "AI_CONVERSATION_FAILED" in sink.types()

    async def test_timeout_scenario_closes_conversation(self):
        manager = make_manager(scenario="timeout")
        conversation = await manager.start_conversation("mock:mock-agent-1")
        with pytest.raises(ProviderError):
            await manager.send_message(conversation, "q")
        assert conversation.status is ConversationStatus.FAILED


class TestResearchSession:
    async def test_full_session_with_follow_ups(self):
        manager = make_manager(scenario="multi_turn")
        sink = MemorySink()
        outcome = await manager.research_session(
            "mock:mock-agent-1",
            "main question",
            follow_ups=["follow one", "follow two"],
            run_id="run-1",
            goal_id="goal-1",
            emit=sink,
        )
        assert outcome.succeeded
        assert len(outcome.turns) == 3
        assert len(outcome.observations) == 3
        assert outcome.status is ConversationStatus.COMPLETED  # full budget used cleanly
        assert "AI_RESPONSE_ANALYZED" in sink.types()

    async def test_failed_session_preserves_provenance_and_error(self):
        manager = make_manager(scenario="failure")
        outcome = await manager.research_session("mock:mock-agent-1", "q")
        assert not outcome.succeeded
        assert outcome.status is ConversationStatus.FAILED
        assert outcome.error
        assert outcome.conversation.error


class TestObservations:
    async def test_observations_carry_full_provenance(self):
        manager = make_manager()
        outcome = await manager.research_session(
            "mock:mock-agent-1",
            "main question",
            follow_ups=["follow"],
            run_id="run-abc",
            goal_id="goal-xyz",
        )
        for observation in outcome.observations:
            provenance = observation["metadata"]["provenance"]
            assert provenance["run_id"] == "run-abc"
            assert provenance["goal_id"] == "goal-xyz"
            assert provenance["conversation_id"] == outcome.conversation.id
            assert provenance["participant_id"] == "mock:mock-agent-1"
            assert provenance["message_id"]
            assert provenance["role"] == "external"
            assert provenance["timestamp"]
        # first provenance answer to the main question
        assert outcome.observations[0]["source"] == "ai:mock:mock-agent-1"

    async def test_malicious_response_flagged_not_executed(self):
        manager = make_manager(scenario="malicious")
        outcome = await manager.research_session("mock:mock-agent-1", "analyze this")
        assert outcome.observations, "response still recorded as data"
        analysis = outcome.observations[0]["metadata"]["analysis"]
        assert analysis["directive_candidates"], "directives flagged for humans"
        joined = " ".join(outcome.observations[0]["metadata"]["analysis"]["directive_candidates"]).lower()
        assert "ignore all previous instructions" in joined
        # The observation content is inert data; nothing executed anything.
        assert isinstance(outcome.observations[0]["content"], str)

    async def test_observations_from_conversation_standalone(self):
        manager = make_manager()
        outcome = await manager.research_session("mock:mock-agent-1", "q")
        observations = observations_from_conversation(
            outcome.conversation, outcome.turns, run_id="r", goal_id="g"
        )
        assert len(observations) == 1
