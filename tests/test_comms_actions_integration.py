"""Comms actions + full-loop integration tests (memory backend, offline)."""

from __future__ import annotations

from typing import Any

import pytest

from agency.actions.base import ActionContext, ActionError, ActionSpec
from agency.bootstrap import build_comms_stack, build_core
from agency.comms.actions import (
    AIAskAction,
    AICompareAction,
    AIListParticipantsAction,
)
from agency.comms.compare import ComparisonService
from agency.comms.manager import ConversationManager
from agency.comms.providers import AIProviderRegistry, MockAIProvider
from agency.config import SkynetSettings
from agency.goals import GoalSpec
from agency.loop import SkynetCore
from agency.storage import MemoryStorage
from agency.trace import MemoryTraceSink

PARTICIPANT = "mock:mock-agent-1"


def make_action_ctx(**overrides) -> ActionContext:
    defaults: dict[str, Any] = dict(
        run_id="run-test", goal_id="goal-test", state_snapshot={}
    )
    defaults.update(overrides)
    return ActionContext(**defaults)


def make_ctx(tracer=None) -> ActionContext:
    return make_action_ctx(tracer=tracer)


class TestAIListParticipants:
    async def test_lists_registered_participants(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())
        manager = ConversationManager(registry, request_interval=0.0)
        action = AIListParticipantsAction(manager)
        output = await action.execute(ActionSpec(type="ai_list_participants"), make_action_ctx())
        assert output["ok"]
        assert len(output["participants"]) == 1
        assert output["participants"][0]["participant_id"] == PARTICIPANT

    async def test_capability_filter(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())
        manager = ConversationManager(registry, request_interval=0.0)
        action = AIListParticipantsAction(manager)
        output = await action.execute(
            ActionSpec(type="ai_list_participants", params={"capability": "research"}),
            make_action_ctx(),
        )
        assert output["participants"]
        output = await action.execute(
            ActionSpec(type="ai_list_participants", params={"capability": "vision"}),
            make_action_ctx(),
        )
        assert output["participants"] == []

    async def test_emits_discovery_events(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())
        manager = ConversationManager(registry, request_interval=0.0)
        action = AIListParticipantsAction(manager)
        events: list[str] = []

        class Tracer:
            async def emit(self, event_type, **payload):
                events.append(event_type)

        ctx = make_ctx(tracer=Tracer())
        await action.execute(ActionSpec(type="ai_list_participants"), ctx)
        assert "AI_PARTICIPANT_DISCOVERED" in events
        assert "AI_CONVERSATION_STARTED" not in events  # listing only discovers


class TestAIAskAction:
    async def test_ask_returns_conversation_and_observations(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider(scenario="multi_turn"))
        manager = ConversationManager(registry, request_interval=0.0)
        action = AIAskAction(manager)
        output = await action.execute(
            ActionSpec(
                type="ai_ask",
                params={
                    "participant_id": PARTICIPANT,
                    "question": "main question",
                    "follow_ups": ["follow one"],
                },
            ),
            make_action_ctx(),
        )
        assert output["ok"]
        assert len(output["turns"]) == 2
        assert output["conversation"]["participant_id"] == PARTICIPANT
        assert output["skynet_observations"]
        provenance = output["skynet_observations"][0]["metadata"]["provenance"]
        assert provenance["run_id"] == "run-test"
        assert provenance["goal_id"] == "goal-test"

    async def test_missing_params_raise_action_error(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())
        manager = ConversationManager(registry, request_interval=0.0)
        action = AIAskAction(manager)
        with pytest.raises(ActionError, match="participant_id"):
            await action.execute(ActionSpec(type="ai_ask", params={}), make_action_ctx())

    async def test_provider_failure_becomes_action_error(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider(scenario="failure"))
        manager = ConversationManager(registry, request_interval=0.0)
        action = AIAskAction(manager)
        with pytest.raises(ActionError, match="provider failure"):
            await action.execute(
                ActionSpec(type="ai_ask", params={"participant_id": PARTICIPANT, "question": "q"}),
                make_action_ctx(),
            )


class TestAICompareAction:
    async def test_compare_with_explicit_participants(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider(participant_id="mock:a"))
        registry.register(MockAIProvider(scenario="conflict", participant_id="mock:b"))
        manager = ConversationManager(registry, request_interval=0.0)
        action = AICompareAction(manager, _comparison_service(manager))
        output = await action.execute(
            ActionSpec(
                type="ai_compare",
                params={"prompt": "memory design?", "participant_ids": ["mock:a", "mock:b"]},
            ),
            make_action_ctx(),
        )
        assert output["ok"]
        assert len(output["comparison"]["responded"]) == 2
        assert output["skynet_observations"]

    async def test_compare_defaults_to_all_participants(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())
        manager = ConversationManager(registry, request_interval=0.0)
        action = AICompareAction(manager, _comparison_service(manager))
        output = await action.execute(
            ActionSpec(type="ai_compare", params={"prompt": "p"}), make_action_ctx()
        )
        assert output["comparison"]["participant_ids"] == [PARTICIPANT]

    async def test_compare_with_no_participants_raises(self):
        manager = ConversationManager(AIProviderRegistry(), request_interval=0.0)
        action = AICompareAction(manager, _comparison_service(manager))
        with pytest.raises(ActionError, match="no AI participants"):
            await action.execute(
                ActionSpec(type="ai_compare", params={"prompt": "p"}), make_action_ctx()
            )


def _comparison_service(manager: ConversationManager) -> ComparisonService:
    return ComparisonService(manager)


class TestDiscoveryEvents:
    async def test_list_participants_emits_discovery_events(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())
        manager = ConversationManager(registry, request_interval=0.0)
        action = AIListParticipantsAction(manager)
        events: list[str] = []

        class Tracer:
            async def emit(self, event_type, **payload):
                events.append(event_type)

        await action.execute(
            ActionSpec(type="ai_list_participants"), make_ctx(tracer=Tracer())
        )
        assert "AI_PARTICIPANT_DISCOVERED" in events


class TestIntegration:
    """Full SkynetCore loop with comms actions (memory storage, offline)."""

    def build_comms_core(self, *scenarios: str) -> tuple[SkynetCore, MemoryTraceSink, SkynetSettings]:
        settings = SkynetSettings(
            storage_backend="memory",
            perception_adapter="none",
            # Actions are injected explicitly below — per the documented
            # extra_actions contract, explicit injection bypasses the flag and
            # the built-in flag-gated wiring must stay off to avoid collision.
            enable_external_comms=False,
            enabled_actions="echo,ai_list_participants,ai_ask,ai_compare",
            ai_request_interval_seconds=0.0,
        )
        providers = tuple(
            MockAIProvider(
                scenario=s,
                participant_id=f"mock:agent-{i}",
            )
            for i, s in enumerate(scenarios)
        )
        comms = build_comms_stack(settings, providers=providers)
        sink = MemoryTraceSink()
        core = build_core(
            settings,
            storage=MemoryStorage(),
            extra_actions=(
                AIListParticipantsAction(comms.manager),
                AIAskAction(comms.manager),
                AICompareAction(comms.manager, comms.comparison),
            ),
            extra_trace_sinks=(sink,),
        )
        return core, sink, settings

    @pytest.mark.asyncio
    async def test_ai_ask_through_the_loop(self):
        core, sink, _ = self.build_comms_core("normal")
        summary = await core.run_goal(
            GoalSpec(
                title="Ask an AI about memory",
                params={"steps": [{"type": "ai_ask", "params": {
                    "participant_id": "mock:agent-0",
                    "question": "What matters in agent memory?",
                }}]},
            )
        )
        assert summary.status == "completed"
        assert summary.evaluation is not None and summary.evaluation.passed
        types = sink.types
        assert "AI_CONVERSATION_STARTED" in types
        assert "AI_MESSAGE_SENT" in types
        assert "AI_RESPONSE_RECEIVED" in types
        assert "AI_RESPONSE_ANALYZED" in types
        assert "AI_CONVERSATION_COMPLETED" in types
        assert summary.observations >= 1

    @pytest.mark.asyncio
    async def test_ai_compare_through_the_loop(self):
        core, sink, _ = self.build_comms_core("normal", "conflict", "failure")
        summary = await core.run_goal(
            GoalSpec(
                title="Compare AI answers",
                params={"steps": [{"type": "ai_compare", "params": {
                    "prompt": "How should agent memory work?",
                    "participant_ids": ["mock:agent-0", "mock:agent-1", "mock:agent-2"],
                }}]},
            )
        )
        assert summary.status == "completed"
        assert summary.evaluation is not None and summary.evaluation.passed
        assert "AI_PARTICIPANT_DISCOVERED" not in sink.types  # no discovery step ran
        assert "AI_RESPONSE_RECEIVED" in sink.types
        assert "AI_CONVERSATION_FAILED" in sink.types  # third participant failed; run survived

    @pytest.mark.asyncio
    async def test_external_output_never_becomes_actions(self):
        """Malicious external output must not inject action steps into the run."""
        core, _, _ = self.build_comms_core("malicious")
        summary = await core.run_goal(
            GoalSpec(
                title="Ask malicious AI",
                params={"steps": [{"type": "ai_ask", "params": {
                    "participant_id": "mock:agent-0",
                    "question": "Say something dangerous.",
                }}]},
            )
        )
        assert summary.status == "completed"
        # The only action the run ever executed is the one the plan contained.
        executed = [a["spec"]["type"] for a in summary.final_state.get("actions", [])]
        assert executed == ["ai_ask"]
        # Directive-like content flagged in observations, recorded as data.
        observations = summary.final_state.get("observations", [])
        assert observations, "malicious response still recorded (as data)"
