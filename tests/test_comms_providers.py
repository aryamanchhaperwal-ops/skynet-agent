"""Provider, registry, and capability-discovery tests (fully offline)."""

from __future__ import annotations

import pytest

from agency.comms.models import AICapability
from agency.comms.providers import (
    AIProviderRegistry,
    MockAIProvider,
    ProviderError,
    ProviderRequest,
    build_provider,
)

PARTICIPANT = "mock:mock-agent-1"


def make_request(**overrides):
    defaults = dict(
        conversation_id="conv-1",
        participant_id=PARTICIPANT,
        message="test question",
        timeout=5.0,
    )
    defaults.update(overrides)
    return ProviderRequest(**defaults)


class TestRegistry:
    def test_register_and_get(self):
        registry = AIProviderRegistry()
        provider = MockAIProvider()
        registry.register(provider)
        assert PARTICIPANT in registry
        assert registry.get(PARTICIPANT) is provider
        assert registry.get("mock") is None  # keyed by participant, not impl name
        assert len(registry) == 1

    def test_duplicate_registration_requires_override(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockAIProvider())  # same participant_id
        override = MockAIProvider(scenario="conflict")
        registry.register(override, override=True)
        assert registry.get(PARTICIPANT) is override

    def test_same_implementation_distinct_participants_coexist(self):
        """Several participants may share one provider implementation."""
        registry = AIProviderRegistry()
        registry.register(MockAIProvider(participant_id="mock:a"))
        registry.register(MockAIProvider(scenario="conflict", participant_id="mock:b"))
        registry.register(MockAIProvider(scenario="failure", participant_id="mock:c"))
        assert len(registry) == 3
        ids = [p.identify().participant_id for p in registry.list_providers()]
        assert ids == ["mock:a", "mock:b", "mock:c"]

    def test_unregister_and_list(self):
        registry = AIProviderRegistry()
        provider = MockAIProvider()
        registry.register(provider)
        assert registry.unregister(PARTICIPANT) is provider
        assert registry.unregister(PARTICIPANT) is None  # idempotent
        assert registry.list_providers() == []

    def test_find_capable_matches_declared_only(self):
        registry = AIProviderRegistry()
        registry.register(MockAIProvider())  # reasoning, research, summarization
        coder = MockAIProvider(
            capabilities=frozenset({AICapability.CODING}),
            participant_id="mock:coder",
        )
        registry.register(coder)
        capable = registry.find_capable(AICapability.RESEARCH)
        assert [p.identify().participant_id for p in capable] == [PARTICIPANT]
        assert registry.find_capable(AICapability.VISION) == []

    def test_capability_never_assumed_from_name(self):
        """A provider named 'genius' declares nothing unless configured to."""
        registry = AIProviderRegistry()
        empty = MockAIProvider(capabilities=frozenset(), participant_id="mock:none")
        registry.register(empty)
        assert registry.find_capable(AICapability.REASONING) == []


class TestMockProvider:
    async def test_normal_scenario_returns_structured_answer(self):
        provider = MockAIProvider()
        response = await provider.send(make_request())
        assert response.text
        assert response.status.value == "completed"
        assert "memory" in response.text.lower()

    async def test_multi_turn_scenario_references_history(self):
        provider = MockAIProvider(scenario="multi_turn")
        first = await provider.send(make_request())
        second = await provider.send(
            make_request(history=[("skynet", first.text)])
        )
        assert "1 question(s)" in first.text
        assert "2 question(s)" in second.text

    async def test_failure_scenario_reports_error(self):
        provider = MockAIProvider(scenario="failure")
        response = await provider.send(make_request())
        assert response.status.value == "failed"
        assert response.error
        assert not response.text

    async def test_timeout_scenario_reports_timeout(self):
        provider = MockAIProvider(scenario="timeout")
        response = await provider.send(make_request(timeout=0.05))
        assert response.status.value == "failed"
        assert "timeout" in (response.error or "")

    async def test_conflict_scenario_disagrees(self):
        provider = MockAIProvider()
        agree = await provider.send(make_request())
        conflict = MockAIProvider(scenario="conflict")
        disagree = await conflict.send(make_request())
        assert "flat files" in disagree.text
        assert disagree.text != agree.text

    async def test_malicious_scenario_returns_directives(self):
        provider = MockAIProvider(scenario="malicious")
        response = await provider.send(make_request())
        assert "ignore all previous instructions" in response.text
        assert "rm -rf" in response.text

    def test_identify_returns_participant(self):
        provider = MockAIProvider()
        participant = provider.identify()
        assert participant.participant_id == PARTICIPANT
        assert participant.provider == "mock"
        assert AICapability.RESEARCH in participant.capabilities

    async def test_records_requests(self):
        provider = MockAIProvider()
        await provider.send(make_request())
        assert len(provider.requests) == 1


class TestFactory:
    def test_mock_builds(self):
        assert isinstance(build_provider("mock"), MockAIProvider)

    def test_case_and_whitespace_tolerant(self):
        assert isinstance(build_provider(" Mock "), MockAIProvider)

    def test_unknown_name_raises_instead_of_inventing(self):
        with pytest.raises(ProviderError, match="unknown AI provider"):
            build_provider("does-not-exist")

    def test_real_provider_names_not_yet_available(self):
        with pytest.raises(ProviderError, match="intelligence phase"):
            build_provider("anthropic")
