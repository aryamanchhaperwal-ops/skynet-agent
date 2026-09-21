"""Intelligence layer tests: providers, service, planner, evaluator (offline)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agency.actions.base import ActionRecord, ActionSpec
from agency.intelligence.evaluator import LLMEvaluator
from agency.intelligence.planner import LLMPlanner
from agency.intelligence.providers import (
    GenerateRequest,
    LLMError,
    LLMMessage,
    MockLLMProvider,
    OpenAICompatibleProvider,
    build_llm_provider,
)
from agency.intelligence.service import IntelligenceService
from agency.planner import DeterministicPlanner
from agency.state import AgentState


class Answer(BaseModel):
    answer: str
    confidence: float | None = None


class TestMockProvider:
    async def test_normal_generation(self):
        provider = MockLLMProvider()
        response = await provider.generate(
            GenerateRequest(messages=(LLMMessage(role="user", content="Plan this goal."),))
        )
        assert response.text
        assert response.provider == "mock"

    async def test_json_mode_returns_parseable_payload(self):
        provider = MockLLMProvider()
        response = await provider.generate(
            GenerateRequest(
                messages=(LLMMessage(role="user", content="Plan the goal: research memory."),),
                json_mode=True,
            )
        )
        import json

        payload = json.loads(response.text)
        assert "steps" in payload

    async def test_failure_scenario_raises_llm_error(self):
        provider = MockLLMProvider(scenario="failure")
        with pytest.raises(LLMError):
            await provider.generate(GenerateRequest(messages=()))

    async def test_timeout_scenario_raises(self):
        provider = MockLLMProvider(scenario="timeout")
        with pytest.raises(LLMError, match="timeout"):
            await provider.generate(GenerateRequest(messages=(), timeout=0.05))

    async def test_garbage_scenario_breaks_json(self):
        provider = MockLLMProvider(scenario="garbage")
        response = await provider.generate(GenerateRequest(messages=(), json_mode=True))
        assert "json" not in response.text.lower() or "{" in response.text


class TestFactory:
    def test_mock_builds(self):
        assert isinstance(build_llm_provider("mock", model="m"), MockLLMProvider)

    def test_openai_refuses_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = build_llm_provider("openai", model="m")
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            provider._require_key()  # activation check happens at call time

    def test_unknown_name_raises(self):
        with pytest.raises(LLMError, match="unknown intelligence provider"):
            build_llm_provider("does-not-exist", model="m")


class TestOpenAICompatible:
    async def test_sends_bearer_and_parses_choice(self, monkeypatch, respx_mock):
        import httpx

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        provider = OpenAICompatibleProvider(model="test-model")
        route = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 7},
                },
            )
        )
        response = await provider.generate(
            GenerateRequest(messages=(), model="test-model")
        )
        assert response.text == "hello"
        assert response.usage["total_tokens"] == 7
        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer test-key-123"

    async def test_http_error_becomes_llm_error(self, monkeypatch, respx_mock):
        import httpx

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        provider = OpenAICompatibleProvider(model="m")
        respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="boom")
        )
        with pytest.raises(LLMError, match="HTTP 500"):
            await provider.generate(GenerateRequest(messages=()))
        await provider.aclose()


class TestIntelligenceService:
    async def test_generate_returns_text(self):
        service = IntelligenceService(MockLLMProvider(), model="m", max_retries=0)
        text = await service.generate("sys", "plan this")
        assert text and "mock" in text.lower()

    async def test_structured_validates_schema(self):
        service = IntelligenceService(MockLLMProvider(), model="m")
        result = await service.structured("sys", "Tell me about persistence", Answer)
        assert isinstance(result, Answer)
        assert result.answer

    async def test_structured_returns_none_on_provider_failure(self):
        service = IntelligenceService(MockLLMProvider(scenario="failure"), max_retries=0)
        assert await service.structured("s", "u", Answer) is None

    async def test_structured_returns_none_on_garbage(self):
        service = IntelligenceService(MockLLMProvider(scenario="garbage"), max_retries=0)
        assert await service.structured("s", "u", Answer) is None

    async def test_retries_exhaust_then_none(self):
        provider = MockLLMProvider(scenario="failure")
        service = IntelligenceService(provider, max_retries=2)
        assert await service.generate("s", "u") is None
        assert len(provider.requests) == 3  # 1 + 2 retries

    def test_info_exposes_provider_descriptor(self):
        service = IntelligenceService(MockLLMProvider(), model="m")
        info = service.info()
        assert info["provider"] == "mock"
        assert "generation" in info["capabilities"]


class TestLLMPlanner:
    async def test_explicit_goal_steps_bypass_the_model(self):
        provider = MockLLMProvider()
        service = IntelligenceService(provider, max_retries=0)
        planner = LLMPlanner(service, fallback=DeterministicPlanner(default_action="echo"))
        goal = _goal(params={"steps": [{"type": "echo", "params": {"text": "hi"}}]})
        plan = await planner.plan(goal, AgentState())
        assert provider.requests == []  # no model call for explicit plans
        assert plan.step_types == ["echo"]

    async def test_model_plan_validated_and_used(self):
        service = IntelligenceService(MockLLMProvider(), max_retries=0)
        planner = LLMPlanner(
            service, fallback=DeterministicPlanner(), allowed_actions=frozenset({"echo"})
        )
        plan = await planner.plan(_goal(), AgentState())
        assert plan.step_types == ["echo"]
        assert plan.rationale

    async def test_disallowed_steps_filtered_to_fallback(self):
        service = IntelligenceService(MockLLMProvider(), max_retries=0)
        planner = LLMPlanner(
            service, fallback=DeterministicPlanner(default_action="echo"),
            allowed_actions=frozenset({"unrelated_action"}),
        )
        plan = await planner.plan(_goal(), AgentState())
        assert plan.step_types == ["echo"]  # fallback, not the mock's echo step

    async def test_provider_failure_falls_back(self):
        service = IntelligenceService(MockLLMProvider(scenario="failure"), max_retries=0)
        planner = LLMPlanner(
            service, fallback=DeterministicPlanner(default_action="echo")
        )
        plan = await planner.plan(_goal(), AgentState())
        assert plan.step_types == ["echo"]
        assert planner.name == "llm"

    async def test_garbage_json_falls_back(self):
        service = IntelligenceService(MockLLMProvider(scenario="garbage"), max_retries=0)
        planner = LLMPlanner(
            service, fallback=DeterministicPlanner(default_action="echo")
        )
        plan = await planner.plan(_goal(), AgentState())
        assert plan.step_types == ["echo"]


class TestLLMEvaluator:
    async def test_llm_verdict_recorded_with_source(self):
        service = IntelligenceService(MockLLMProvider(), max_retries=0)
        evaluator = LLMEvaluator(service)
        state = _state_with_action(success=True)
        evaluation = await evaluator.evaluate_run(state)
        assert evaluation.source == "llm"
        assert evaluation.passed
        assert 0.0 <= evaluation.score <= 1.0
        assert "llm" in evaluation.notes

    async def test_failure_falls_back_to_deterministic(self):
        service = IntelligenceService(MockLLMProvider(scenario="failure"), max_retries=0)
        evaluator = LLMEvaluator(service)
        state = _state_with_action(success=True)
        evaluation = await evaluator.evaluate_run(state)
        assert evaluation.source == "deterministic"

    async def test_action_evaluation_is_deterministic(self):
        service = IntelligenceService(MockLLMProvider(), max_retries=0)
        evaluator = LLMEvaluator(service)
        state = AgentState()
        spec = ActionSpec(type="echo")
        result = _state_with_action(success=True).actions[0].result
        evaluation = await evaluator.evaluate_action(spec, result, state)
        assert evaluation.source == "deterministic"
        assert evaluation.passed


# -- helpers -----------------------------------------------------------------


def _goal(params=None):
    from agency.goals import Goal

    return Goal(title="Research memory systems", params=params or {})


def _state_with_action(*, success: bool) -> AgentState:
    from agency.actions.base import ActionResult

    spec = ActionSpec(type="echo")
    result = ActionResult(
        action_id=spec.id,
        action_type="echo",
        success=success,
        error=None if success else "boom",
    )
    state = AgentState(goal_title="Research memory systems")
    state.add_action(ActionRecord(spec=spec, result=result))
    return state
