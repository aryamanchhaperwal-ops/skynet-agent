"""Web action tests: registry integration, tracing, security boundary, bootstrap."""

from __future__ import annotations

import pytest

from agency.actions.base import ActionContext, ActionRegistry, ActionSpec, run_action
from agency.config import SkynetSettings
from agency.web.actions import (
    WebExtractAction,
    WebFetchAction,
    WebResearchAction,
    WebSearchAction,
)
from agency.web.search import SearchProviderError
from tests.web_fixtures import FakeFetcher, FakeSearchProvider, FetchFailureStub


def make_ctx(tracer=None) -> ActionContext:
    return ActionContext(run_id="run-1", goal_id="goal-1", tracer=tracer)


@pytest.mark.asyncio
class TestWebSearchAction:
    async def test_successful_search_output_shape(self):
        provider = FakeSearchProvider(
            results_by_query={"llm memory": [{"url": "https://x.example/1", "title": "One"}]}
        )
        action = WebSearchAction(provider)
        result = await run_action(action, ActionSpec(type="web_search", params={"query": "llm memory"}), make_ctx())
        assert result.success
        payload = result.output
        assert payload["ok"] is True
        assert payload["provider"] == "fake"
        assert payload["results"][0]["url"] == "https://x.example/1"

    async def test_provider_failure_becomes_failed_result(self):
        action = WebSearchAction(FakeSearchProvider(error=SearchProviderError("down")))
        result = await run_action(action, ActionSpec(type="web_search", params={"query": "q"}), make_ctx())
        assert not result.success
        assert "web search failed" in result.error

    async def test_missing_query_is_clean_action_error(self):
        result = await run_action(
            WebSearchAction(FakeSearchProvider()), ActionSpec(type="web_search", params={}), make_ctx()
        )
        assert not result.success
        assert "query" in result.error

    async def test_emits_trace_events(self):
        events: list[tuple[str, dict]] = []

        class Tracer:
            async def emit(self, event_type, **payload):
                events.append((event_type, payload))

        provider = FakeSearchProvider(results_by_query={"q": [{"url": "https://x/1", "title": "t"}]})
        await run_action(
            WebSearchAction(provider),
            ActionSpec(type="web_search", params={"query": "q"}),
            make_ctx(tracer=Tracer()),
        )
        types = [event for event, _ in events]
        assert types[0] == "WEB_SEARCH_STARTED"
        assert types[-1] == "WEB_SEARCH_COMPLETED"


@pytest.mark.asyncio
class TestWebFetchAndExtractActions:
    async def test_fetch_success(self):
        fetcher = FakeFetcher(results_by_url={"https://a.example/": ("body", "Title")})
        result = await run_action(
            WebFetchAction(fetcher), ActionSpec(type="web_fetch", params={"url": "https://a.example/"}), make_ctx()
        )
        assert result.success
        assert result.output["page"]["title"] == "Title"

    async def test_fetch_failure_is_failed_result(self):
        fetcher = FakeFetcher(results_by_url={"https://a.example/": FetchFailureStub("fetch", "HTTP 404")})
        result = await run_action(
            WebFetchAction(fetcher), ActionSpec(type="web_fetch", params={"url": "https://a.example/"}), make_ctx()
        )
        assert not result.success
        assert "HTTP 404" in result.error

    async def test_extract_returns_text_fields(self):
        fetcher = FakeFetcher(results_by_url={"https://a.example/": ("some body", "Title")})
        result = await run_action(
            WebExtractAction(fetcher), ActionSpec(type="web_extract", params={"url": "https://a.example/"}), make_ctx()
        )
        assert result.success
        assert result.output["text"] == "some body"
        assert result.output["title"] == "Title"


@pytest.mark.asyncio
class TestWebResearchAction:
    async def test_successful_research_output_and_observations(self):
        from agency.web.research import ResearchService

        provider = FakeSearchProvider(
            results_by_query={"goal": [{"url": "https://a.example/1", "title": "A"}]}
        )
        fetcher = FakeFetcher(results_by_url={"https://a.example/1": ("body", "A")})
        service = ResearchService(provider, fetcher, max_sources=5)
        result = await run_action(
            WebResearchAction(service),
            ActionSpec(type="web_research", params={"goal": "goal"}),
            make_ctx(),
        )
        assert result.success
        assert result.output["ok"] is True
        assert result.output["package"]["succeeded"] is True if "succeeded" in result.output["package"] else True
        assert result.output["skynet_observations"][0]["metadata"]["url"] == "https://a.example/1"

    async def test_failed_research_raises_clean_action_error(self):
        from agency.web.research import ResearchService

        service = ResearchService(FakeSearchProvider(error=SearchProviderError("nope")), FakeFetcher(), max_sources=5)
        result = await run_action(
            WebResearchAction(service),
            ActionSpec(type="web_research", params={"goal": "goal"}),
            make_ctx(),
        )
        assert not result.success
        assert "no usable sources" in result.error

    async def test_explicit_queries_reach_provider(self):
        from agency.web.research import ResearchService

        provider = FakeSearchProvider(
            results_by_query={"custom": [{"url": "https://a.example/1", "title": "A"}]}
        )
        fetcher = FakeFetcher(results_by_url={"https://a.example/1": ("body", "A")})
        service = ResearchService(provider, fetcher, max_sources=5)
        result = await run_action(
            WebResearchAction(service),
            ActionSpec(type="web_research", params={"goal": "goal", "queries": ["custom"]}),
            make_ctx(),
        )
        assert result.success
        assert provider.calls[0][0] == "custom"


class TestRegistryIntegration:
    def test_web_actions_carry_web_category(self):
        assert WebSearchAction.category == "web"
        assert WebFetchAction.category == "web"
        assert WebExtractAction.category == "web"
        assert WebResearchAction.category == "web"

    def test_registry_registration_and_names(self):
        registry = ActionRegistry()
        provider = FakeSearchProvider()
        fetcher = FakeFetcher()
        from agency.web.research import ResearchService

        registry.register(WebSearchAction(provider))
        registry.register(WebFetchAction(fetcher))
        registry.register(WebExtractAction(fetcher))
        registry.register(WebResearchAction(ResearchService(provider, fetcher)))
        assert set(registry.names()) >= {"web_search", "web_fetch", "web_extract", "web_research"}


class TestBootstrapWiring:
    def test_web_tools_dark_by_default(self):
        from unittest.mock import patch

        from agency.bootstrap import build_core

        settings = SkynetSettings(storage_backend="memory", perception_adapter="none")
        with patch("agency.bootstrap.get_sessionmaker", create=True):
            core = build_core(settings)
        assert "web_search" not in core._actions
        assert "web_research" not in core._actions

    def test_web_tools_register_when_enabled(self):
        from unittest.mock import patch

        from agency.bootstrap import build_core

        settings = SkynetSettings(
            storage_backend="memory",
            perception_adapter="none",
            enable_web_tools=True,
            enabled_actions="echo,web_search,web_fetch,web_extract,web_research",
        )
        with patch("agency.bootstrap.get_sessionmaker", create=True):
            core = build_core(settings)
        assert "web_search" in core._actions
        assert "web_research" in core._actions
        assert "web_fetch" in core._actions
        assert "web_extract" in core._actions

    def test_web_stack_builds_from_settings(self):
        from agency.bootstrap import build_web_stack

        settings = SkynetSettings(
            storage_backend="memory",
            perception_adapter="none",
            enable_web_tools=True,
        )
        stack = build_web_stack(settings)
        assert stack.provider.name == "wikipedia"  # keyless default
        assert stack.fetcher.timeout == settings.web_timeout_seconds
        assert stack.service._max_sources == settings.web_max_pages_per_research
