"""Research orchestration tests — multi-query, dedup, partial failure, provenance."""

from __future__ import annotations

import pytest

from agency.web.research import ResearchPlanner, ResearchService, SourceSelector
from agency.web.search import SearchProviderError
from tests.web_fixtures import FakeFetcher, FakeSearchProvider, FetchFailureStub


class TestResearchPlanner:
    def test_goal_text_becomes_single_query(self):
        planner = ResearchPlanner()
        assert planner.queries_for("  how do   agents remember  ") == [
            "how do agents remember"
        ]

    def test_explicit_queries_win_and_cap(self):
        planner = ResearchPlanner(max_queries=2)
        queries = planner.queries_for("goal", param_queries=["a", "b", "c", ""])
        assert queries == ["a", "b"]

    def test_empty_goal_yields_no_queries(self):
        assert ResearchPlanner().queries_for("   ") == []


class TestSourceSelector:
    def test_dedups_by_canonical_url(self):
        selector = SourceSelector(max_sources=10)
        from agency.web.models import SearchResultItem

        items = [
            SearchResultItem(query="q", title="a", url="https://www.example.com/x/", rank=1, provider="p"),
            SearchResultItem(query="q", title="b", url="https://example.com/x", rank=2, provider="p"),
            SearchResultItem(query="q", title="c", url="https://example.com/y", rank=3, provider="p"),
        ]
        selected = selector.select(items)
        assert [item.url for item in selected] == ["https://www.example.com/x/", "https://example.com/y"]

    def test_rank_order_and_cap(self):
        selector = SourceSelector(max_sources=2)
        from agency.web.models import SearchResultItem

        items = [
            SearchResultItem(query="q", title=f"t{i}", url=f"https://h{i}.example/x", rank=i, provider="p")
            for i in range(1, 6)
        ]
        assert len(selector.select(items)) == 2


@pytest.mark.asyncio
class TestResearchService:
    async def test_full_research_pipeline_with_provenance(self):
        provider = FakeSearchProvider(
            results_by_query={
                "q1": [
                    {"url": "https://one.example/a", "title": "Alpha", "snippet": "s"},
                    {"url": "https://two.example/b", "title": "Beta", "snippet": "s"},
                ],
            }
        )
        fetcher = FakeFetcher(
            results_by_url={
                "https://one.example/a": ("alpha body text", "Alpha"),
                "https://two.example/b": ("beta body text", "Beta"),
            }
        )
        service = ResearchService(provider, fetcher, max_sources=5)
        package = await service.research("q1")
        assert package.succeeded
        assert package.queries == ["q1"]
        assert len(package.discovered) == 2
        assert len(package.sources) == 2
        assert package.sources[0].title == "Alpha"
        assert package.sources[0].url == "https://one.example/a"
        assert package.failures == []
        # observations preserve provenance
        assert package.observations[0]["metadata"]["url"] == "https://one.example/a"
        assert package.observations[0]["source"].startswith("web:")

    async def test_multi_query_merges_and_traces(self):
        provider = FakeSearchProvider(
            results_by_query={
                "q1": [{"url": "https://a.example/1", "title": "A"}],
                "q2": [{"url": "https://b.example/2", "title": "B"}],
            }
        )
        fetcher = FakeFetcher(
            results_by_url={
                "https://a.example/1": ("body a", "A"),
                "https://b.example/2": ("body b", "B"),
            }
        )
        events: list[tuple[str, dict]] = []

        async def emit(event, **payload):
            events.append((event, payload))

        service = ResearchService(provider, fetcher, max_sources=5)
        package = await service.research("goal", queries=["q1", "q2"], emit=emit)
        assert package.succeeded
        assert len(package.sources) == 2
        types = [event for event, _ in events]
        assert types[0] == "RESEARCH_STARTED"
        assert types.count("WEB_SEARCH_COMPLETED") == 2
        assert types.count("WEB_SOURCE_RECORDED") == 1
        assert types[-1] == "RESEARCH_COMPLETED"

    async def test_duplicate_urls_across_queries_deduped(self):
        shared = {"url": "https://same.example/page", "title": "Same"}
        provider = FakeSearchProvider(
            results_by_query={
                "q1": [shared],
                "q2": [dict(shared)],
            }
        )
        fetcher = FakeFetcher(results_by_url={"https://same.example/page": ("body", "Same")})
        service = ResearchService(provider, fetcher, max_sources=5)
        package = await service.research("goal", queries=["q1", "q2"])
        assert len(package.discovered) == 2  # both searches returned it
        assert len(package.sources) == 1  # fetched once

    async def test_single_page_failure_does_not_kill_research(self):
        provider = FakeSearchProvider(
            results_by_query={
                "q1": [
                    {"url": "https://dead.example/x", "title": "Dead"},
                    {"url": "https://alive.example/y", "title": "Alive"},
                ]
            }
        )
        fetcher = FakeFetcher(
            results_by_url={
                "https://dead.example/x": FetchFailureStub("fetch", "HTTP 500"),
                "https://alive.example/y": ("live body", "Alive"),
            }
        )
        service = ResearchService(provider, fetcher, max_sources=5)
        package = await service.research("q1")
        assert package.succeeded
        assert len(package.sources) == 1
        assert len(package.failures) == 1
        recorded = package.failures[0]
        assert (recorded.url, recorded.stage, recorded.error) == (
            "https://dead.example/x",
            "fetch",
            "HTTP 500",
        )
        assert recorded.occurred_at is not None

    async def test_all_searches_failing_yields_failed_package(self):
        provider = FakeSearchProvider(error=SearchProviderError("provider down"))
        service = ResearchService(provider, FakeFetcher(), max_sources=5)
        package = await service.research("anything")
        assert not package.succeeded
        assert "provider down" in package.warnings[0]

    async def test_no_search_results_yields_clean_failure(self):
        service = ResearchService(FakeSearchProvider({}), FakeFetcher(), max_sources=5)
        package = await service.research("obscure topic")
        assert not package.succeeded
        assert "no results" in package.warnings[-1]

    async def test_all_pages_failing_yields_failed_package(self):
        provider = FakeSearchProvider(
            results_by_query={"q": [{"url": "https://x.example/1", "title": "X"}]}
        )
        fetcher = FakeFetcher(results_by_url={"https://x.example/1": FetchFailureStub("robots", "disallowed")})
        service = ResearchService(provider, fetcher, max_sources=5)
        package = await service.research("q")
        assert not package.succeeded
        assert len(package.failures) == 1

    async def test_broken_trace_hook_cannot_kill_research(self):
        provider = FakeSearchProvider(
            results_by_query={"q": [{"url": "https://ok.example/1", "title": "OK"}]}
        )
        fetcher = FakeFetcher(results_by_url={"https://ok.example/1": ("body", "OK")})

        async def broken_emit(event, **payload):
            raise RuntimeError("trace sink exploded")

        service = ResearchService(provider, fetcher, max_sources=5)
        package = await service.research("q", emit=broken_emit)
        assert package.succeeded  # research survives the broken hook

    async def test_empty_goal_returns_empty_package(self):
        service = ResearchService(FakeSearchProvider(), FakeFetcher(), max_sources=5)
        package = await service.research("   ")
        assert not package.succeeded
        assert "empty research goal" in package.warnings
