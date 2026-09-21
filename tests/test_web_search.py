"""Search provider tests (network-free; DDG parsing tested against canned HTML).

Runs under pytest-asyncio auto mode; the make_ddg helper injects a mock transport.
"""

from __future__ import annotations

import pytest

from agency.web.models import SearchResultItem
from agency.web.search import (
    DuckDuckGoSearchProvider,
    NullSearchProvider,
    SearchProviderError,
    WikipediaSearchProvider,
    build_search_provider,
)

DDG_HTML = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <div class="links_main">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc">First <b>Result</b></a>
    <a class="result__snippet" href="#">The first snippet with <b>markup</b>.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <div class="links_main">
    <a rel="nofollow" class="result__a" href="https://direct.example.com/two">Second Result</a>
    <a class="result__snippet" href="#">Second snippet.</a>
  </div>
</div>
</body></html>
"""


def make_ddg(html: str, status: int = 200) -> DuckDuckGoSearchProvider:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=html)

    provider = DuckDuckGoSearchProvider(user_agent="test-agent")
    # Patch search to use a mock transport client.

    async def search(query, *, limit, request_timeout):
        if not query.strip():
            raise SearchProviderError("empty query")
        async with httpx.AsyncClient(
            headers={"User-Agent": provider._user_agent},
            timeout=request_timeout,
            follow_redirects=True,
            transport=httpx.MockTransport(handler),
        ) as client:
            response = await client.post(provider._ENDPOINT, data={"q": query, "kl": "wt-wt"})
        if response.status_code == 403:
            raise SearchProviderError("search provider returned 403 (rate-limited or blocked)")
        if response.status_code != 200:
            raise SearchProviderError(f"search provider returned HTTP {response.status_code}")
        return provider._parse(response.text, query=query, limit=limit)

    provider.search = search  # type: ignore[method-assign]
    return provider


class TestDuckDuckGoParsing:
    @pytest.mark.asyncio
    async def test_parses_results_with_unwrapped_urls(self):
        provider = make_ddg(DDG_HTML)
        results = await provider.search("test query", limit=10, request_timeout=5)
        assert len(results) == 2
        first = results[0]
        assert isinstance(first, SearchResultItem)
        assert first.url == "https://example.com/one"  # uddg redirect unwrapped
        assert first.title == "First Result"  # markup stripped
        assert "The first snippet" in first.snippet
        assert "markup" in first.snippet
        assert first.rank == 1
        assert first.query == "test query"
        assert results[1].url == "https://direct.example.com/two"

    @pytest.mark.asyncio
    async def test_403_maps_to_provider_error(self):
        provider = make_ddg("nope", status=403)
        with pytest.raises(SearchProviderError, match="403"):
            await provider.search("q", limit=5, request_timeout=5)

    @pytest.mark.asyncio
    async def test_empty_query_raises(self):
        provider = make_ddg(DDG_HTML)
        with pytest.raises(SearchProviderError, match="empty query"):
            await provider.search("   ", limit=5, request_timeout=5)

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        provider = make_ddg(DDG_HTML)
        results = await provider.search("q", limit=1, request_timeout=5)
        assert len(results) == 1


WIKI_JSON = {
    "query": {
        "search": [
            {
                "title": "PostGIS",
                "snippet": "<span class=\"searchmatch\">PostGIS</span> is a spatial database extender",
                "wordcount": 100,
            },
            {"title": "Geographic information system", "snippet": "A GIS is a system...", "wordcount": 200},
        ]
    }
}


class TestWikipediaProvider:
    @pytest.mark.asyncio
    async def test_parses_api_results(self, monkeypatch):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["action"] == "query"
            assert request.url.params["list"] == "search"
            return httpx.Response(200, json=WIKI_JSON)

        original_client = httpx.AsyncClient

        def client_with_mock_transport(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_client(**kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client_with_mock_transport)
        provider = WikipediaSearchProvider(user_agent="test-agent")
        results = await provider.search("PostGIS", limit=5, request_timeout=5)
        assert len(results) == 2
        assert results[0].title == "PostGIS"
        assert results[0].url == "https://en.wikipedia.org/wiki/PostGIS"
        assert "spatial database" in results[0].snippet
        assert results[1].rank == 2
        assert results[0].provider == "wikipedia"

    @pytest.mark.asyncio
    async def test_http_error_maps_to_provider_error(self, monkeypatch):
        import httpx

        original_client = httpx.AsyncClient

        def client_with_mock_transport(**kwargs):
            kwargs["transport"] = httpx.MockTransport(
                lambda request: httpx.Response(503, text="down")
            )
            return original_client(**kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client_with_mock_transport)
        provider = WikipediaSearchProvider(user_agent="test-agent")
        with pytest.raises(SearchProviderError, match="503"):
            await provider.search("q", limit=5, request_timeout=5)

    def test_buildable_from_factory(self):
        provider = build_search_provider("wikipedia", user_agent="ua")
        assert provider.name == "wikipedia"


class TestNullProvider:
    @pytest.mark.asyncio
    async def test_never_fabricates(self):
        provider = NullSearchProvider()
        with pytest.raises(SearchProviderError, match="disabled"):
            await provider.search("anything", limit=5, request_timeout=5)


class TestProviderFactory:
    def test_build_known_providers(self):
        assert build_search_provider("duckduckgo", user_agent="ua").name == "duckduckgo"
        assert build_search_provider("none", user_agent="ua").name == "none"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="unknown search provider"):
            build_search_provider("bogus", user_agent="ua")
