"""Search provider abstraction.

One interface, many backends: today a keyless DuckDuckGo HTML endpoint
("lite" endpoint, parsed with stdlib), tomorrow Brave/Serper/Tavily/Google
CSE — each is a :class:`SearchProvider` subclass keyed in
``SKYNET_SEARCH_PROVIDER``. Nothing in the research pipeline knows which
provider ran; results are the same :class:`SearchResultItem` shape either
way. No API keys live in code — providers read their own config from the
environment.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html import unescape
from urllib.parse import quote_plus

import httpx

from agency.web.models import SearchResultItem

logger = logging.getLogger("skynet.web.search")


class SearchProviderError(RuntimeError):
    """Raised by providers on failure; the research layer converts to data."""


@dataclass
class SearchOutcome:
    """Structured search outcome: results or a failure, never both."""

    results: list[SearchResultItem] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if self.results and not isinstance(self.results, list):
            self.results = list(self.results)


class SearchProvider(ABC):
    """One search backend."""

    name: str = "abstract"

    @abstractmethod
    async def search(
        self, query: str, *, limit: int, request_timeout: float
    ) -> list[SearchResultItem]:
        """Run one query and return ranked results. Raise SearchProviderError on failure."""
        ...


class DuckDuckGoSearchProvider(SearchProvider):
    """Keyless search via the DuckDuckGo ``html`` endpoint.

    Parsing a lightly structured HTML page — acceptable for a research
    prototype; a real API provider (Brave, Serper, Tavily) slots in beside
    it without touching any caller. No key, no account: legitimate public
    access, honorably used (rate-limited by the caller's politeness layer).
    """

    name = "duckduckgo"
    _ENDPOINT = "https://html.duckduckgo.com/html/"
    _RESULT_DIV = re.compile(
        r'<div[^>]*class="[^"]*result[^"]*results_links[^"]*"[^>]*>(.*?)</div>\s*</div>',
        re.DOTALL,
    )
    _LINK = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    _SNIPPET = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
    )
    _UDG_REDIRECT = re.compile(r"uddg=([^&]+)")

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent

    async def search(
        self, query: str, *, limit: int, request_timeout: float
    ) -> list[SearchResultItem]:
        if not query.strip():
            raise SearchProviderError("empty query")
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": self._user_agent},
                timeout=request_timeout,
                follow_redirects=True,
            ) as client:
                response = await client.post(
                    self._ENDPOINT, data={"q": query, "kl": "wt-wt"}
                )
        except httpx.TimeoutException as exc:
            raise SearchProviderError(f"search timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"search request failed: {exc}") from exc

        if response.status_code == 403:
            raise SearchProviderError("search provider returned 403 (rate-limited or blocked)")
        if response.status_code != 200:
            raise SearchProviderError(f"search provider returned HTTP {response.status_code}")

        return self._parse(response.text, query=query, limit=limit)

    # -- parsing ---------------------------------------------------------------
    def _parse(self, html: str, *, query: str, limit: int) -> list[SearchResultItem]:
        results: list[SearchResultItem] = []
        for rank, (link_match, snippet_match) in enumerate(
            zip(
                self._LINK.finditer(html),
                self._SNIPPET.finditer(html),
                strict=False,
            ),
            start=1,
        ):
            if rank > limit:
                break
            raw_url = unescape(link_match.group(1))
            url = self._resolve_url(raw_url)
            if not url:
                continue
            title = " ".join(re.sub(r"<[^>]+>", " ", link_match.group(2)).split())
            snippet = (
                " ".join(re.sub(r"<[^>]+>", " ", snippet_match.group(1)).split())
                if snippet_match
                else ""
            )
            results.append(
                SearchResultItem(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    rank=rank,
                    provider=self.name,
                )
            )
        return results

    def _resolve_url(self, raw: str) -> str | None:
        """DDG wraps results in /l/?uddg=<urlencoded> redirects — unwrap."""
        if not raw:
            return None
        if "duckduckgo.com/l/" in raw or "uddg=" in raw:
            match = self._UDG_REDIRECT.search(raw)
            if match:
                from urllib.parse import unquote

                return unquote(match.group(1))
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
        return None


class WikipediaSearchProvider(SearchProvider):
    """Search via the MediaWiki API (en.wikipedia.org).

    A genuinely open, documented, no-key API that explicitly welcomes
    automated access with a descriptive User-Agent — the polite default for
    keyless research. Results link to encyclopedia articles; for broader web
    coverage configure a keyed commercial provider (Brave, Serper, Tavily).
    """

    name = "wikipedia"
    _ENDPOINT = "https://en.wikipedia.org/w/api.php"

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent

    async def search(
        self, query: str, *, limit: int, request_timeout: float
    ) -> list[SearchResultItem]:
        if not query.strip():
            raise SearchProviderError("empty query")
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": self._user_agent},
                timeout=request_timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    self._ENDPOINT,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json",
                        "srlimit": str(limit),
                    },
                )
        except httpx.TimeoutException as exc:
            raise SearchProviderError(f"search timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"search request failed: {exc}") from exc

        if response.status_code == 403:
            raise SearchProviderError("search provider returned 403 (rate-limited or blocked)")
        if response.status_code != 200:
            raise SearchProviderError(f"search provider returned HTTP {response.status_code}")
        try:
            payload = response.json()
            hits = payload["query"]["search"]
        except Exception as exc:
            raise SearchProviderError(f"malformed search response: {exc}") from exc

        results: list[SearchResultItem] = []
        for rank, hit in enumerate(hits[:limit], start=1):
            title = hit.get("title", "")
            snippet = " ".join(re.sub(r"<[^>]+>", " ", hit.get("snippet", "")).split())
            url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            results.append(
                SearchResultItem(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    rank=rank,
                    provider=self.name,
                )
            )
        return results


class NullSearchProvider(SearchProvider):
    """Always-errors provider for offline/CI modes — never fabricates results."""

    name = "none"

    def __init__(self, user_agent: str = "") -> None:
        self._user_agent = user_agent

    async def search(
        self, query: str, *, limit: int, request_timeout: float
    ) -> list[SearchResultItem]:
        raise SearchProviderError("search provider disabled (SKYNET_SEARCH_PROVIDER=none)")


def build_search_provider(name: str, *, user_agent: str) -> SearchProvider:
    """Resolve a provider by configured name. New providers register here."""
    providers: dict[str, type[SearchProvider]] = {
        WikipediaSearchProvider.name: WikipediaSearchProvider,
        DuckDuckGoSearchProvider.name: DuckDuckGoSearchProvider,
        NullSearchProvider.name: NullSearchProvider,
    }
    provider_cls = providers.get(name)
    if provider_cls is None:
        raise ValueError(
            f"unknown search provider {name!r}; available: {', '.join(sorted(providers))}"
        )
    return provider_cls(user_agent=user_agent)  # type: ignore[call-arg]


__all__ = [
    "DuckDuckGoSearchProvider",
    "NullSearchProvider",
    "SearchOutcome",
    "SearchProvider",
    "SearchProviderError",
    "WikipediaSearchProvider",
    "build_search_provider",
]
