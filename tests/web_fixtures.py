"""Shared web-layer test fixtures (Phase P4)."""

from __future__ import annotations

from typing import Any

import pytest


class FakeSearchProvider:
    """Scriptable search provider: raises or returns canned results."""

    name = "fake"

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]] | None = None,
                 error: Exception | None = None) -> None:
        self.results_by_query = results_by_query or {}
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, limit: int, request_timeout: float) -> list[Any]:
        from agency.web.models import SearchResultItem

        self.calls.append((query, limit))
        if self.error is not None:
            raise self.error
        results = self.results_by_query.get(query, [])
        return [
            SearchResultItem(
                query=query,
                title=item.get("title", ""),
                url=item["url"],
                snippet=item.get("snippet", ""),
                rank=index + 1,
                provider=self.name,
            )
            for index, item in enumerate(results[:limit])
        ]


class FetchFailureStub:
    """Explicit failure marker for FakeFetcher: (stage, error)."""

    def __init__(self, stage: str, error: str) -> None:
        self.stage = stage
        self.error = error


class FakeFetcher:
    """Scriptable fetcher standing in for SafeFetcher.

    Values: ``FetchFailureStub(stage, error)`` → failure; ``Exception`` →
    raised; ``None`` → generic failure; ``(text, title)`` tuple or str →
    successful content.
    """

    def __init__(self, results_by_url: dict[str, Any] | None = None) -> None:
        self.results_by_url = results_by_url or {}
        self.calls: list[str] = []

    async def fetch(self, url: str, *, source_type: Any = None) -> Any:
        from agency.web.fetcher import FetchFailure, FetchResult
        from agency.web.models import ExtractedContent, ExtractionStatus, SourceType

        self.calls.append(url)
        canned = self.results_by_url.get(url)
        if isinstance(canned, FetchFailureStub):
            return FetchResult(
                failure=FetchFailure(url=url, stage=canned.stage, error=canned.error)
            )
        if isinstance(canned, Exception):
            raise canned
        if canned is None:
            return FetchResult(failure=FetchFailure(url=url, stage="fetch", error="no canned page"))
        text, title = canned if isinstance(canned, tuple) else (canned, "Fake Title")
        content = ExtractedContent(
            url=url,
            final_url=url,
            domain=url.split("//")[-1].split("/")[0],
            title=title,
            text=text,
            headings=[title] if title else [],
            extraction_status=ExtractionStatus.SUCCEEDED if text else ExtractionStatus.PARTIAL,
            source_type=source_type or SourceType.WEBPAGE,
        )
        return FetchResult(content=content)


@pytest.fixture
def fake_provider() -> FakeSearchProvider:
    return FakeSearchProvider()


@pytest.fixture
def fake_fetcher() -> FakeFetcher:
    return FakeFetcher()
