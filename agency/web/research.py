"""High-level research orchestration: queries → search → select → fetch → package.

This is the "Research task" layer from the blueprint: it turns one goal like
"research the current state of AI agent memory systems" into a structured,
provenance-preserving :class:`agency.web.models.ResearchPackage`.

Design:

- **Deterministic planner first.** Query generation is simple (the goal text
  plus optional operator-supplied variants); the intelligence phase swaps in
  LLM query generation behind the same interface.
- **Every stage is traced** through an injected async ``emit(event, **payload)``
  hook — the same event stream, run_id and seq ordering as the core loop.
- **Partial failure is normal.** A dead page or a failed query degrades the
  package (recorded in ``failures``/``warnings``); only a total failure (no
  sources at all) yields ``succeeded=False``.
- **Untrusted content boundary.** Extracted text is packaged as data; nothing
  in this module interprets, follows, or executes anything found in a page.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from agency.web.models import (
    FailedSource,
    ResearchPackage,
    SearchResultItem,
    SourceType,
    classify_url,
)
from agency.web.search import SearchProvider, SearchProviderError

logger = logging.getLogger("skynet.web.research")

#: Callback signature for trace emission (wired to the run's Trace facade).
EmitFn = Callable[..., Awaitable[None]]


class ResearchPlanner:
    """Deterministic query planner (placeholder for the future LLM planner).

    Given a research goal, produce the search queries to run. Operators can
    pass explicit queries; otherwise the goal text is used as a single query.
    The interface is stable for the LLM replacement.
    """

    def __init__(self, max_queries: int = 3) -> None:
        self._max_queries = max_queries

    def queries_for(self, goal: str, param_queries: list[str] | None = None) -> list[str]:
        if param_queries:
            return [q.strip() for q in param_queries if q.strip()][: self._max_queries]
        text = " ".join(goal.split())
        return [text] if text else []


class SourceSelector:
    """Deterministic ranking/dedup over search results.

    Ranking: provider rank is the primary key (relevance proxy), deduped by
    canonicalized URL (host minus www, path without trailing slash — query
    strings kept distinct). Selection is deliberately simple; source
    *reliability* is a future evaluator's job (blueprint §9) — here we only
    pick a bounded, deduplicated set of pages to fetch.
    """

    def __init__(self, max_sources: int) -> None:
        self._max_sources = max_sources

    def select(self, discovered: list[SearchResultItem]) -> list[SearchResultItem]:
        seen: set[str] = set()
        selected: list[SearchResultItem] = []
        for item in sorted(discovered, key=lambda r: (r.rank, r.url)):
            key = self._canonical_key(item.url)
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= self._max_sources:
                break
        return selected

    @staticmethod
    def _canonical_key(url: str) -> str:
        try:
            parts = urlparse(url)
        except ValueError:
            return url
        host = (parts.hostname or "").lower().removeprefix("www.")
        path = parts.path.rstrip("/")
        return f"{host}{path}"


class ResearchService:
    """Orchestrates one full research task end to end."""

    def __init__(
        self,
        provider: SearchProvider,
        fetcher: Any,
        *,
        max_results_per_query: int = 10,
        max_sources: int = 8,
        emit: EmitFn | None = None,
    ) -> None:
        self._provider = provider
        self._fetcher = fetcher
        self._max_results_per_query = max_results_per_query
        self._max_sources = max_sources
        self._selector = SourceSelector(max_sources=max_sources)
        self.planner = ResearchPlanner()
        self._emit = emit

    async def _emit_event(self, event: str, **payload: Any) -> None:
        if self._emit is None:
            return
        try:
            await self._emit(event, **payload)
        except Exception:  # a broken trace hook must not kill research
            logger.exception("research trace emission failed for %s", event)

    async def research(
        self,
        goal: str,
        *,
        queries: list[str] | None = None,
        request_timeout: float = 15.0,
        emit: EmitFn | None = None,
    ) -> ResearchPackage:
        """Run one research task; always returns a package, never raises.

        ``emit`` overrides the service-level trace hook for this call —
        web_research actions pass the calling run's tracer here.
        """
        emit_fn = emit or self._emit

        async def trace(event: str, **payload: Any) -> None:
            if emit_fn is None:
                return
            try:
                await emit_fn(event, **payload)
            except Exception:  # broken trace hook must not kill research
                logger.exception("research trace emission failed for %s", event)

        package = ResearchPackage(goal=goal)
        if not goal.strip():
            package.warnings.append("empty research goal")
            package.finished_at = datetime.now(UTC)
            return package

        await trace("RESEARCH_STARTED", goal=goal)
        try:
            package.queries = self.planner.queries_for(goal, param_queries=queries)
            if not package.queries:
                package.warnings.append("no usable queries produced")
                package.finished_at = datetime.now(UTC)
                await trace("RESEARCH_FAILED", reason="no queries")
                return package

            discovered: list[SearchResultItem] = []
            search_failures = 0
            for query in package.queries:
                try:
                    results = await self._provider.search(
                        query, limit=self._max_results_per_query, request_timeout=request_timeout
                    )
                    await trace("WEB_SEARCH_COMPLETED", query=query, result_count=len(results))
                    discovered.extend(results)
                except SearchProviderError as exc:
                    search_failures += 1
                    message = f"search failed for query {query!r}: {exc}"
                    package.warnings.append(message)
                    await trace("WEB_SEARCH_FAILED", query=query, error=str(exc))

            if not discovered:
                package.finished_at = datetime.now(UTC)
                if search_failures:
                    await trace("RESEARCH_FAILED", reason="all searches failed")
                else:
                    package.warnings.append("search returned no results")
                    await trace("RESEARCH_FAILED", reason="no search results")
                return package

            package.discovered = discovered
            selected = self._selector.select(discovered)
            await trace(
                "WEB_SOURCE_RECORDED",
                discovered=len(discovered),
                selected=len(selected),
            )

            for item in selected:
                await trace("WEB_PAGE_FETCH_STARTED", url=item.url)
                fetch_result = await self._fetcher.fetch(
                    item.url, source_type=self._infer_source_type(item)
                )
                if fetch_result.ok:
                    content = fetch_result.content
                    assert content is not None  # narrowed by fetch_result.ok
                    package.sources.append(content)
                    await trace(
                        "WEB_CONTENT_EXTRACTED",
                        url=item.url,
                        status=(
                            content.extraction_status.value
                            if hasattr(content.extraction_status, "value")
                            else str(content.extraction_status)
                        ),
                        text_chars=len(content.text),
                    )
                else:
                    failure = fetch_result.failure
                    assert failure is not None
                    package.failures.append(
                        FailedSource(
                            url=failure.url,
                            stage=failure.stage,  # type: ignore[arg-type]
                            error=failure.error,
                        )
                    )
                    await trace(
                        "WEB_PAGE_FETCH_FAILED", url=failure.url, error=failure.error
                    )

            if not package.sources:
                package.warnings.append("no sources could be retrieved")
                package.finished_at = datetime.now(UTC)
                await trace("RESEARCH_FAILED", reason="no sources retrieved")
                return package

            package.observations = self._to_observations(package)
            package.finished_at = datetime.now(UTC)
            await trace(
                "RESEARCH_COMPLETED",
                sources=len(package.sources),
                failures=len(package.failures),
            )
            return package

        except Exception as exc:
            logger.exception("research task failed unexpectedly")
            package.warnings.append(f"research aborted: {type(exc).__name__}: {exc}")
            package.finished_at = datetime.now(UTC)
            await trace("RESEARCH_FAILED", error=f"{type(exc).__name__}: {exc}")
            return package

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _infer_source_type(item: SearchResultItem) -> SourceType:
        """URL-based classification for the fetched page; search provenance
        stays in the originating :class:`SearchResultItem`."""
        return classify_url(item.url)

    @staticmethod
    def _to_observations(package: ResearchPackage) -> list[dict[str, Any]]:
        """Convert extracted sources into Skynet-shaped observation dicts.

        Kept as plain dicts here: the actual Observation objects are created
        by the calling action with the live run_id/goal_id attached (the
        action knows them; the service doesn't).
        """
        observations: list[dict[str, Any]] = []
        for source in package.sources:
            if not source.succeeded:
                continue
            observations.append(
                {
                    "source": f"web:{source.domain}" if source.domain else "web",
                    "kind": "text",
                    "content": source.text,
                    "summary": source.title or source.url,
                    "confidence": None,
                    "metadata": {
                        "source_id": source.source_id,
                        "url": source.url,
                        "final_url": source.final_url,
                        "source_type": (
                            source.source_type.value
                            if hasattr(source.source_type, "value")
                            else str(source.source_type)
                        ),
                        "retrieved_at": source.retrieved_at.isoformat(),
                        "http_status": source.retrieval.http_status,
                        "duration_ms": source.retrieval.duration_ms,
                        "truncated": source.retrieval.truncated,
                        "robots_allowed": source.retrieval.robots_allowed,
                    },
                }
            )
        return observations


__all__ = [
    "EmitFn",
    "ResearchPlanner",
    "ResearchService",
    "SourceSelector",
]
