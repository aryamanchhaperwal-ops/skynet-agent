"""Web actions: the only way the Skynet core touches the web.

Four actions registered under category ``web`` (gated by
``SKYNET_ENABLE_WEB_TOOLS``):

- ``web_search``       — one query through the configured provider
- ``web_fetch``        — one controlled page retrieval
- ``web_extract``      — fetch + extraction as one step (same fetcher)
- ``web_research``     — full multi-query, multi-source research task

All of them: emit lifecycle trace events through the run's tracer (via
``ActionContext.emit_trace``), convert their outputs to Skynet observations
where useful, and return structured JSON-serializable payloads. Failures
return ``{"ok": False, "error": ...}`` data — the action system converts
exceptions to failed results, so these never crash a run.
"""

from __future__ import annotations

import logging
from typing import Any

from agency.actions.base import Action, ActionContext, ActionError, ActionSpec
from agency.web.fetcher import SafeFetcher
from agency.web.research import ResearchService
from agency.web.search import SearchProvider, SearchProviderError

logger = logging.getLogger("skynet.web.actions")


def _require_str(params: dict[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"missing or empty string parameter {key!r}")
    return value.strip()


def _optional_int(params: dict[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ActionError(f"parameter {key!r} must be an integer")
    return value


def _optional_str_list(params: dict[str, Any], key: str) -> list[str] | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ActionError(f"parameter {key!r} must be a list of strings")
    return value


class WebSearchAction(Action):
    """Search the web through the configured provider."""

    name = "web_search"
    category = "web"
    description = "Search the web via the configured provider; returns ranked results."

    def __init__(self, provider: SearchProvider, *, timeout: float = 15.0) -> None:
        self._provider = provider
        self._timeout = timeout

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        query = _require_str(spec.params, "query")
        limit = _optional_int(spec.params, "limit", 10)
        await ctx.emit_trace("WEB_SEARCH_STARTED", query=query)
        try:
            results = await self._provider.search(
                query, limit=limit, request_timeout=self._timeout
            )
        except SearchProviderError as exc:
            await ctx.emit_trace("WEB_SEARCH_FAILED", query=query, error=str(exc))
            raise ActionError(f"web search failed: {exc}") from exc
        await ctx.emit_trace("WEB_SEARCH_COMPLETED", query=query, result_count=len(results))
        return {
            "ok": True,
            "query": query,
            "provider": self._provider.name,
            "results": [item.model_dump(mode="json") for item in results],
        }


class WebFetchAction(Action):
    """Fetch one public page with full safety controls."""

    name = "web_fetch"
    category = "web"
    description = "Retrieve one public webpage safely (robots, timeouts, SSRF guard)."

    def __init__(self, fetcher: SafeFetcher) -> None:
        self._fetcher = fetcher

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        url = _require_str(spec.params, "url")
        await ctx.emit_trace("WEB_PAGE_FETCH_STARTED", url=url)
        result = await self._fetcher.fetch(url)
        if not result.ok:
            failure = result.failure
            assert failure is not None
            await ctx.emit_trace(
                "WEB_PAGE_FETCH_FAILED", url=url, stage=failure.stage, error=failure.error
            )
            raise ActionError(f"fetch failed ({failure.stage}): {failure.error}")
        content = result.content
        assert content is not None
        await ctx.emit_trace(
            "WEB_PAGE_FETCH_COMPLETED",
            url=url,
            final_url=content.final_url,
            status=content.extraction_status.value
            if hasattr(content.extraction_status, "value")
            else str(content.extraction_status),
        )
        await ctx.emit_trace(
            "WEB_CONTENT_EXTRACTED",
            url=url,
            text_chars=len(content.text),
            headings=len(content.headings),
        )
        return {"ok": True, "page": content.model_dump(mode="json")}


class WebExtractAction(Action):
    """Fetch + extract as one named step (alias semantics of web_fetch)."""

    name = "web_extract"
    category = "web"
    description = "Fetch a page and return its extracted, bounded text content."

    def __init__(self, fetcher: SafeFetcher) -> None:
        self._fetcher = fetcher

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        # web_extract IS web_fetch's extraction path; delegate for one code path.
        delegate = WebFetchAction(self._fetcher)
        output = await delegate.execute(spec, ctx)
        page = output.get("page", {})
        return {"ok": True, "url": page.get("url"), "text": page.get("text", ""),
                "title": page.get("title", ""), "headings": page.get("headings", [])}


class WebResearchAction(Action):
    """Full multi-query, multi-source research task with provenance."""

    name = "web_research"
    category = "web"
    description = (
        "Run a full research task: queries → search → select → fetch → "
        "structured package with per-source provenance."
    )

    def __init__(self, service: ResearchService) -> None:
        self._service = service

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        goal = _require_str(spec.params, "goal")
        queries = _optional_str_list(spec.params, "queries")

        async def emit(event: str, **payload: Any) -> None:
            await ctx.emit_trace(event, **payload)

        package = await self._service.research(goal, queries=queries, emit=emit)
        if not package.succeeded:
            raise ActionError(
                "research produced no usable sources: "
                + "; ".join(package.warnings or ["unknown reason"])
            )
        # The loop absorbs these into agent state via the standard Observation
        # model (source/kind/content/metadata preserved; run/goal ids attached
        # by the loop). Package stays in the output for the full provenance.
        return {
            "ok": True,
            "package": package.model_dump(mode="json"),
            "skynet_observations": package.observations,
        }


__all__ = [
    "WebExtractAction",
    "WebFetchAction",
    "WebResearchAction",
    "WebSearchAction",
]
