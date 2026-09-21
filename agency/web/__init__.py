"""SKYNET web exploration engine (Phase P4).

Modular, provenance-preserving web research:

- :mod:`agency.web.models`     — source/result/package structures
- :mod:`agency.web.extractor`  — HTML → bounded structured content
- :mod:`agency.web.fetcher`    — safe, polite, robots-respecting retrieval
- :mod:`agency.web.robots`     — robots.txt compliance with caching
- :mod:`agency.web.search`     — pluggable search provider interface
- :mod:`agency.web.research`   — full research-task orchestration
- :mod:`agency.web.actions`    — the four web actions for the core registry
"""

from __future__ import annotations

import httpx

from agency.web.actions import (
    WebExtractAction,
    WebFetchAction,
    WebResearchAction,
    WebSearchAction,
)
from agency.web.extractor import extract as extract_content
from agency.web.fetcher import FetchFailure, FetchResult, SafeFetcher
from agency.web.models import (
    ExtractedContent,
    ExtractionStatus,
    FailedSource,
    ResearchPackage,
    SearchResultItem,
    SourceType,
    classify_url,
)
from agency.web.research import ResearchPlanner, ResearchService
from agency.web.robots import RobotsCache
from agency.web.search import (
    DuckDuckGoSearchProvider,
    NullSearchProvider,
    SearchProvider,
    SearchProviderError,
    WikipediaSearchProvider,
    build_search_provider,
)

__all__ = [
    "DuckDuckGoSearchProvider",
    "ExtractedContent",
    "ExtractionStatus",
    "FailedSource",
    "FetchFailure",
    "FetchResult",
    "NullSearchProvider",
    "ResearchPackage",
    "ResearchPlanner",
    "ResearchService",
    "RobotsCache",
    "SafeFetcher",
    "SearchProvider",
    "SearchProviderError",
    "SearchResultItem",
    "SourceType",
    "WebExtractAction",
    "WebFetchAction",
    "WebResearchAction",
    "WebSearchAction",
    "WikipediaSearchProvider",
    "build_search_provider",
    "classify_url",
    "extract_content",
    "httpx",
]
