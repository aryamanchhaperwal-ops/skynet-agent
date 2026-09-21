"""Structured models for the Skynet web exploration engine.

Every piece of web-derived information keeps **provenance**: the source URL,
how it was retrieved, when, and with what outcome. Nothing in this module
interprets truthfulness — reliability assessment is a future evaluator's job
(blueprint §9); we only preserve the metadata such an evaluator will need.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class SourceType(StrEnum):
    """Coarse public-source classification.

    Distinguishing these types lets future evaluators weight primary vs
    secondary sources — the classification is metadata, not a trust score.
    """

    SEARCH_RESULT = "search_result"
    WEBPAGE = "webpage"
    DOCUMENTATION = "documentation"
    RESEARCH_PAPER = "research_paper"
    GITHUB_REPOSITORY = "github_repository"
    API_DOCUMENTATION = "api_documentation"
    OTHER = "other"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"  # e.g. truncated, empty text
    FAILED = "failed"


def classify_url(url: str) -> SourceType:
    """Best-effort deterministic source classification from the URL alone.

    Deliberately conservative: it never invents a specific type. Future
    reasoning can refine this with page content.
    """
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    if "github.com" in host:
        return SourceType.GITHUB_REPOSITORY
    if any(host.endswith(suffix) for suffix in (".gov", ".edu")) or "/docs" in path:
        return SourceType.DOCUMENTATION
    paper_hosts = ("arxiv.org", "doi.org", "aclanthology.org", "ieee", "acm.org")
    if any(marker in host for marker in paper_hosts):
        return SourceType.RESEARCH_PAPER
    return SourceType.WEBPAGE


class RetrievalMetadata(BaseModel):
    """How a source was actually retrieved — full provenance, no judgment."""

    http_status: int | None = None
    content_type: str | None = None
    duration_ms: int | None = None
    bytes_downloaded: int | None = None
    truncated: bool = False
    redirect_chain: list[str] = Field(default_factory=list)
    robots_allowed: bool | None = None
    errors: list[str] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    """One entry from a search provider."""

    id: str = Field(default_factory=_new_id)
    query: str
    title: str
    url: str
    snippet: str = ""
    rank: int = Field(ge=1, description="1 = provider's first result (relevance proxy).")
    provider: str
    source_type: SourceType = SourceType.SEARCH_RESULT
    retrieved_at: datetime = Field(default_factory=_utcnow)


class ExtractedContent(BaseModel):
    """Normalized, bounded content extracted from one retrieved page."""

    source_id: str = Field(default_factory=_new_id)
    url: str
    final_url: str = ""
    domain: str = ""
    title: str = ""
    text: str = ""
    headings: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    source_type: SourceType = SourceType.WEBPAGE
    retrieved_at: datetime = Field(default_factory=_utcnow)
    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.extraction_status is ExtractionStatus.SUCCEEDED


class FailedSource(BaseModel):
    """A source we tried to use and could not — failures stay visible."""

    url: str
    stage: Literal["validation", "robots", "search", "fetch", "extraction"]
    error: str
    occurred_at: datetime = Field(default_factory=_utcnow)


class ResearchPackage(BaseModel):
    """Complete, provenance-aware output of one research task."""

    id: str = Field(default_factory=_new_id)
    goal: str
    queries: list[str] = Field(default_factory=list)
    discovered: list[SearchResultItem] = Field(default_factory=list)
    sources: list[ExtractedContent] = Field(default_factory=list)
    failures: list[FailedSource] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None

    @property
    def succeeded(self) -> bool:
        return bool(self.sources) and any(source.succeeded for source in self.sources)
