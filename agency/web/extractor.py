"""Deterministic HTML → structured-content extraction.

Design decisions:

- **stdlib only.** ``html.parser`` handles malformed real-world HTML without
  an extra dependency; this is extraction, not rendering.
- **Text extraction is not title-from-``<h1>``.** The ``<title>`` tag is
  metadata; headings list is separate; main text is the concatenation of
  text nodes outside script/style — the standard heuristic pipeline.
- **Instruction-like content stays content.** Anything a page says (including
  text that *looks like* agent instructions) is extracted as text and never
  interpreted, executed, or followed. See docs/SKYNET_WEB.md, security.
- Bounded: caps on text length, link count, and payload prevent one huge or
  hostile page from dominating memory.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from agency.web.models import (
    ExtractedContent,
    ExtractionStatus,
    RetrievalMetadata,
    SourceType,
)

logger = logging.getLogger("skynet.web.extractor")

#: Tags whose entire content is skipped (invisible / non-authorial text).
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "iframe"})
#: Headings worth keeping (h1..h3; deeper nesting is noise).
_HEADING_TAGS = frozenset({"h1", "h2", "h3"})
#: If the main text extractor finds nothing in <body>, fall back to title.
_TEXT_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "td", "article", "section", "blockquote", "pre", "dd", "dt"}
)
MAX_LINKS = 100
MAX_HEADINGS = 40


class _TextHarvester(HTMLParser):
    """Single-pass, bounded text/heading/link harvester."""

    def __init__(self, *, max_chars: int, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._max_chars = max_chars
        self._base_url = base_url
        self.title = ""
        self.chunks: list[str] = []
        self.headings: list[str] = []
        self.links: list[str] = []
        self._skip_depth = 0
        self._skip_tag: str | None = None
        self._in_heading: str | None = None
        self._heading_buf: list[str] = []
        self._title_done = False
        self._seen_block_open = False

    # -- tag handling -------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _SKIP_TAGS:
            self._skip_depth = 1
            self._skip_tag = tag
            return
        if tag == "title" and not self._title_done:
            self._in_heading = "title"
            self._heading_buf = []
            return
        if tag in _HEADING_TAGS and len(self.headings) < MAX_HEADINGS:
            self._in_heading = tag
            self._heading_buf = []
            return
        if tag == "a":
            href = next((value for key, value in attrs if key == "href"), None)
            forbidden = ("mailto:", "javascript:", "tel:")
            if (
                href
                and not href.startswith(forbidden)
                and len(self.links) < MAX_LINKS
            ):
                absolute = urljoin(self._base_url, href.strip())
                if absolute.startswith(("http://", "https://")):
                    self.links.append(absolute)
        if tag in _TEXT_BLOCK_TAGS:
            self._seen_block_open = True

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth = 0
                self._skip_tag = None
            return
        if self._in_heading == tag or (self._in_heading == "title" and tag == "title"):
            text = " ".join("".join(self._heading_buf).split())
            if self._in_heading == "title":
                self.title = text
                self._title_done = True
            elif text:
                self.headings.append(text)
                self.chunks.append(text)  # headings are part of the main text too
            self._in_heading = None
            self._heading_buf = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_heading is not None:
            self._heading_buf.append(data)
            return
        text = " ".join(data.split())
        if text:
            self.chunks.append(text)

    # -- results ------------------------------------------------------------
    @property
    def truncated(self) -> bool:
        return sum(len(chunk) + 1 for chunk in self.chunks) > self._max_chars

    def text(self) -> str:
        parts: list[str] = []
        total = 0
        for chunk in self.chunks:
            if total + len(chunk) + 1 > self._max_chars:
                break
            parts.append(chunk)
            total += len(chunk) + 1
        return "\n".join(parts)


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def extract(
    *,
    url: str,
    final_url: str,
    status_code: int | None,
    content_type: str | None,
    body: bytes,
    duration_ms: int,
    truncated_download: bool,
    redirect_chain: list[str],
    robots_allowed: bool | None,
    source_type: SourceType = SourceType.WEBPAGE,
    max_chars: int = 20_000,
) -> ExtractedContent:
    """Extract one page's content into a bounded, structured record.

    Never raises: extraction failure is data, not an exception.
    """
    retrieval = RetrievalMetadata(
        http_status=status_code,
        content_type=content_type,
        duration_ms=duration_ms,
        bytes_downloaded=len(body),
        truncated=truncated_download,
        redirect_chain=redirect_chain,
        robots_allowed=robots_allowed,
    )
    record = ExtractedContent(
        url=url,
        final_url=final_url or url,
        domain=_domain(final_url or url),
        source_type=source_type,
        retrieval=retrieval,
    )

    # Non-HTML payloads: keep metadata, skip text parsing.
    supported_types = ("text/html", "application/xhtml+xml", "text/plain")
    effective_type = (content_type or "").split(";")[0].strip().lower()
    if effective_type and effective_type not in supported_types:
        record.extraction_status = ExtractionStatus.FAILED
        record.error = f"unsupported content type: {effective_type or 'unknown'}"
        return record

    try:
        text = body.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - decode with 'replace' cannot fail
        record.extraction_status = ExtractionStatus.FAILED
        record.error = f"decode failed: {exc}"
        return record

    harvester = _TextHarvester(
        max_chars=max_chars, base_url=final_url or url
    )
    try:
        harvester.feed(text)
        harvester.close()
    except Exception as exc:
        # html.parser rarely raises on malformed input, but never lose the page.
        logger.debug("parser issue on %s: %s", url, exc)
        record.extraction_status = ExtractionStatus.PARTIAL
        record.error = f"html parse incomplete: {exc}"
    else:
        record.extraction_status = ExtractionStatus.SUCCEEDED

    record.title = harvester.title
    record.headings = list(harvester.headings)
    record.links = list(harvester.links)
    record.text = harvester.text()
    if not record.text:
        record.extraction_status = ExtractionStatus.PARTIAL
        record.error = "no textual content extracted (empty or non-textual page)"
    if harvester.truncated:
        record.extraction_status = ExtractionStatus.PARTIAL
    return record


__all__ = [
    "MAX_HEADINGS",
    "MAX_LINKS",
    "extract",
]
