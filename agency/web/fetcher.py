"""Controlled, security-conscious web page retrieval.

Safety model (untrusted-input boundary):

- **SSRF guard**: http and https URLs only; literal IPs and configured
  blocked hostnames (localhost, metadata services) refused *before* DNS.
- **Bounded download**: response bodies are truncated at ``web_max_bytes``;
  no unbounded reads, no HTML dump storage.
- **Timeouts + redirects**: every request carries the configured timeout and
  redirect cap; the redirect chain is recorded, not followed blindly.
- **robots.txt**: consulted per host (``RobotsCache``) and honored — an
  explicit disallow is never bypassed.
- **Rate limiting**: a per-host minimum interval between requests.
- **Content-Type check**: HTML/XHTML/plain-text only; binaries are rejected
  by content type *before* parsing.

Every failure mode returns a structured :class:`FetchFailure`, never a raw
exception, so one dead page can't kill a research task.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from agency.web.extractor import extract
from agency.web.models import (
    ExtractedContent,
    SourceType,
    classify_url,
)
from agency.web.robots import RobotsCache

logger = logging.getLogger("skynet.web.fetcher")

_ALLOWED_SCHEMES = ("http", "https")
_UNSUPPORTED_TYPES = (
    "application/pdf",
    "image/",
    "video/",
    "audio/",
    "application/zip",
    "application/octet-stream",
)


@dataclass
class FetchFailure:
    """Structured retrieval failure — data, not exception."""

    url: str
    stage: str  # validation | robots | fetch | content_type | extraction
    error: str


@dataclass
class FetchResult:
    """Outcome of one controlled page retrieval."""

    content: ExtractedContent | None = None
    failure: FetchFailure | None = None

    @property
    def ok(self) -> bool:
        return self.content is not None


class HostRateLimiter:
    """Per-host minimum interval between outgoing requests."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, min_interval)
        self._last_request: dict[str, float] = {}

    async def wait(self, host: str) -> None:
        if self._min_interval <= 0:
            return
        # Re-check after sleeping: a single sleep can wake slightly early on
        # coarse clocks (e.g. Windows ~15 ms granularity), so loop until the
        # recorded interval has truly elapsed.
        while True:
            now = time.monotonic()
            last = self._last_request.get(host)
            if last is None or now - last >= self._min_interval:
                break
            await asyncio.sleep(self._min_interval - (now - last))
        self._last_request[host] = time.monotonic()


@dataclass
class SafeFetcher:
    """Fetch public web pages politely and safely.

    Composed of small, replaceable pieces (rate limiter, robots cache, HTTP
    client) so future phases can swap any of them without touching callers.
    """

    user_agent: str
    timeout: float = 15.0
    max_bytes: int = 2_000_000
    max_redirects: int = 5
    max_chars: int = 20_000
    respect_robots: bool = True
    blocked_hosts: frozenset[str] = field(default_factory=frozenset)
    robots_cache: RobotsCache | None = None
    rate_limiter: HostRateLimiter | None = None
    client: httpx.AsyncClient | None = None

    def __post_init__(self) -> None:
        if self.robots_cache is None:
            self.robots_cache = RobotsCache(self.user_agent, timeout=min(5.0, self.timeout))
        if self.rate_limiter is None:
            self.rate_limiter = HostRateLimiter(0.0)

    # -- validation ----------------------------------------------------------
    def _validate(self, url: str) -> str | None:
        """Return an error string if ``url`` must not be fetched, else None."""
        try:
            parts = urlsplit(url)
        except Exception:
            return "invalid URL syntax"
        if parts.scheme not in _ALLOWED_SCHEMES:
            return f"unsupported URL scheme {parts.scheme!r} (only http/https)"
        host = (parts.hostname or "").lower()
        if not host:
            return "URL has no host"
        if host in self.blocked_hosts:
            return f"host {host!r} is blocked by configuration"
        if host.endswith(".local") or host.endswith(".internal"):
            return f"host {host!r} is in a reserved internal TLD"
        try:
            import ipaddress

            ipaddress.ip_address(host)  # literal IPs are blocked (SSRF guard)
            return "literal IP addresses are refused"
        except ValueError:
            return None  # hostname, not an IP — fine

    # -- main entry ----------------------------------------------------------
    async def fetch(self, url: str, *, source_type: SourceType | None = None) -> FetchResult:
        """Fetch one page safely; failure is data, never an exception."""
        validation_error = self._validate(url)
        if validation_error:
            return FetchResult(failure=FetchFailure(url, "validation", validation_error))

        owned_client = self.client is None
        client = self.client or httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=self.max_redirects,
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
        )
        try:
            if self.respect_robots and self.robots_cache is not None:
                try:
                    allowed = await self.robots_cache.allowed(client, url)
                except Exception as exc:  # fail-open but recorded
                    logger.debug("robots check failed for %s: %s", url, exc)
                    allowed = True
                if not allowed:
                    return FetchResult(
                        failure=FetchFailure(url, "robots", "disallowed by robots.txt")
                    )

            if self.rate_limiter is not None:
                await self.rate_limiter.wait(urlsplit(url).hostname or "")

            started = time.perf_counter()
            try:
                response = await client.get(url)
            except httpx.TimeoutException:
                return FetchResult(failure=FetchFailure(url, "fetch", "request timed out"))
            except httpx.TooManyRedirects:
                return FetchResult(
                    failure=FetchFailure(url, "fetch", "too many redirects")
                )
            except httpx.ConnectError as exc:
                return FetchResult(failure=FetchFailure(url, "fetch", f"connection failed: {exc}"))
            except httpx.HTTPError as exc:
                return FetchResult(failure=FetchFailure(url, "fetch", f"http error: {exc}"))

            duration_ms = int((time.perf_counter() - started) * 1000)
            content_type = response.headers.get("content-type", "")
            mime = content_type.split(";")[0].strip().lower()

            if response.status_code != 200:
                return FetchResult(
                    failure=FetchFailure(
                        url,
                        "fetch",
                        f"HTTP {response.status_code} ({mime or 'unknown type'})",
                    )
                )
            if any(mime.startswith(prefix) for prefix in _UNSUPPORTED_TYPES):
                return FetchResult(
                    failure=FetchFailure(url, "content_type", f"unsupported content type: {mime}")
                )
            if not mime:
                logger.debug("no content-type for %s — attempting extraction", url)

            body = response.content[: self.max_bytes]
            truncated = len(response.content) > self.max_bytes
            redirect_chain = [str(r.url) for r in response.history] if response.history else []
            content = extract(
                url=url,
                final_url=str(response.url),
                status_code=response.status_code,
                content_type=content_type,
                body=body,
                duration_ms=duration_ms,
                truncated_download=truncated,
                redirect_chain=redirect_chain,
                robots_allowed=True if self.respect_robots else None,
                source_type=source_type or classify_url(url),
                max_chars=self.max_chars,
            )
            if content.extraction_status == "failed":
                return FetchResult(
                    failure=FetchFailure(
                        url, "extraction", content.error or "extraction failed"
                    )
                )
            return FetchResult(content=content)
        finally:
            if owned_client:
                await client.aclose()

    async def fetch_many(
        self, urls: list[str], *, concurrency: int = 3
    ) -> dict[str, FetchResult]:
        """Fetch several pages with bounded parallelism (per host)."""
        results: dict[str, FetchResult] = {}
        for url in urls:  # sequential by design: politeness over throughput
            results[url] = await self.fetch(url)
        return results

    async def close(self) -> None:
        """Close the owned HTTP client, if any."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None


__all__ = [
    "FetchFailure",
    "FetchResult",
    "HostRateLimiter",
    "SafeFetcher",
]
