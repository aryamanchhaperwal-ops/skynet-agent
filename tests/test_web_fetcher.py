"""Fetcher, robots, and rate-limit tests (all network-free via httpx MockTransport).

Note: the project runs pytest-asyncio in auto mode — async tests need no marker.
"""

from __future__ import annotations

import httpx
import pytest

from agency.web.fetcher import SafeFetcher
from agency.web.models import ExtractionStatus

UA = "SKYNET-TestBot/0.1"


def make_fetcher(handler, **overrides) -> SafeFetcher:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    defaults = dict(
        user_agent=UA,
        respect_robots=False,
        rate_limiter=None,
    )
    defaults.update(overrides)
    fetcher = SafeFetcher(**defaults)
    fetcher.client = client
    return fetcher


@pytest.mark.asyncio
async def test_successful_fetch_and_extract():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><head><title>OK</title></head><body><p>hello world</p></body></html>",
        )

    fetcher = make_fetcher(handler)
    result = await fetcher.fetch("https://example.com/page")
    assert result.ok
    assert result.content.text == "hello world"
    assert result.content.extraction_status is ExtractionStatus.SUCCEEDED
    await fetcher.close()


@pytest.mark.asyncio
async def test_http_error_is_structured_failure():
    fetcher = make_fetcher(lambda request: httpx.Response(500, text="boom"))
    result = await fetcher.fetch("https://example.com/broken")
    assert not result.ok
    assert result.failure.stage == "fetch"
    assert "HTTP 500" in result.failure.error
    await fetcher.close()


@pytest.mark.asyncio
async def test_timeout_is_structured_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow")

    fetcher = make_fetcher(handler)
    result = await fetcher.fetch("https://example.com/slow")
    assert not result.ok
    assert result.failure.stage == "fetch"
    assert "timed out" in result.failure.error
    await fetcher.close()


@pytest.mark.asyncio
async def test_invalid_urls_refused():
    fetcher = make_fetcher(
        lambda request: httpx.Response(200),
        blocked_hosts=frozenset({"localhost", "127.0.0.1", "169.254.169.254"}),
    )
    for url in (
        "ftp://example.com/x",
        "https://127.0.0.1/secret",
        "https://localhost/admin",
        "https://169.254.169.254/latest/meta-data",
        "https://[not a host]/x",
        "file:///etc/passwd",
    ):
        result = await fetcher.fetch(url)
        assert not result.ok, url
        assert result.failure.stage == "validation", url
    await fetcher.close()


@pytest.mark.asyncio
async def test_unsupported_content_type_rejected_before_parsing():
    fetcher = make_fetcher(
        lambda request: httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, content=b"\x00\x01"
        )
    )
    result = await fetcher.fetch("https://example.com/blob")
    assert not result.ok
    assert result.failure.stage == "content_type"
    await fetcher.close()


@pytest.mark.asyncio
async def test_download_truncation_recorded():
    big = "<html><body>" + "x" * 5000 + "</body></html>"
    fetcher = make_fetcher(
        lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text=big),
        max_bytes=100,
    )
    result = await fetcher.fetch("https://example.com/big")
    assert result.ok
    assert result.content.retrieval.truncated is True
    assert result.content.retrieval.bytes_downloaded <= 100
    await fetcher.close()


@pytest.mark.asyncio
async def test_redirect_chain_preserved():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.com/final"})
        return httpx.Response(
            200, headers={"content-type": "text/html"},
            text="<html><body><p>landed</p></body></html>",
        )

    fetcher = make_fetcher(handler)
    result = await fetcher.fetch("https://example.com/start")
    assert result.ok
    assert result.content.final_url == "https://example.com/final"
    assert "https://example.com/start" in result.content.retrieval.redirect_chain
    await fetcher.close()


@pytest.mark.asyncio
async def test_robots_disallow_blocks_fetch():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>x</p>")

    fetcher = make_fetcher(handler, respect_robots=True)
    blocked = await fetcher.fetch("https://example.com/private/data")
    assert not blocked.ok
    assert blocked.failure.stage == "robots"
    assert calls == ["https://example.com/robots.txt"]  # page never requested

    allowed = await fetcher.fetch("https://example.com/public/data")
    assert allowed.ok
    await fetcher.close()


@pytest.mark.asyncio
async def test_robots_missing_fails_open():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>fine</p>")

    fetcher = make_fetcher(handler, respect_robots=True)
    result = await fetcher.fetch("https://example.com/anything")
    assert result.ok
    await fetcher.close()


class TestHostRateLimiter:
    @pytest.mark.asyncio
    async def test_no_wait_configured_means_no_delay(self):
        import time as _time

        from agency.web.fetcher import HostRateLimiter

        limiter = HostRateLimiter(0.0)
        start = _time.monotonic()
        await limiter.wait("example.com")
        await limiter.wait("example.com")
        assert _time.monotonic() - start < 0.05

    @pytest.mark.asyncio
    async def test_interval_enforced_per_host(self):
        import time as _time

        from agency.web.fetcher import HostRateLimiter

        limiter = HostRateLimiter(0.05)
        start = _time.monotonic()
        await limiter.wait("a.example")
        await limiter.wait("a.example")
        assert _time.monotonic() - start >= 0.05
        # different host is not delayed by the other host's history
        start2 = _time.monotonic()
        await limiter.wait("b.example")
        assert _time.monotonic() - start2 < 0.05
