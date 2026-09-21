"""robots.txt compliance for the web fetcher.

``urllib.robotparser`` is synchronous and downloads the file itself; this
module adapts it to async httpx by parsing robots content we fetch ourselves,
with per-host caching and TTLs so politeness never becomes a per-request tax.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.robotparser import RobotFileParser

logger = logging.getLogger("skynet.web.robots")

#: A robots.txt decision older than this is refetched.
_CACHE_TTL_SECONDS = 1800.0


@dataclass
class _CachedRules:
    parser: RobotFileParser | None  # None = unreachable/missing robots
    fetched_at: float


class RobotsCache:
    """Per-host robots.txt cache with allow/deny decisions.

    Semantics:

    - missing / unreachable / malformed robots.txt → **allowed** (de-facto
      convention: no robots.txt means no restrictions),
    - an explicit ``Disallow`` for our UA (or ``*``) → not allowed,
    - the fetch itself honors timeouts and is fail-open only on network
      errors — never on explicit disallow.
    """

    def __init__(self, user_agent: str, timeout: float = 5.0) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache: dict[str, _CachedRules] = {}

    async def allowed(self, client, url: str) -> bool:
        """Return whether our UA may fetch ``url`` per its robots.txt.

        ``client`` is an ``httpx.AsyncClient`` (dependency-injected for tests).
        Never raises.
        """
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        host = parts.netloc
        cached = self._cache.get(host)
        if cached is not None and time.monotonic() - cached.fetched_at < _CACHE_TTL_SECONDS:
            return self._decide(cached.parser, url)

        robots_url = f"{parts.scheme}://{host}/robots.txt"
        parser: RobotFileParser | None = None
        try:
            response = await client.get(robots_url, timeout=self._timeout)
            if 200 <= response.status_code < 300:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
        except Exception as exc:  # network failure — fail open, log it
            logger.debug("robots.txt unreachable for %s: %s", host, exc)
        self._cache[host] = _CachedRules(parser=parser, fetched_at=time.monotonic())
        return self._decide(parser, url)

    def _decide(self, parser: RobotFileParser | None, url: str) -> bool:
        if parser is None:
            return True
        try:
            return parser.can_fetch(self._user_agent, url)
        except Exception:  # pragma: no cover - malformed robots content
            return True


def robots_metadata(allowed: bool | None) -> dict:
    """Attach the robots decision to a :class:`RetrievalMetadata`-shaped dict."""
    return {"robots_allowed": allowed}


__all__ = ["RobotsCache", "robots_metadata"]
