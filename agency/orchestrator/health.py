"""Provider health checks (real-capabilities phase).

A health check answers one question honestly: *can this provider actually
serve a request right now?* It performs one minimal, cheap, read-only
interaction — never an expensive completion — and returns a structured
:class:`HealthStatus` instead of raising for expected conditions.

Classification vocabulary used across Skynet (never silently substituted):

- ``real``        — a configured live provider (OpenAI-compatible endpoint,
                    commercial search API…). "local" Ollama endpoints are
                    still real providers — they serve real completions.
- ``local``       — deterministic in-repo implementations over injected
                    data (local corpus research, in-process stores).
- ``mock``        — scriptable test doubles (MockLLMProvider,
                    MockAIProvider). A mock result is never labelled real.
- ``unavailable`` — configured but not usable right now (unreachable
                    endpoint, missing credentials, offline mode).

The check is deliberately cheap: one tiny completion (16 tokens) against
the configured model, no retries, structured result either way.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ProviderKind = Literal["real", "local", "mock", "unavailable"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class HealthStatus(BaseModel):
    """Structured result of one provider health check."""

    component: str = "llm"  # "llm" | "web" | "comms"
    provider: str = ""
    kind: ProviderKind = "unavailable"
    model: str = ""
    endpoint: str = ""
    #: One minimal request was sent and answered.
    reachable: bool = False
    #: Credentials/configuration were accepted (True when none required).
    authenticated: bool = False
    #: The configured model served the request (or the endpoint does not
    #: expose per-model checks; then mirrors ``reachable``).
    model_available: bool = False
    #: The response could be parsed by the existing stack.
    parse_ok: bool = False
    #: Overall verdict — every applicable check above passed.
    ok: bool = False
    #: Human-readable classification label (REAL / LOCAL / MOCK / UNAVAILABLE).
    label: str = "UNAVAILABLE"
    #: False for classification-only rows (``provider status``): liveness
    #: was not probed, so "unavailable" would be a false claim.
    probed: bool = False
    detail: str = ""
    latency_ms: int = 0
    checked_at: datetime = Field(default_factory=_utcnow)
    #: Request id / response metadata when the provider reports one (never
    #: credentials).
    metadata: dict[str, Any] = Field(default_factory=dict)

    def summary_line(self) -> str:
        if self.ok:
            state = "ok"
        else:
            state = "unprobed" if not self.probed else "unavailable"
        target = f"{self.provider}" + (f":{self.model}" if self.model else "")
        return (
            f"[{self.component}] {target:<28} {self.label:<12} {state:<11} "
            f"{self.latency_ms:>5}ms  {self.detail[:70]}"
        )


def _classify(provider_name: str, endpoint: str) -> ProviderKind:
    """Classify a provider implementation (never its liveness)."""
    if provider_name == "mock":
        return "mock"
    if provider_name in {"openai", "openai-compatible"}:
        return "real"
    return "local"


async def check_llm_provider(settings: Any) -> HealthStatus:
    """Health-check the configured intelligence (LLM) provider.

    Sends ONE minimal generation request through the existing provider
    seam (:func:`agency.intelligence.providers.build_llm_provider`). The
    provider is built fresh and closed after the check; credentials stay
    in the environment and are never copied into the result.
    """
    from agency.intelligence.providers import (
        GenerateRequest,
        LLMError,
        LLMMessage,
        build_llm_provider,
    )

    provider_name = str(getattr(settings, "llm_provider", "mock"))
    model = str(getattr(settings, "llm_model", "") or "")
    base_url = getattr(settings, "llm_base_url", None)
    endpoint = (base_url or ("https://api.openai.com/v1" if provider_name == "openai" else "internal://mock")).rstrip("/")

    kind = _classify(provider_name, endpoint)
    status = HealthStatus(
        component="llm",
        provider=provider_name,
        kind=kind,
        model=model,
        endpoint=endpoint,
        label={"real": "REAL", "local": "LOCAL", "mock": "MOCK"}.get(kind, "UNAVAILABLE"),
        probed=kind != "mock",
    )
    if kind == "mock":
        status.detail = "mock provider (deterministic, offline) — not a real LLM"
        status.reachable = status.authenticated = status.model_available = status.parse_ok = True
        status.ok = True
        return status

    started = time.perf_counter()
    provider: Any = None
    try:
        provider = build_llm_provider(provider_name, model=model, base_url=base_url)
    except Exception as exc:
        status.detail = f"provider construction failed: {type(exc).__name__}: {exc}"
        status.latency_ms = int((time.perf_counter() - started) * 1000)
        return status

    try:
        response = await provider.generate(
            GenerateRequest(
                messages=(
                    LLMMessage(role="system", content="You are a health probe."),
                    LLMMessage(role="user", content="Reply with the single word: ok"),
                ),
                model=model,
                temperature=0.0,
                max_tokens=16,
                timeout=min(float(getattr(settings, "llm_timeout_seconds", 30.0)), 20.0),
                metadata={"purpose": "health_check"},
            )
        )
    except LLMError as exc:
        status.detail = f"provider request failed: {exc}"
        status.latency_ms = int((time.perf_counter() - started) * 1000)
        await _close(provider)
        return status
    except Exception as exc:  # pragma: no cover - defensive
        status.detail = f"unexpected provider error: {type(exc).__name__}: {exc}"
        status.latency_ms = int((time.perf_counter() - started) * 1000)
        await _close(provider)
        return status

    status.latency_ms = int((time.perf_counter() - started) * 1000)
    status.reachable = True
    status.authenticated = True  # the endpoint accepted the configured auth
    status.model_available = True
    status.parse_ok = bool(response.text)
    status.ok = status.parse_ok
    status.detail = "one minimal completion succeeded" if status.ok else "empty completion"
    status.metadata = {
        "latency_ms": response.latency_ms,
        "finish_reason": response.finish_reason,
        "usage": dict(response.usage or {}),
        "response_model": response.model,
    }
    await _close(provider)
    return status


async def _close(provider: Any) -> None:
    closer = getattr(provider, "aclose", None)
    if callable(closer):
        try:
            await closer()
        except Exception:  # pragma: no cover - best effort
            pass


async def check_web_stack(settings: Any) -> HealthStatus:
    """Health-check the web/research stack with one real search.

    Uses the configured :class:`~agency.web.search.SearchProvider` (the
    keyless Wikipedia API by default). Offline mode reports UNAVAILABLE —
    never a synthesized success.
    """
    from agency.web.search import SearchProviderError, build_search_provider

    provider_name = str(getattr(settings, "search_provider", "wikipedia"))
    kind = "unavailable" if provider_name == "none" else "real"
    if getattr(settings, "web_offline_mode", False):
        kind = "unavailable"
    status = HealthStatus(
        component="web",
        provider=provider_name,
        kind=kind,  # type: ignore[arg-type]
        endpoint="configuration: SKYNET_SEARCH_PROVIDER",
        label={"real": "REAL", "local": "LOCAL", "mock": "MOCK"}.get(kind, "UNAVAILABLE"),
        probed=kind != "unavailable",
    )
    if kind == "unavailable":
        status.detail = (
            "web_offline_mode is on" if getattr(settings, "web_offline_mode", False)
            else f"search provider {provider_name!r} is configured off"
        )
        return status

    started = time.perf_counter()
    try:
        provider = build_search_provider(
            provider_name,
            user_agent=str(getattr(settings, "web_user_agent", "SKYNET-HealthCheck/0.1")),
        )
        results = await provider.search(
            " skynet provider health check ",  # harmless real query
            limit=1,
            request_timeout=min(float(getattr(settings, "web_timeout_seconds", 15.0)), 15.0),
        )
    except SearchProviderError as exc:
        status.detail = f"search provider failed: {exc}"
        status.latency_ms = int((time.perf_counter() - started) * 1000)
        return status
    except Exception as exc:  # pragma: no cover - defensive
        status.detail = f"unexpected search error: {type(exc).__name__}: {exc}"
        status.latency_ms = int((time.perf_counter() - started) * 1000)
        return status

    status.latency_ms = int((time.perf_counter() - started) * 1000)
    status.reachable = True
    status.authenticated = True  # keyless providers: no auth to validate
    status.model_available = True
    status.parse_ok = bool(results)
    status.ok = status.parse_ok
    status.detail = (
        f"one live search returned {len(results)} result(s)"
        if results
        else "search reachable but returned no results"
    )
    if results:
        status.metadata = {"sample_source": results[0].url}
    return status


async def check_all(settings: Any) -> list[HealthStatus]:
    """Check every provider seam relevant to real execution."""
    return [await check_llm_provider(settings), await check_web_stack(settings)]


__all__ = [
    "HealthStatus",
    "ProviderKind",
    "check_all",
    "check_llm_provider",
    "check_web_stack",
]
