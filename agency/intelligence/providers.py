"""Intelligence provider abstraction for Skynet's own reasoning.

One interface, many backends — mirrors :mod:`agency.comms.providers` (the
external-AI layer) but serves the *internal* intelligence seam: planning,
evaluation, synthesis and memory processing.

Providers never hard-code credentials: ``openai`` reads ``OPENAI_API_KEY``
from the environment (optionally ``SKYNET_LLM_BASE_URL`` for Ollama/vLLM/
OpenRouter-style OpenAI-compatible endpoints) and refuses to activate
without it. ``mock`` is deterministic and offline — tests and demos never
touch a paid API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx

logger = logging.getLogger("skynet.intelligence.providers")


class LLMError(RuntimeError):
    """Raised when an intelligence provider fails cleanly."""


class LLMCapability(StrEnum):
    """Declared capabilities an intelligence provider may advertise."""

    GENERATION = "generation"
    STRUCTURED_OUTPUT = "structured_output"
    STREAMING = "streaming"


@dataclass(frozen=True)
class LLMMessage:
    """One chat message in an intelligence request."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class GenerateRequest:
    """One intelligence request.

    ``json_mode`` asks (not guarantees) for machine-parseable output —
    structured requests are *validated*, never trusted.
    """

    messages: tuple[LLMMessage, ...]
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False
    timeout: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerateResponse:
    """One intelligence response with request metadata preserved."""

    text: str
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class LLMProvider(ABC):
    """Interface to one kind of intelligence backend."""

    #: Provider implementation name (configuration key, e.g. "mock").
    name: str

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Provider/model descriptor (id, model, capabilities, endpoint)."""

    @abstractmethod
    def capabilities(self) -> frozenset[LLMCapability]:
        """Declared capabilities."""

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Complete one request. Raise :class:`LLMError` on failure."""


class MockLLMProvider(LLMProvider):
    """Deterministic offline intelligence provider.

    Scriptable via ``scenario``:

    - ``normal``   — structured, useful deterministic answers
    - ``failure``  — raises :class:`LLMError` (fallback tests)
    - ``timeout``  — raises after ``timeout`` seconds
    - ``garbage``  — returns unparseable JSON for json_mode requests
    - ``conflict`` — answers that disagree with ``normal`` (comparison tests)

    The mock never calls the network; every answer is derived from the
    request contents so tests can assert on it deterministically.
    """

    name = "mock"

    def __init__(
        self,
        *,
        scenario: str = "normal",
        model: str = "skynet-mock-1",
        capabilities: frozenset[LLMCapability] | None = None,
    ) -> None:
        self.scenario = scenario
        self._model = model
        self._capabilities = (
            capabilities
            if capabilities is not None
            else frozenset(
                {LLMCapability.GENERATION, LLMCapability.STRUCTURED_OUTPUT}
            )
        )
        self.requests: list[GenerateRequest] = []

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self._model,
            "scenario": self.scenario,
            "capabilities": sorted(c.value for c in self._capabilities),
            "endpoint": "internal://mock",
        }

    def capabilities(self) -> frozenset[LLMCapability]:
        return self._capabilities

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        started = time.perf_counter()
        model = request.model or self._model

        def _finish(text: str, finish: str = "stop") -> GenerateResponse:
            return GenerateResponse(
                text=text,
                model=model,
                provider=self.name,
                latency_ms=int((time.perf_counter() - started) * 1000),
                usage={"mock_requests": len(self.requests)},
                finish_reason=finish,
            )

        if self.scenario == "failure":
            raise LLMError("mock intelligence provider simulated failure")
        if self.scenario == "timeout":
            import asyncio

            await asyncio.sleep(request.timeout + 0.1)
            raise LLMError(f"mock intelligence timeout after {request.timeout:.0f}s")
        if self.scenario == "garbage":
            if request.json_mode:
                return _finish("this is not json at all {")
            return _finish("words without structure")
        if self.scenario == "conflict":
            return _finish(
                "Deterministic mock (conflict): the opposite strategy is preferable; "
                "prioritize breadth over persistence."
            )

        # -- normal scenario: deterministic, request-derived answers ---------
        user_text = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        lowered = user_text.lower()

        if request.json_mode:
            # Emit JSON the *caller* should validate: content echoes the task.
            payload: dict[str, Any]
            if "step" in lowered or "plan" in lowered:
                payload = {
                    "steps": [
                        {
                            "type": "echo",
                            "params": {"text": f"mock plan step derived from: {user_text[:80]}"},
                        }
                    ],
                    "rationale": "mock deterministic plan (offline, no external calls)",
                }
            elif "evaluat" in lowered or "score" in lowered:
                payload = {
                    "score": 0.9,
                    "passed": True,
                    "reason": "mock deterministic evaluation: actions completed cleanly",
                    "evidence": ["mock evaluation evidence"],
                    "confidence": 0.8,
                }
            elif "extract" in lowered or "memory" in lowered:
                payload = {
                    "memories": [
                        {
                            "content": f"Key point from: {user_text[:100]}",
                            "type": "semantic",
                            "importance": 0.7,
                            "tags": ["mock", "extracted"],
                        }
                    ],
                    "summary": "mock deterministic extraction",
                }
            else:
                payload = {
                    "answer": f"mock deterministic answer to: {user_text[:100]}",
                    "confidence": 0.75,
                }
            return _finish(json.dumps(payload))

        if "summar" in lowered:
            return _finish(
                f"Deterministic mock summary: {user_text[:200]}"
                + (" …" if len(user_text) > 200 else "")
            )
        if "evaluat" in lowered or "score" in lowered:
            return _finish(
                "Deterministic mock evaluation: the recorded actions completed "
                "without errors; treat structural checks as the source of truth."
            )
        return _finish(
            f"Deterministic mock reasoning about: {user_text[:160]}"
        )


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat-completions provider (OpenAI, Ollama, vLLM…).

    Activates only when ``OPENAI_API_KEY`` exists in the environment (the
    key is read at call time, never stored). Ollama-style local endpoints
    frequently accept any placeholder key; set ``SKYNET_LLM_BASE_URL`` and
    export ``OPENAI_API_KEY=ollama`` for those.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self._model = model
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key_env = api_key_env
        self._client: httpx.AsyncClient | None = None

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self._model,
            "capabilities": sorted(c.value for c in self.capabilities()),
            "endpoint": self._base_url,
        }

    def capabilities(self) -> frozenset[LLMCapability]:
        return frozenset({LLMCapability.GENERATION, LLMCapability.STRUCTURED_OUTPUT})

    def _require_key(self) -> str:
        key = os.environ.get(self._api_key_env, "").strip()
        if not key:
            raise LLMError(
                f"provider 'openai' requires {self._api_key_env} in the environment; "
                "refusing to activate without credentials (use provider 'mock' offline)"
            )
        return key

    def _client_and_key(self) -> tuple[httpx.AsyncClient, str]:
        key = self._require_key()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client, key

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        client, key = self._client_and_key()
        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}"}
        started = time.perf_counter()
        try:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=request.timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMError(f"intelligence request failed: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(
                f"intelligence endpoint returned HTTP {response.status_code}"
            )
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
            finish = (body.get("choices") or [{}])[0].get("finish_reason", "") or ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"malformed intelligence response: {exc}") from exc
        return GenerateResponse(
            text=text,
            model=payload["model"],
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=dict(usage),
            finish_reason=finish,
        )


class ProviderActivationError(LLMError):
    """A configured provider cannot activate (missing credentials, etc.)."""


def build_llm_provider(
    name: str,
    *,
    model: str,
    base_url: str | None = None,
    scenario: str = "normal",
) -> LLMProvider:
    """Build an intelligence provider by configuration name.

    Fails loudly on unknown names or missing credentials — never invents
    behavior or silently falls back here; fallback policy belongs to the
    service layer.
    """
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockLLMProvider(scenario=scenario, model=model)
    if normalized == "openai":
        return OpenAICompatibleProvider(model=model, base_url=base_url)
    raise ProviderActivationError(
        f"unknown intelligence provider {name!r}; available: mock, openai"
    )


__all__ = [
    "GenerateRequest",
    "GenerateResponse",
    "LLMCapability",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "ProviderActivationError",
    "build_llm_provider",
]
