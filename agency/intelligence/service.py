"""Intelligence service — Skynet's single doorway to model reasoning.

Every model call in Skynet goes through here (planning, evaluation,
synthesis, memory extraction): one retry policy, one timeout policy, one
usage log. Callers never talk to providers directly.

Failure policy is deliberate and local: *service-level* methods catch
provider failures and return ``None``, so callers apply their own fallback
(deterministic planner, deterministic evaluator). No call is retried at
two layers, and a broken provider can never crash the core loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from agency.intelligence.providers import (
    GenerateRequest,
    GenerateResponse,
    LLMError,
    LLMMessage,
    LLMProvider,
)

logger = logging.getLogger("skynet.intelligence.service")

#: Transient error classes retried between attempts (LLMError covers both
#: provider-raised errors and timeouts converted by providers).


class IntelligenceService:
    """High-level operations over one configured provider.

    All operations log usage (provider, model, latency, tokens when
    reported) and treat every model answer as *unvalidated data*: JSON
    outputs are parsed and schema-checked here, and a malformed or failed
    generation surfaces as ``None`` for the caller to fall back from.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 1,
    ) -> None:
        self._provider = provider
        self._model = model or ""
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self.calls: list[GenerateResponse] = []

    # -- introspection -------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def info(self) -> dict[str, Any]:
        return self._provider.info()

    # -- core operations -------------------------------------------------------

    async def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Free-form generation; ``None`` after exhausted retries."""
        response = await self._complete(
            system=system,
            user=user,
            json_mode=False,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        return response.text if response else None

    async def structured(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel | None:
        """JSON generation validated against ``schema``.

        Returns a parsed model instance, or ``None`` when the provider
        failed **or** returned unparseable/schema-violating output (both
        are logged; neither is ever trusted or patched silently).
        """
        response = await self._complete(
            system=system,
            user=user,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        if response is None:
            return None
        return self._validate_json(response.text, schema)

    # -- internals ---------------------------------------------------------------

    def _validate_json(
        self, text: str, schema: type[BaseModel]
    ) -> BaseModel | None:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "intelligence: %s returned unparseable JSON (%d chars); "
                "caller must fall back",
                self._provider.name,
                len(text),
            )
            return None
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            logger.warning(
                "intelligence: %s returned schema-violating JSON: %s",
                self._provider.name,
                exc.error_count(),
            )
            return None

    async def _complete(
        self,
        *,
        system: str,
        user: str,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
        metadata: dict[str, Any] | None,
    ) -> GenerateResponse | None:
        request = GenerateRequest(
            messages=(
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ),
            model=self._model,
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            json_mode=json_mode,
            timeout=self._timeout,
            metadata=dict(metadata or {}),
        )
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = await self._provider.generate(request)
                self.calls.append(response)
                logger.info(
                    "intelligence call: provider=%s model=%s latency_ms=%d "
                    "usage=%s json_mode=%s attempt=%d",
                    response.provider,
                    response.model,
                    response.latency_ms,
                    response.usage or {},
                    json_mode,
                    attempt,
                )
                return response
            except LLMError as exc:  # provider-signaled failure or timeout
                last_error = exc
                logger.warning(
                    "intelligence attempt %d/%d failed: %s", attempt, attempts, exc
                )
        logger.error(
            "intelligence: provider %s failed after %d attempt(s): %s",
            self._provider.name,
            attempts,
            last_error,
        )
        return None


__all__ = ["IntelligenceService"]
