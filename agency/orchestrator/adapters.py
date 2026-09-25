"""External-world adapters — the real-integration boundary (spec §22).

The orchestrator never imports providers or fetchers directly; it speaks
to two narrow protocols:

- :class:`ResearchAdapter` — search/fetch over permitted web resources,
  returning provenance-carrying :class:`Evidence`.
- :class:`CommsAdapter` — questions to legitimately-available external AI
  systems, returning Evidence plus the conversation id.

Adapters enforce budgets (they are handed the run's remaining allowance),
respect robots/rate limits inside the wrapped P4/P5 stacks, and treat all
external content as **data** — the security module strips directive-like
payloads before anything is stored. ``LocalResearchAdapter`` and
``LocalCommsAdapter`` are deterministic offline implementations for
tests and demos; the ``real_*`` factories wrap the actual P4/P5 stacks
without embedding provider logic in the orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from agency.orchestrator.models import Evidence

logger = logging.getLogger("skynet.orchestrator.adapters")


class ResearchAdapter(Protocol):
    async def search(
        self, query: str, *, max_results: int, run_id: str
    ) -> list[Evidence]: ...


class CommsAdapter(Protocol):
    async def ask(
        self, question: str, *, run_id: str, timeout_seconds: float
    ) -> Evidence | None: ...

    async def available(self) -> list[str]: ...


# -- Local deterministic adapters -------------------------------------------------


class LocalResearchAdapter:
    """Offline, deterministic research over an injected corpus.

    Mirrors the P6 demo corpus service: the corpus is injected data, the
    same keyword matcher, the same Evidence provenance shape. Real web
    access swaps in via :func:`build_research_adapter` without touching
    the orchestrator.
    """

    def __init__(self, corpus: dict[str, list[dict[str, str]]] | None = None) -> None:
        self._corpus = corpus

    async def search(
        self, query: str, *, max_results: int, run_id: str
    ) -> list[Evidence]:
        del run_id  # provenance is carried in each Evidence
        if self._corpus is None:
            return []
        from agency.experiments.demo import _search

        urls = _search(self._corpus, query, limit=max_results)
        pages = {p["url"]: p for p in self._corpus.get("pages", [])}
        return [
            Evidence(
                source=url,
                source_type="web",
                content=pages[url]["text"][:2000],
                confidence=0.6,
                provenance={
                    "adapter": "local_corpus",
                    "title": pages[url].get("title", ""),
                    "query": query,
                },
            )
            for url in urls
        ]


class LocalCommsAdapter:
    """Offline deterministic 'AI participant' for tests and demos.

    Produces a cautious, clearly-labelled synthetic answer — never
    presented as a real external system.
    """

    def __init__(self, *, answers: dict[str, str] | None = None) -> None:
        self._answers = answers or {}

    async def ask(
        self, question: str, *, run_id: str, timeout_seconds: float
    ) -> Evidence | None:
        del run_id, timeout_seconds
        if not question.strip():
            return None
        text = self._answers.get(
            question,
            "local mock: no external AI was contacted; treat as placeholder only",
        )
        return Evidence(
            source="local:mock-ai",
            source_type="ai",
            content=text[:2000],
            confidence=0.3,  # placeholder opinions stay low-confidence
            provenance={"adapter": "local_mock", "question": question},
        )

    async def available(self) -> list[str]:
        return ["local:mock-ai"]


# -- Real adapters (wrap existing P4/P5 stacks) --------------------------------------


class WebResearchAdapter:
    """Real research via the P4 stack (search provider + SafeFetcher)."""

    def __init__(self, research_service: Any, max_chars: int = 2000) -> None:
        self._service = research_service
        self._max_chars = max_chars

    async def search(
        self, query: str, *, max_results: int, run_id: str
    ) -> list[Evidence]:
        try:
            package = await self._service.research(query)
        except Exception as exc:
            logger.warning("research adapter failure: %s", exc)
            return []
        evidence: list[Evidence] = []
        sources = getattr(package, "sources", None) or []
        for source in sources[:max_results]:
            evidence.append(
                Evidence(
                    source=getattr(source, "url", "") or str(source),
                    source_type="web",
                    content=(getattr(source, "extract", "") or "")[: self._max_chars],
                    confidence=0.55,
                    provenance={
                        "adapter": "web",
                        "query": query,
                        "run_id": run_id,
                        "title": getattr(source, "title", ""),
                    },
                )
            )
        return evidence


class CommsCommsAdapter:
    """Real AI↔AI communication via the P5 ConversationManager."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    async def available(self) -> list[str]:
        try:
            return [
                p.participant_id
                for p in self._manager.participants()
            ]
        except Exception:
            return []

    async def ask(
        self, question: str, *, run_id: str, timeout_seconds: float
    ) -> Evidence | None:
        del timeout_seconds  # per-request timeout is the manager's policy
        participants = await self.available()
        if not participants:
            return None
        try:
            outcome = await self._manager.research_session(
                participants[0],
                question,
                run_id=run_id,
            )
        except Exception as exc:
            logger.warning("comms adapter failure: %s", exc)
            return None
        turns = getattr(outcome, "turns", None) or []
        if not turns:
            return None
        first = turns[0]
        conversation = getattr(outcome, "conversation", None)
        return Evidence(
            source=(
                f"ai:{getattr(conversation, 'participant_id', participants[0])}"
            ),
            source_type="ai",
            content=(getattr(first, "answer", "") or "")[:2000],
            confidence=0.4,  # unverified external claim
            provenance={
                "adapter": "comms",
                "conversation_id": getattr(conversation, "id", None),
                "turns": len(turns),
                # Directive-like fragments detected by the P5 security layer
                # ride along as data — never executed (spec §18).
                "analysis_directives": list(
                    getattr(first, "analysis_directives", []) or []
                ),
            },
        )


def build_research_adapter(settings: Any) -> ResearchAdapter:
    """Local corpus adapter offline; real P4 stack when web tools enabled."""
    if getattr(settings, "web_offline_mode", False) or not getattr(
        settings, "enable_web_tools", False
    ):
        return LocalResearchAdapter()
    from agency.bootstrap import build_web_stack

    stack = build_web_stack(settings)
    return WebResearchAdapter(stack.service)


def build_comms_adapter(settings: Any) -> CommsAdapter:
    """Local mock offline; real P5 ConversationManager when comms enabled."""
    if not getattr(settings, "enable_external_comms", False):
        return LocalCommsAdapter()
    try:
        from agency.bootstrap import build_comms_stack

        stack = build_comms_stack(settings)
        return CommsCommsAdapter(stack.manager)
    except Exception:
        logger.warning("comms stack unavailable; falling back to local mock")
        return LocalCommsAdapter()


__all__ = [
    "CommsAdapter",
    "CommsCommsAdapter",
    "LocalCommsAdapter",
    "LocalResearchAdapter",
    "ResearchAdapter",
    "WebResearchAdapter",
    "build_comms_adapter",
    "build_research_adapter",
]
