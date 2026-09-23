"""Built-in deterministic demo strategies (offline, no paid APIs).

The demo registers a real `research_strategy` pair against the *offline
search corpus* injected as a sandbox service:

- ``single_query:v1``   — one query derived from the goal text
- ``multi_query:v1``    — the goal plus generated variants

Metrics are **measured**, not narrated: ``sources_found`` (unique URLs),
``duplicate_rate`` (1 − unique/discovered), ``search_errors``,
``duration_ms``. Strategies read their corpus from
``ctx.service("search_corpus")`` — injected data, mirroring how a real
search provider would be injected — and never touch anything else.
"""

from __future__ import annotations

import re
import time
from typing import Any

from agency.experiments.sandbox import TrialContext

_CORPUS: dict[str, list[dict[str, str]]] = {
    "pages": [
        {
            "url": "https://en.wikipedia.org/wiki/Reinforcement_learning",
            "title": "Reinforcement learning",
            "text": (
                "reinforcement learning agents improve behavior through trial and "
                "error, delayed reward, exploration and exploitation trade-offs, "
                "and policy improvement loops with measured baselines"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Large_language_model",
            "title": "Large language model",
            "text": (
                "large language models are neural networks trained on text; "
                "agent architectures wrap them with memory, tool use, and "
                "planning loops for autonomous behavior"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Memory_(psychology)",
            "title": "Memory",
            "text": (
                "memory research distinguishes episodic memory of events, "
                "semantic memory of facts and concepts, and procedural memory "
                "of learned skills and strategies; consolidation strengthens "
                "traces over time"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Knowledge_graph",
            "title": "Knowledge graph",
            "text": (
                "a knowledge graph stores entities and relationships as a graph "
                "database, supporting structured long-term knowledge, retrieval "
                "and reasoning over facts with provenance"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Autonomous_system_(Internet)",
            "title": "Autonomous system (Internet)",
            "text": (
                "an autonomous system is a network under one administrative "
                "domain using border gateway protocol routing; unrelated to "
                "autonomous AI agents despite the shared word"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Experiment",
            "title": "Experiment",
            "text": (
                "a controlled experiment compares a treatment against a baseline "
                "under repeated trials, measures outcomes, and evaluates "
                "evidence against pre-registered acceptance criteria to avoid bias"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/A/B_testing",
            "title": "A/B testing",
            "text": (
                "A/B testing compares variant B against control A on a metric "
                "with multiple trials; regression and improvement are judged "
                "against the measured baseline, never assumed"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Metacognition",
            "title": "Metacognition",
            "text": (
                "metacognition is thinking about thinking: self-evaluation of "
                "one's own reasoning, detecting weaknesses, and selecting "
                "strategies that measurably improve future performance"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Long-term_memory",
            "title": "Long-term memory",
            "text": (
                "long-term memory stores information for extended periods; it "
                "includes episodic memory and semantic memory, consolidated by "
                "the hippocampus over time"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Hippocampus",
            "title": "Hippocampus",
            "text": (
                "the hippocampus consolidates long-term memory traces; damage "
                "impairs forming new episodic memories while leaving old ones"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Vector_database",
            "title": "Vector database",
            "text": (
                "vector databases store embeddings for semantic retrieval in ai "
                "applications, powering similarity search over documents"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            "title": "Retrieval-augmented generation",
            "text": (
                "retrieval-augmented generation grounds ai model answers in "
                "external memory sources fetched at query time"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Agent-based_model",
            "title": "Agent-based model",
            "text": (
                "agent-based models simulate autonomous agents interacting in "
                "an environment, each following simple rules"
            ),
        },
        {
            "url": "https://en.wikipedia.org/wiki/Episodic_memory",
            "title": "Episodic memory",
            "text": (
                "episodic memory recalls personal events in time and place, "
                "distinct from semantic memory of facts"
            ),
        },
    ]
}

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it of on or that the "
    "to what when where which who why with".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _corpus_for(ctx: TrialContext) -> dict[str, list[dict[str, str]]]:
    corpus = ctx.service("search_corpus")
    if not isinstance(corpus, dict):
        raise TypeError("search_corpus service must be a dict[str, list[dict]]")
    return corpus


def _search(
    corpus: dict[str, list[dict[str, str]]], query: str, *, limit: int = 5
) -> list[str]:
    """Deterministic word-boundary keyword search with a per-query cap.

    The cap mirrors every real search API (bounded result pages): a
    single-query strategy can therefore only ever see the first ``limit``
    matches, while multi-query unions several capped result sets.
    """
    tokens = _tokens(query)
    if not tokens:
        return []
    patterns = [re.compile(rf"\b{re.escape(token)}\b") for token in tokens]
    scored: list[tuple[float, str]] = []
    for entry in corpus.get("pages", []):
        text = f"{entry.get('title', '')} {entry.get('text', '')}".lower()
        score = sum(1.0 if pattern.search(text) else 0.0 for pattern in patterns)
        if score > 0:
            scored.append((score, entry["url"]))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [url for _score, url in scored[:limit]]


def _run_queries(ctx: TrialContext, queries: list[str]) -> dict[str, float]:
    """Shared measurement: run queries, dedupe, report metrics."""
    corpus = _corpus_for(ctx)
    started = time.perf_counter()
    discovered: list[str] = []
    for query in queries:
        discovered.extend(_search(corpus, query))
    duration_ms = int((time.perf_counter() - started) * 1000)
    unique = set(discovered)
    metrics = {
        "sources_found": float(len(unique)),
        "duplicate_rate": (
            round(1.0 - len(unique) / len(discovered), 4) if discovered else 0.0
        ),
        "search_errors": 0.0,
        "duration_ms": float(duration_ms),
    }
    return metrics


async def single_query_procedure(ctx: TrialContext) -> dict[str, float]:
    """Strategy A: one query from the goal text."""
    goal = str(ctx.params.get("goal", "")).strip()
    if not goal:
        raise ValueError("procedure_params['goal'] is required")
    return _run_queries(ctx, [goal])


async def multi_query_procedure(ctx: TrialContext) -> dict[str, float]:
    """Strategy B: goal query plus keyword-bigram sub-queries.

    Mirrors what a real query planner does: drop generic leading words
    ("approaches to") and search the remaining keyword pairs separately —
    each capped result set surfaces pages the full goal query ranks below
    its cap.
    """
    goal = str(ctx.params.get("goal", "")).strip()
    if not goal:
        raise ValueError("procedure_params['goal'] is required")
    tokens = _tokens(goal)
    variants: list[str] = [goal]
    bigrams = [
        f"{tokens[i]} {tokens[i + 1]}" for i in range(1, max(1, len(tokens) - 1))
    ]
    variants.extend(bigrams[:3])
    return _run_queries(ctx, variants[:4])


def register_demo_strategies(registry: Any) -> None:
    """Register both demo strategies (idempotent for repeat demos)."""
    for name, version, procedure, description in (
        (
            "single_query",
            "v1",
            single_query_procedure,
            "One search query derived from the goal text (baseline).",
        ),
        (
            "multi_query",
            "v1",
            multi_query_procedure,
            "Goal query plus deterministic keyword variants (candidate).",
        ),
    ):
        try:
            registry.register(
                name=name,
                version=version,
                capability="research_strategy",
                procedure=procedure,
                description=description,
                config={},
            )
        except ValueError:
            # Already registered (repeat demo in one process) — fine.
            pass


__all__ = ["register_demo_strategies"]
