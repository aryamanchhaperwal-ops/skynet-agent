"""Evidence verification — corroboration status of gathered evidence.

The objective asks for *verified* findings. Verification here means: are
the findings supported by more than one independent source? This module
answers that mechanically and honestly. It never invents a source.

Classification (``status``):

- ``NO_EXTERNAL_EVIDENCE`` — nothing retrieved; verification unavailable.
- ``SINGLE_SOURCE`` — one independent source covers the topic; the finding
  is *partially verified* at best, never corroborated.
- ``MULTI_SOURCE_CORROBORATED`` — at least two independent sources (distinct
  hosts) cover the same topic.
- ``CONFLICTING`` — two independent sources covering the same topic share
  almost no vocabulary; recorded as a candidate disagreement for a human
  or evaluator to resolve, never silently averaged into a claim.

Independence is measured by **host diversity**: two pages on the same host
are not two sources. Topic grouping uses the research query that produced
the evidence (falling back to the source host), so the comparison is
between sources about the same question, not across unrelated questions.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from agency.orchestrator.models import Evidence

#: Minimum content length before a lexical comparison is meaningful.
_MIN_COMPARE_CHARS = 40
#: Jaccard similarity below this (on significant vocabulary) is treated as a
#: *candidate* conflict. Deliberately low: over-flagging is safe, pretending
#: agreement is not.
_CONFLICT_SIMILARITY = 0.05
_WORD = re.compile(r"[a-z]{4,}")


def _host(source: str, provenance: dict[str, Any]) -> str:
    url = str(provenance.get("url") or source or "")
    if "://" not in url:
        return ""
    host = (urlsplit(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _topic_key(evidence: Evidence) -> str:
    query = str(evidence.provenance.get("query") or "").strip().lower()
    if query:
        return re.sub(r"[^a-z0-9 ]+", " ", query).strip()[:120]
    host = _host(evidence.source, evidence.provenance)
    return host or "objective"


def _vocabulary(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall((text or "").lower()))


def _similarity(a: str, b: str) -> float:
    left, right = _vocabulary(a), _vocabulary(b)
    if not left or not right:
        return 1.0  # nothing to compare: do not manufacture a conflict
    return len(left & right) / len(left | right)


def verify_evidence(evidence: list[Evidence]) -> dict[str, Any]:
    """Return a structured corroboration verdict over the evidence list."""
    external = [e for e in evidence if e.source_type in {"web", "ai"}]
    if not external:
        return {
            "status": "NO_EXTERNAL_EVIDENCE",
            "external_sources": 0,
            "independent_sources": [],
            "topics": [],
            "method": "host diversity + lexical-overlap heuristic",
            "note": "no external evidence gathered; verification unavailable",
        }

    topics: dict[str, list[Evidence]] = {}
    for item in external:
        topics.setdefault(_topic_key(item), []).append(item)

    topic_reports: list[dict[str, Any]] = []
    any_corroborated = False
    any_conflict = False
    independent: set[str] = set()

    for key, items in topics.items():
        hosts = sorted({h for h in (_host(i.source, i.provenance) for i in items) if h})
        independent.update(hosts)
        conflicts: list[dict[str, Any]] = []
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                if _host(left.source, left.provenance) == _host(right.source, right.provenance):
                    continue  # same host: not independent, no conflict claim
                if (
                    len(left.content) < _MIN_COMPARE_CHARS
                    or len(right.content) < _MIN_COMPARE_CHARS
                ):
                    continue
                similarity = _similarity(left.content, right.content)
                if similarity < _CONFLICT_SIMILARITY:
                    conflicts.append(
                        {
                            "left": left.source,
                            "right": right.source,
                            "similarity": round(similarity, 4),
                        }
                    )
        corroborated = len(hosts) >= 2
        any_corroborated = any_corroborated or corroborated
        any_conflict = any_conflict or bool(conflicts)
        topic_reports.append(
            {
                "topic": key,
                "sources": [i.source for i in items],
                "hosts": hosts,
                "corroborated": corroborated,
                "conflicts": conflicts,
            }
        )

    if any_conflict:
        status = "CONFLICTING"
        note = (
            "independent sources covering the same topic share almost no "
            "vocabulary; recorded as a candidate disagreement for review"
        )
    elif any_corroborated:
        status = "MULTI_SOURCE_CORROBORATED"
        note = "at least two independent hosts cover the same topic"
    else:
        status = "SINGLE_SOURCE"
        note = "only one independent host covers the topic; not corroborated"

    return {
        "status": status,
        "external_sources": len(external),
        "independent_sources": sorted(independent),
        "topics": topic_reports,
        "method": "host diversity + lexical-overlap heuristic",
        "note": note,
    }


__all__ = ["verify_evidence"]
