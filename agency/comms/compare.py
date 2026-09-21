"""Multi-AI response comparison — infrastructure, not a consensus engine.

Ask the same question to several participants, collect responses with full
provenance, and record where they agree or diverge **textually**. Agreement is
data for future evaluators, never treated as truth: different AI systems can
share the same training blind spot, so majority text overlap proves nothing
about correctness.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from agency.comms.manager import ConversationManager
from agency.comms.models import ResponseComparison

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i in is it its of on or
    our that the their there these they this to was we what when where which who
    will with you your do does""".split()
)

_TOKEN = re.compile(r"[a-z']{3,}")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _terms(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def compare_responses(
    prompt: str,
    responses: dict[str, str],
    failures: dict[str, str] | None = None,
) -> ResponseComparison:
    """Build a structured comparison from raw responses (pure function)."""
    comparison = ResponseComparison(
        prompt=prompt,
        participant_ids=sorted(responses) + sorted(failures or {}),
        responses={pid: _normalize(text) for pid, text in responses.items()},
        responded=sorted(responses),
        failures=dict(failures or {}),
    )
    if len(comparison.responded) < 2:
        comparison.warnings.append(
            "fewer than two usable responses; comparison is inconclusive"
        )
        return comparison

    term_sets = {pid: _terms(text) for pid, text in comparison.responses.items()}
    counts = Counter(term for terms in term_sets.values() for term in terms)
    total = len(term_sets)
    for term, count in counts.most_common():
        if count > total / 2:
            comparison.agreement_terms.append(term)
        else:
            # Present in only some responses — a candidate disagreement
            # marker. Recorded for future evaluators, never treated as proof.
            comparison.divergence_terms.append(term)
    return comparison


class ComparisonService:
    """Orchestrates one prompt to several participants and compares results.

    Each participant gets its own bounded conversation (single turn), so a
    failure with one participant never affects the others — partial results
    are the normal outcome, not an error.
    """

    def __init__(self, manager: ConversationManager) -> None:
        self._manager = manager

    async def compare(
        self,
        prompt: str,
        participant_ids: list[str],
        *,
        run_id: str | None = None,
        goal_id: str | None = None,
        goal_title: str = "",
        emit: Any = None,
    ) -> ResponseComparison:
        responses: dict[str, str] = {}
        failures: dict[str, str] = {}
        for participant_id in participant_ids:
            try:
                outcome = await self._manager.research_session(
                    participant_id,
                    prompt,
                    run_id=run_id,
                    goal_id=goal_id,
                    goal_title=goal_title,
                    emit=emit,
                )
            except Exception as exc:
                failures[participant_id] = f"{type(exc).__name__}: {exc}"
                continue
            if outcome.succeeded:
                responses[participant_id] = outcome.turns[0].answer
            else:
                failures[participant_id] = outcome.error or "no usable response"
        return compare_responses(prompt, responses, failures)


__all__ = ["ComparisonService", "compare_responses"]
