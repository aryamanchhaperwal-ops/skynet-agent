"""Experiment registry — stable ids, history, and persistence.

Experiments are recorded as JSON lines in one append-friendly file
(``data/experiments.jsonl`` by default): every write rewrites the file
atomically through a temp file + rename, so a crash mid-write cannot
corrupt history. The registry answers "have I tried this before?" via
:meth:`search` and feeds finished experiments into long-term memory
(§20) so prior evidence is recallable before new experiments start.

JSONL keeps this phase dependency-free; the blueprint's ``experiments``
Postgres table is the later migration target and the public API here is
shaped to survive it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from agency.experiments.models import Experiment, ExperimentSummary

logger = logging.getLogger("skynet.experiments.registry")

#: Cap on a single experiment record's serialized size (experiments carry
#: aggregates + criteria, not raw payloads; trials keep metadata small).
MAX_RECORD_CHARS = 200_000


class ExperimentRegistry:
    """Persistent store of experiment records (JSONL backend)."""

    def __init__(self, path: str | Path = "data/experiments.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._cache: dict[str, Experiment] | None = None

    # -- internal loading ---------------------------------------------------

    def _load(self) -> dict[str, Experiment]:
        if self._cache is not None:
            return self._cache
        cache: dict[str, Experiment] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = Experiment.model_validate(json.loads(line))
                    cache[record.id] = record
                except Exception:
                    logger.warning("skipping malformed experiment record in %s", self._path)
        self._cache = cache
        return cache

    async def _flush(self, records: dict[str, Experiment]) -> None:
        """Atomically rewrite the file (temp file + rename)."""
        lines = [
            record.model_dump_json()
            for record in sorted(records.values(), key=lambda r: r.created_at)
        ]
        payload = "\n".join(lines) + ("\n" if lines else "")
        if len(payload) > MAX_RECORD_CHARS * len(records) + 1024:
            logger.warning("experiment file growing large: %d chars", len(payload))

        def _write() -> None:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._path)

        await asyncio.to_thread(_write)

    # -- public API ----------------------------------------------------------

    async def save(self, experiment: Experiment) -> Experiment:
        """Insert or update one experiment record (by id)."""
        records = self._load()
        async with self._lock:
            records[experiment.id] = experiment.model_copy(deep=True)
            await self._flush(records)
        return experiment

    async def get(self, experiment_id: str) -> Experiment | None:
        return self._load().get(experiment_id)

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ExperimentSummary]:
        records = self._load()
        summaries = [
            ExperimentSummary(
                id=record.id,
                name=record.name,
                status=record.status,
                baseline=record.baseline.ref,
                candidate=record.candidate.ref,
                verdict=record.evaluation.verdict if record.evaluation else None,
                trials=len(record.trials_results),
                created_at=record.created_at,
                completed_at=record.completed_at,
            )
            for record in records.values()
        ]
        if status:
            summaries = [s for s in summaries if s.status.value == status]
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries[:limit]

    async def search(self, query: str, *, limit: int = 10) -> list[Experiment]:
        """Keyword search over name/description/objective/hypothesis text.

        This is the "have I tried this before?" seam (§20); relevance is
        deterministic term overlap, consistent with the memory stores.
        """
        tokens = {t for t in query.lower().split() if len(t) >= 3}
        if not tokens:
            return []
        scored: list[tuple[Experiment, float]] = []
        for record in self._load().values():
            haystack = " ".join(
                [
                    record.name,
                    record.description,
                    record.objective,
                    record.hypothesis.statement,
                ]
            ).lower()
            overlap = sum(1 for token in tokens if token in haystack) / len(tokens)
            if overlap > 0:
                scored.append((record, overlap))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [record for record, _score in scored[:limit]]

    async def prior_for(
        self, baseline: str, candidate: str, *, limit: int = 5
    ) -> list[Experiment]:
        """Experiments that already compared this exact pair (either order)."""
        matches = [
            record
            for record in self._load().values()
            if {
                record.baseline.ref,
                record.candidate.ref,
            }
            == {baseline, candidate}
        ]
        matches.sort(key=lambda r: r.created_at, reverse=True)
        return matches[:limit]


__all__ = ["MAX_RECORD_CHARS", "ExperimentRegistry"]
