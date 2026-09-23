"""Improvement history store — weaknesses, proposals, versions, regressions.

One JSONL file (``data/improvements.jsonl`` by default, atomic writes like
the experiment registry) so improvement history survives restarts and can
answer: *have I attempted this before? what happened? which version is
active? what was the previous known-good version?* A path of ``None``
keeps everything in-process (tests).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from agency.improve.models import (
    ActiveStrategy,
    CandidateStrategy,
    ImprovementProposal,
    RegressionEvent,
    StrategyVersion,
    Weakness,
)

logger = logging.getLogger("skynet.improve.store")


class ImprovementStore:
    """Persistent store of the self-improvement engine's artifacts."""

    def __init__(self, path: str | Path | None = "data/improvements.jsonl") -> None:
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._weaknesses: dict[str, Weakness] = {}
        self._proposals: dict[str, ImprovementProposal] = {}
        self._candidates: dict[str, CandidateStrategy] = {}
        self._versions: dict[str, list[StrategyVersion]] = {}
        self._active: dict[str, ActiveStrategy] = {}
        self._regressions: list[RegressionEvent] = []
        self._loaded = False

    # -- loading -----------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load history once, on first access (sync like the P6 registry:
        small JSONL, read-only after load; every accessor calls this)."""
        if self._loaded or self._path is None:
            self._loaded = True
            return
        text = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                kind = record.get("kind")
                if kind == "weakness":
                    parsed = Weakness.model_validate(record["data"])
                    self._weaknesses[parsed.id] = parsed
                elif kind == "proposal":
                    parsed = ImprovementProposal.model_validate(record["data"])
                    self._proposals[parsed.id] = parsed
                elif kind == "candidate":
                    parsed = CandidateStrategy.model_validate(record["data"])
                    self._candidates[parsed.id] = parsed
                elif kind == "version":
                    parsed = StrategyVersion.model_validate(record["data"])
                    self._versions.setdefault(parsed.name, []).append(parsed)
                elif kind == "active":
                    parsed = ActiveStrategy.model_validate(record["data"])
                    self._active[parsed.name] = parsed
                elif kind == "regression":
                    self._regressions.append(RegressionEvent.model_validate(record["data"]))
            except Exception:
                logger.warning("skipping malformed improvement record in %s", self._path)
        self._loaded = True

    async def _append(self, kind: str, data: object) -> None:
        if self._path is None:
            return
        payload = json.dumps({"kind": kind, "data": data}, default=_json_default) + "\n"

        def _write() -> None:
            with self._path.open("a", encoding="utf-8") as handle:  # type: ignore[union-attr]
                handle.write(payload)

        async with self._lock:
            await asyncio.to_thread(_write)

    # -- weaknesses ---------------------------------------------------------------

    async def save_weakness(self, weakness: Weakness) -> None:
        self._ensure_loaded()
        self._weaknesses[weakness.id] = weakness
        await self._append("weakness", weakness.model_dump(mode="json"))

    def weakness(self, weakness_id: str) -> Weakness:
        self._ensure_loaded()
        if weakness_id not in self._weaknesses:
            raise KeyError(f"unknown weakness id {weakness_id!r}")
        return self._weaknesses[weakness_id]

    def weakness_or_none(self, weakness_id: str) -> Weakness | None:
        self._ensure_loaded()
        return self._weaknesses.get(weakness_id)

    def weaknesses(self, status: str | None = None) -> list[Weakness]:
        self._ensure_loaded()
        items = sorted(self._weaknesses.values(), key=lambda w: w.created_at)
        if status:
            items = [w for w in items if w.status.value == status]
        return items

    # -- proposals -------------------------------------------------------------------

    async def save_proposal(self, proposal: ImprovementProposal) -> None:
        self._ensure_loaded()
        self._proposals[proposal.id] = proposal
        await self._append("proposal", proposal.model_dump(mode="json"))

    def proposal(self, proposal_id: str) -> ImprovementProposal:
        self._ensure_loaded()
        if proposal_id not in self._proposals:
            raise KeyError(f"unknown proposal id {proposal_id!r}")
        return self._proposals[proposal_id]

    def proposals(self, status: str | None = None) -> list[ImprovementProposal]:
        self._ensure_loaded()
        items = sorted(self._proposals.values(), key=lambda p: p.created_at)
        if status:
            items = [p for p in items if p.status.value == status]
        return items

    def prior_proposals_for(
        self, weakness_id: str, *, exclude: str | None = None
    ) -> list[ImprovementProposal]:
        """Previous attempts against the same weakness (dedupe seam)."""
        return [
            p
            for p in self.proposals()
            if p.weakness_id == weakness_id and p.id != exclude
        ]

    # -- candidates -----------------------------------------------------------------

    async def save_candidate(self, candidate: CandidateStrategy) -> None:
        self._ensure_loaded()
        self._candidates[candidate.id] = candidate
        await self._append("candidate", candidate.model_dump(mode="json"))

    def candidate(self, candidate_id: str) -> CandidateStrategy:
        self._ensure_loaded()
        if candidate_id not in self._candidates:
            raise KeyError(f"unknown candidate id {candidate_id!r}")
        return self._candidates[candidate_id]

    def candidate_for_proposal(self, proposal_id: str) -> CandidateStrategy | None:
        self._ensure_loaded()
        for candidate in self._candidates.values():
            if candidate.proposal_id == proposal_id:
                return candidate
        return None

    # -- versions + active pointers -----------------------------------------------------

    async def save_version(self, version: StrategyVersion) -> None:
        self._ensure_loaded()
        self._versions.setdefault(version.name, []).append(version)
        await self._append("version", version.model_dump(mode="json"))

    async def save_active(self, active: ActiveStrategy) -> None:
        self._ensure_loaded()
        self._active[active.name] = active
        await self._append("active", active.model_dump(mode="json"))

    async def save_regression(self, event: RegressionEvent) -> None:
        self._ensure_loaded()
        self._regressions.append(event)
        await self._append("regression", event.model_dump(mode="json"))

    def versions(self, name: str) -> list[StrategyVersion]:
        self._ensure_loaded()
        return list(self._versions.get(name, []))

    def active(self, name: str) -> ActiveStrategy | None:
        self._ensure_loaded()
        return self._active.get(name)

    def regressions(self) -> list[RegressionEvent]:
        self._ensure_loaded()
        return list(self._regressions)


def _json_default(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return]
    return str(value)


__all__ = ["ImprovementStore"]
