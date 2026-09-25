"""Run store — persistent, resumable autonomous runs (spec §3/§15).

One JSONL file (``data/autonomous_runs.jsonl`` by default). Unlike the
P7 improvement store (which caches at first access), this store **re-reads
the file for every operation**: pause, resume and cancel are explicitly
cross-process operations — an operator in a second terminal must be able
to pause a run this process is executing. Writes go through a lock and
an atomic temp-file replace; the in-memory snapshot exists only inside
an operation's critical section.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from agency.orchestrator.models import AutonomousRun, RunStatus

logger = logging.getLogger("skynet.orchestrator.store")


class RunStore:
    """Persistent store of autonomous runs (cross-process safe JSONL)."""

    def __init__(self, path: str | Path | None = "data/autonomous_runs.jsonl") -> None:
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # -- internal -----------------------------------------------------------------

    def _read_all_sync(self) -> dict[str, AutonomousRun]:
        """Fresh read of every run record from disk (sync; run in a thread)."""
        runs: dict[str, AutonomousRun] = {}
        if self._path is None or not self._path.exists():
            return runs
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = AutonomousRun.model_validate_json(line)
                runs[record.id] = record
            except Exception:
                logger.warning("skipping malformed run record in %s", self._path)
        return runs

    async def _write_all(self, records: dict[str, AutonomousRun]) -> None:
        if self._path is None:
            return
        lines = [
            json.dumps(run.model_dump(mode="json"))
            for run in sorted(records.values(), key=lambda r: r.created_at)
        ]
        payload = "\n".join(lines) + ("\n" if lines else "")
        path = self._path

        def _write() -> None:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)

        async with self._lock:
            await asyncio.to_thread(_write)

    # -- public ---------------------------------------------------------------------

    async def save(self, run: AutonomousRun) -> AutonomousRun:
        runs = await asyncio.to_thread(self._read_all_sync)
        runs[run.id] = run.model_copy(deep=True)
        await self._write_all(runs)
        return run

    async def get(self, run_id: str) -> AutonomousRun | None:
        runs = await asyncio.to_thread(self._read_all_sync)
        run = runs.get(run_id)
        return run.model_copy(deep=True) if run else None

    async def list(self, status: str | None = None) -> list[AutonomousRun]:
        runs = await asyncio.to_thread(self._read_all_sync)
        items = sorted(runs.values(), key=lambda r: r.created_at)
        if status:
            items = [r for r in items if r.status.value == status]
        return items

    async def find_resumable(self) -> list[AutonomousRun]:
        """Runs a restart can pick up again (paused or waiting)."""
        runs = await asyncio.to_thread(self._read_all_sync)
        return [
            r.model_copy(deep=True)
            for r in runs.values()
            if r.status in (RunStatus.PAUSED, RunStatus.WAITING)
        ]

    async def _transition(
        self, run_id: str, status: RunStatus
    ) -> AutonomousRun | None:
        runs = await asyncio.to_thread(self._read_all_sync)
        run = runs.get(run_id)
        if run is None or run.is_terminal:
            return None
        run.status = status
        run.touch()
        await self._write_all(runs)
        return run

    async def cancel(self, run_id: str) -> AutonomousRun | None:
        """Cross-process cancel: flips a live run to CANCELLED on disk."""
        return await self._transition(run_id, RunStatus.CANCELLED)

    async def pause(self, run_id: str) -> AutonomousRun | None:
        return await self._transition(run_id, RunStatus.PAUSED)

    async def resume(self, run_id: str) -> AutonomousRun | None:
        runs = await asyncio.to_thread(self._read_all_sync)
        run = runs.get(run_id)
        if run is None or run.status not in (RunStatus.PAUSED, RunStatus.WAITING):
            return None
        run.status = RunStatus.RUNNING
        run.touch()
        await self._write_all(runs)
        return run


__all__ = ["RunStore"]
