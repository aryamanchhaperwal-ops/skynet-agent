"""Persisted mission trace stream (append-only JSONL).

The run record stores *state*; this store stores the **event stream** — the
ordered, inspectable history of what the mission actually did (states
visited, providers/actions called, evidence references, memory operations,
evaluation, improvements, failures). It is deliberately minimal and
independent of the run store so it survives process restarts and can be
read back by a fresh process or the CLI.

Two hard rules:

1. **Append-only.** One JSON object per line, appended under a lock; never
   rewritten, never re-read to append. Concurrent writers (other processes
   pausing/inspecting a run) cannot corrupt it.
2. **No secrets.** Keys that look like credentials are dropped and
   secret-shaped values are masked before the line is written, so an API
   key can never reach the trace file even if a caller passes one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("skynet.orchestrator.trace_store")

#: Metadata keys whose values must never be persisted (name-matched).
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "authorization",
    "auth",
)

#: Secret-shaped *values*: masked even when the key looks innocuous.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
)

_REDACTED = "[REDACTED]"


def redact_text(text: str) -> str:
    """Mask secret-shaped substrings in free text."""
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def redact_payload(value: Any) -> Any:
    """Recursively drop/redact secret-shaped keys and values.

    Name-matched keys are replaced with ``[REDACTED]`` (the key is kept so
    the trace still shows *that* a credential was present, never its value);
    strings are then scanned for secret-shaped substrings.
    """
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS):
                result[key] = _REDACTED
            else:
                result[key] = redact_payload(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class TraceRecord(BaseModel):
    """One persisted mission trace event."""

    run_id: str = ""
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Stage/state the mission was in when the event was emitted.
    state: str = ""
    #: Emitting component (e.g. adapter class or subsystem name).
    component: str = ""
    #: Action/tool named by the event, when there is one.
    action: str = ""
    #: Provider named by the event, when there is one.
    provider: str = ""
    #: Event status/outcome marker, when there is one.
    status: str = ""
    #: Free-text error, when applicable.
    error: str = ""
    #: Sanitized event payload (secrets already removed).
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunTraceStore:
    """Append-only, restart-safe persisted trace stream."""

    def __init__(self, path: str | Path | None = "data/autonomous_run_traces.jsonl") -> None:
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path | None:
        return self._path

    @staticmethod
    def _record_from_event(
        event_type: str, payload: dict[str, Any]
    ) -> TraceRecord:
        sanitized = redact_payload(payload)
        if not isinstance(sanitized, dict):  # pragma: no cover - defensive
            sanitized = {"metadata": sanitized}
        run_id = str(sanitized.get("run_id") or sanitized.get("autonomous_run_id") or "")
        state = str(sanitized.get("stage") or "")
        component = str(
            sanitized.get("component")
            or sanitized.get("adapter")
            or sanitized.get("subsystem")
            or ""
        )
        action = str(
            sanitized.get("action")
            or sanitized.get("step_type")
            or sanitized.get("core_run_id")
            or ""
        )
        provider = str(sanitized.get("provider") or "")
        status = str(
            sanitized.get("status")
            or sanitized.get("evidence_class")
            or sanitized.get("kind")
            or ""
        )
        error = str(sanitized.get("error") or "")
        return TraceRecord(
            run_id=run_id,
            event_type=str(event_type),
            state=state,
            component=component,
            action=action,
            provider=provider,
            status=status,
            error=error,
            metadata=sanitized,
        )

    async def append(self, event_type: str, payload: dict[str, Any] | None = None) -> TraceRecord:
        """Persist one event (redacted). A write failure is logged, not raised."""
        record = self._record_from_event(event_type, dict(payload or {}))
        if self._path is not None:
            line = json.dumps(record.model_dump(mode="json")) + "\n"
            path = self._path

            def _write() -> None:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)

            async with self._lock:
                try:
                    await asyncio.to_thread(_write)
                except Exception:
                    logger.exception("trace store append failed for %s", event_type)
        return record

    def _read_sync(self, run_id: str | None) -> list[TraceRecord]:
        if self._path is None or not self._path.exists():
            return []
        records: list[TraceRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = TraceRecord.model_validate_json(line)
            except Exception:
                logger.warning("skipping malformed trace record in %s", self._path)
                continue
            if run_id is None or record.run_id == run_id:
                records.append(record)
        return records

    async def read(self, run_id: str | None = None) -> list[TraceRecord]:
        """Read the persisted stream (optionally for one run).

        A fresh store instance reading the same file is exactly the
        process-restart case: nothing is cached in memory.
        """
        return await asyncio.to_thread(self._read_sync, run_id)

    async def runs(self) -> list[str]:
        """Distinct run ids present in the stream (insertion order)."""
        records = await self.read()
        seen: list[str] = []
        for record in records:
            if record.run_id and record.run_id not in seen:
                seen.append(record.run_id)
        return seen


__all__ = [
    "RunTraceStore",
    "TraceRecord",
    "redact_payload",
    "redact_text",
]
