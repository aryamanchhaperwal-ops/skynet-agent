"""Structured execution tracing for Skynet runs.

Every run has a unique ``run_id`` and emits an ordered, typed event stream
(RUN_STARTED → ... → RUN_COMPLETED/RUN_FAILED). Sinks decide where events
go: the standard ``logging`` framework as JSON lines (no duplicate logging
framework), an in-memory list for tests, and the ``run_events`` table for
inspection via SQL/API later.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database.models import RunEvent as RunEventORM

logger = logging.getLogger("skynet.trace")


class TraceEventType(StrEnum):
    """Lifecycle events of a Skynet run."""

    RUN_STARTED = "RUN_STARTED"
    GOAL_CREATED = "GOAL_CREATED"
    OBSERVATION_RECEIVED = "OBSERVATION_RECEIVED"
    PLAN_CREATED = "PLAN_CREATED"
    ACTION_SELECTED = "ACTION_SELECTED"
    ACTION_STARTED = "ACTION_STARTED"
    ACTION_COMPLETED = "ACTION_COMPLETED"
    ACTION_FAILED = "ACTION_FAILED"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    # -- Web exploration (Phase P4) -------------------------------------------
    # Emitted by ``agency.web`` actions through the run's ``Trace`` facade so
    # web operations share one event stream, one run_id and one seq ordering.
    WEB_SEARCH_STARTED = "WEB_SEARCH_STARTED"
    WEB_SEARCH_COMPLETED = "WEB_SEARCH_COMPLETED"
    WEB_SEARCH_FAILED = "WEB_SEARCH_FAILED"
    WEB_PAGE_FETCH_STARTED = "WEB_PAGE_FETCH_STARTED"
    WEB_PAGE_FETCH_COMPLETED = "WEB_PAGE_FETCH_COMPLETED"
    WEB_PAGE_FETCH_FAILED = "WEB_PAGE_FETCH_FAILED"
    WEB_CONTENT_EXTRACTED = "WEB_CONTENT_EXTRACTED"
    WEB_SOURCE_RECORDED = "WEB_SOURCE_RECORDED"
    RESEARCH_STARTED = "RESEARCH_STARTED"
    RESEARCH_COMPLETED = "RESEARCH_COMPLETED"
    RESEARCH_FAILED = "RESEARCH_FAILED"
    # -- AI ↔ AI communication (Phase P5) --------------------------------------
    # Emitted by ``agency.comms`` actions through the run's ``Trace`` facade so
    # external-AI interactions share one event stream, one run_id, one ordering.
    # Payloads never contain credentials or API keys.
    AI_PARTICIPANT_DISCOVERED = "AI_PARTICIPANT_DISCOVERED"
    AI_CONVERSATION_STARTED = "AI_CONVERSATION_STARTED"
    AI_MESSAGE_SENT = "AI_MESSAGE_SENT"
    AI_RESPONSE_RECEIVED = "AI_RESPONSE_RECEIVED"
    AI_CONVERSATION_CONTINUED = "AI_CONVERSATION_CONTINUED"
    AI_RESPONSE_ANALYZED = "AI_RESPONSE_ANALYZED"
    AI_CONVERSATION_COMPLETED = "AI_CONVERSATION_COMPLETED"
    AI_CONVERSATION_FAILED = "AI_CONVERSATION_FAILED"
    # -- Intelligence + long-term memory (Phase P2) ----------------------------
    # Memory events share the run's event stream; payloads carry ids and
    # counts, never full memory bodies (those live in the memory store).
    MEMORY_RECALLED = "MEMORY_RECALLED"
    MEMORY_STORED = "MEMORY_STORED"
    # -- Experimentation (Phase P6) ---------------------------------------------
    # Emitted by ``agency.experiments`` through the run's ``Trace`` facade;
    # payloads carry experiment/trial ids and metric values only.
    EXPERIMENT_CREATED = "EXPERIMENT_CREATED"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    TRIAL_STARTED = "TRIAL_STARTED"
    TRIAL_COMPLETED = "TRIAL_COMPLETED"
    TRIAL_FAILED = "TRIAL_FAILED"
    METRIC_RECORDED = "METRIC_RECORDED"
    BASELINE_MEASURED = "BASELINE_MEASURED"
    CANDIDATE_MEASURED = "CANDIDATE_MEASURED"
    EXPERIMENT_COMPARED = "EXPERIMENT_COMPARED"
    EXPERIMENT_EVALUATED = "EXPERIMENT_EVALUATED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    EXPERIMENT_FAILED = "EXPERIMENT_FAILED"
    EXPERIMENT_CANCELLED = "EXPERIMENT_CANCELLED"
    # -- Self-improvement (Phase P7) ---------------------------------------------
    # Emitted by ``agency.improve`` through the run's ``Trace`` facade; payloads
    # carry weakness/proposal/candidate/experiment ids and metric values only —
    # never full memory bodies or external content.
    WEAKNESS_DETECTED = "WEAKNESS_DETECTED"
    HYPOTHESIS_CREATED = "HYPOTHESIS_CREATED"
    IMPROVEMENT_PROPOSED = "IMPROVEMENT_PROPOSED"
    CANDIDATE_CREATED = "CANDIDATE_CREATED"
    CANDIDATE_TEST_STARTED = "CANDIDATE_TEST_STARTED"
    CANDIDATE_TEST_COMPLETED = "CANDIDATE_TEST_COMPLETED"
    BENCHMARK_STARTED = "BENCHMARK_STARTED"
    BENCHMARK_COMPLETED = "BENCHMARK_COMPLETED"
    ACCEPTANCE_EVALUATED = "ACCEPTANCE_EVALUATED"
    IMPROVEMENT_ACCEPTED = "IMPROVEMENT_ACCEPTED"
    IMPROVEMENT_REJECTED = "IMPROVEMENT_REJECTED"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    IMPROVEMENT_APPLIED = "IMPROVEMENT_APPLIED"
    IMPROVEMENT_ROLLED_BACK = "IMPROVEMENT_ROLLED_BACK"
    IMPROVEMENT_MONITORED = "IMPROVEMENT_MONITORED"


class TraceEvent(BaseModel):
    """One ordered trace event."""

    run_id: str
    seq: int = Field(ge=1)
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict = Field(default_factory=dict)


class TraceSink(ABC):
    """Destination for trace events."""

    @abstractmethod
    async def emit(self, event: TraceEvent) -> None:
        """Consume one event."""


class LoggingTraceSink(TraceSink):
    """Emits events as structured JSON lines on the ``skynet.trace`` logger."""

    async def emit(self, event: TraceEvent) -> None:
        logger.info(json.dumps(event.model_dump(mode="json"), default=str))


class MemoryTraceSink(TraceSink):
    """Collects events in-process (tests, dry runs)."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event.model_copy(deep=True))

    @property
    def types(self) -> list[str]:
        return [event.event_type for event in self.events]


class DatabaseTraceSink(TraceSink):
    """Persists events into the ``run_events`` table."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def emit(self, event: TraceEvent) -> None:
        row = RunEventORM(
            run_id=uuid.UUID(event.run_id),
            seq=event.seq,
            event_type=event.event_type,
            payload=event.payload,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()


class Trace:
    """Per-run facade: assigns sequence numbers and fans out to sinks."""

    def __init__(self, run_id: str, sinks: Sequence[TraceSink]) -> None:
        self.run_id = run_id
        self._sinks = list(sinks)
        self._seq = 0

    @property
    def event_count(self) -> int:
        return self._seq

    async def emit(self, event_type: TraceEventType | str, **payload: object) -> TraceEvent:
        """Emit one event. Accepts a ``TraceEventType`` or a plain string so
        feature modules (e.g. ``agency.web``) can define their own constants
        without a second event system."""
        self._seq += 1
        event = TraceEvent(
            run_id=self.run_id,
            seq=self._seq,
            event_type=(
                event_type.value if isinstance(event_type, TraceEventType) else str(event_type)
            ),
            payload=dict(payload),
        )
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception:
                logger.exception(
                    "trace sink %s failed for run %s event %s",
                    type(sink).__name__,
                    self.run_id,
                    event.event_type,
                )
        return event
