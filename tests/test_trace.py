"""Trace system tests: event types, ordering, sinks, fanout, failure isolation."""

from __future__ import annotations

import json
import logging

import pytest

from agency.trace import (
    DatabaseTraceSink,
    LoggingTraceSink,
    MemoryTraceSink,
    Trace,
    TraceEvent,
    TraceEventType,
    TraceSink,
)

REQUIRED_EVENT_NAMES = {
    "RUN_STARTED",
    "GOAL_CREATED",
    "OBSERVATION_RECEIVED",
    "PLAN_CREATED",
    "ACTION_SELECTED",
    "ACTION_STARTED",
    "ACTION_COMPLETED",
    "ACTION_FAILED",
    "EVALUATION_COMPLETED",
    "RUN_COMPLETED",
    "RUN_FAILED",
}


def test_all_required_event_types_exist() -> None:
    names = {member.value for member in TraceEventType}
    assert REQUIRED_EVENT_NAMES <= names


@pytest.mark.asyncio
async def test_seq_starts_at_one_and_increments() -> None:
    trace = Trace("run-1", [MemoryTraceSink()])
    first = await trace.emit(TraceEventType.RUN_STARTED)
    second = await trace.emit(TraceEventType.GOAL_CREATED, goal_id="g1")
    assert first.seq == 1
    assert second.seq == 2
    assert trace.event_count == 2


@pytest.mark.asyncio
async def test_event_carries_payload_and_timestamp() -> None:
    trace = Trace("run-1", [])
    event = await trace.emit(TraceEventType.ACTION_FAILED, error="boom", action_id="a1")
    assert event.payload == {"error": "boom", "action_id": "a1"}
    assert event.timestamp.tzinfo is not None
    assert event.run_id == "run-1"


@pytest.mark.asyncio
async def test_memory_sink_records_events() -> None:
    sink = MemoryTraceSink()
    trace = Trace("run-1", [sink])
    await trace.emit(TraceEventType.RUN_STARTED)
    await trace.emit(TraceEventType.RUN_COMPLETED)
    assert sink.types == ["RUN_STARTED", "RUN_COMPLETED"]
    assert len(sink.events) == 2


@pytest.mark.asyncio
async def test_fanout_reaches_all_sinks() -> None:
    sink_a, sink_b = MemoryTraceSink(), MemoryTraceSink()
    trace = Trace("run-1", [sink_a, sink_b])
    await trace.emit(TraceEventType.RUN_STARTED)
    assert len(sink_a.events) == 1 and len(sink_b.events) == 1


@pytest.mark.asyncio
async def test_broken_sink_does_not_break_the_run(caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingSink(TraceSink):
        async def emit(self, event: TraceEvent) -> None:
            raise RuntimeError("sink exploded")

    healthy = MemoryTraceSink()
    trace = Trace("run-1", [ExplodingSink(), healthy])
    with caplog.at_level(logging.ERROR, logger="skynet.trace"):
        event = await trace.emit(TraceEventType.RUN_STARTED)
    assert event.seq == 1
    assert len(healthy.events) == 1  # healthy sink still received it
    # The isolation log line names the broken sink (exception detail rides
    # in the traceback, not in record.message).
    assert any("ExplodingSink failed" in record.message for record in caplog.records)
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.asyncio
async def test_logging_sink_emits_json_line(caplog: pytest.LogCaptureFixture) -> None:
    sink = LoggingTraceSink()
    trace = Trace("run-1", [sink])
    with caplog.at_level(logging.INFO, logger="skynet.trace"):
        await trace.emit(TraceEventType.GOAL_CREATED, goal_id="g42")
    record = caplog.records[-1]
    payload = json.loads(record.message)
    assert payload["event_type"] == "GOAL_CREATED"
    assert payload["run_id"] == "run-1"
    assert payload["seq"] == 1
    assert payload["payload"] == {"goal_id": "g42"}


def test_database_sink_is_lazy() -> None:
    # Construction must not open connections or require a real engine.
    sink = DatabaseTraceSink(session_factory=None)  # type: ignore[arg-type]
    assert sink is not None


def test_trace_event_model_defaults() -> None:
    event = TraceEvent(run_id="r", seq=1, event_type="RUN_STARTED")
    assert event.payload == {}
    assert event.timestamp.tzinfo is not None
