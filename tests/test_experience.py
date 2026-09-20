"""Experience layer tests: records, sinks, buffer/flush semantics."""

from __future__ import annotations

import pytest

from agency.experience import (
    DatabaseExperienceSink,
    ExperienceRecord,
    ExperienceRecorder,
    InMemoryExperienceSink,
)


@pytest.mark.asyncio
async def test_record_defaults() -> None:
    record = ExperienceRecord(kind="observation", summary="saw things")
    assert record.id
    assert record.outcome == "neutral"
    assert record.run_id is None and record.goal_id is None
    assert record.payload == {}
    assert record.confidence is None
    assert record.created_at.tzinfo is not None


def test_recorder_buffers_without_saving() -> None:
    sink = InMemoryExperienceSink()
    recorder = ExperienceRecorder(sink)
    recorder.record(run_id="r1", goal_id="g1", kind="action", summary="did a thing")
    assert recorder.pending == 1
    assert sink.records == []


@pytest.mark.asyncio
async def test_flush_persists_and_clears_buffer() -> None:
    sink = InMemoryExperienceSink()
    recorder = ExperienceRecorder(sink)
    recorder.record(run_id="r1", goal_id="g1", kind="observation", summary="o")
    recorder.record(run_id="r1", goal_id="g1", kind="action", summary="a", outcome="success")
    persisted = await recorder.flush()
    assert persisted == 2
    assert recorder.pending == 0
    assert len(sink.records) == 2
    assert [r.kind for r in sink.records] == ["observation", "action"]


@pytest.mark.asyncio
async def test_flush_empty_buffer_is_zero() -> None:
    recorder = ExperienceRecorder(InMemoryExperienceSink())
    assert await recorder.flush() == 0


@pytest.mark.asyncio
async def test_recorder_propagates_sink_failures() -> None:
    """Pin the contract: the recorder itself raises on sink failure; it is
    the LOOP's _safe_flush that absorbs sink outages (see test_loop.py)."""

    class FailingSink(InMemoryExperienceSink):
        async def save(self, records):
            raise RuntimeError("sink down")

    recorder = ExperienceRecorder(FailingSink())
    recorder.record(run_id="r", goal_id=None, kind="error", summary="x")
    with pytest.raises(RuntimeError, match="sink down"):
        await recorder.flush()


@pytest.mark.asyncio
async def test_records_are_deep_copied_into_sink() -> None:
    sink = InMemoryExperienceSink()
    recorder = ExperienceRecorder(sink)
    entry = recorder.record(run_id="r", goal_id=None, kind="decision", summary="d",
                           payload={"n": 1})
    entry.payload["n"] = 999  # mutate after recording
    await recorder.flush()
    assert sink.records[0].payload["n"] == 1


@pytest.mark.asyncio
async def test_kind_and_outcome_validation() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExperienceRecord(kind="vibes", summary="s")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExperienceRecord(kind="action", summary="s", outcome="meh")  # type: ignore[arg-type]


def test_database_sink_is_importable_and_lazy() -> None:
    # Constructing the sink must not open any connection.
    sink = DatabaseExperienceSink(session_factory=None)  # type: ignore[arg-type]
    assert sink is not None
