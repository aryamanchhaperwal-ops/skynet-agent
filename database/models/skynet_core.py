"""ORM models for the Skynet core foundation.

Minimal persistence for goals, runs, trace events and experiences. These
tables are purely additive: nothing here touches the God's Eye ``events``
table, and God's Eye code never imports this module (the dependency rule
``database <- agency`` is one-way).

Schema management: ``agency.storage.ensure_core_schema`` creates these
tables idempotently (``CREATE TABLE IF NOT EXISTS`` semantics via
``Base.metadata.create_all`` restricted to this module's tables). Alembic
migrations take over schema management in a later phase; the model
definitions here are migration-ready (named constraints, tz-aware UTC).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base, CreatedAtMixin, TimestampMixin


class Goal(Base, TimestampMixin):
    """A unit of intent Skynet pursues.

    Status lifecycle (enforced in ``agency.goals.GoalManager``, not here):
    ``pending -> running -> completed | failed | cancelled``, with
    ``paused`` reachable from ``pending``/``running``. ``completed``,
    ``failed`` and ``cancelled`` are terminal.
    """

    __tablename__ = "goals"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("goals.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    #: Who issued the goal: ``human`` or ``self`` (a self-subgoal).
    origin: Mapped[str] = mapped_column(String(16), default="human", nullable=False)
    #: Structured success criteria consumed by evaluators (Phase P6+).
    success_criteria: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    #: Free-form planner hints (e.g. an explicit step list for a run).
    params: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    #: Arbitrary caller metadata (provenance, tags, correlations).
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class Run(Base, TimestampMixin):
    """One execution of the Skynet core loop for a goal."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("goals.id", ondelete="SET NULL"), index=True, nullable=True
    )
    #: ``running | completed | failed | cancelled``
    status: Mapped[str] = mapped_column(String(16), default="running", index=True, nullable=False)
    #: Component names used for this run (replaced per version in P7+).
    planner: Mapped[str] = mapped_column(String(64), default="deterministic", nullable=False)
    perception: Mapped[str] = mapped_column(String(64), default="none", nullable=False)
    #: Serialized :class:`agency.actions.base.ActionSpec` list.
    steps_planned: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    steps_executed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    #: Structured run result (summary payload produced by the loop).
    result: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RunEvent(Base, CreatedAtMixin):
    """One structured lifecycle trace event (see ``agency.trace``).

    Event ordering within a run is guaranteed by the ``(run_id, seq)``
    unique constraint; ``seq`` is assigned by the tracer.
    """

    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    #: A ``TraceEventType`` value (RUN_STARTED, ACTION_COMPLETED, ...).
    event_type: Mapped[str] = mapped_column(String(40), index=True, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class Experience(Base, CreatedAtMixin):
    """A structured experience record — the raw material for future learning.

    Kinds: ``observation | decision | action | evaluation | error | summary``.
    Outcomes: ``success | failure | neutral``.
    """

    __tablename__ = "experiences"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("runs.id", ondelete="SET NULL"), index=True, nullable=True
    )
    goal_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("goals.id", ondelete="SET NULL"), index=True, nullable=True
    )
    kind: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), default="neutral", nullable=False)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
