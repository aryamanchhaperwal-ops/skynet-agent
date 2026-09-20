"""Perception adapters — the God's Eye → Skynet integration boundary.

    God's Eye (existing connectors, events table)
        ↓  PerceptionAdapter (read-only)
    Skynet Observation
        ↓  Skynet Core

Adapters never write to God's Eye state and never refactor it. A failing
perception source degrades into an ``error``-kind observation (recorded,
run continues) instead of crashing the run — a prototype must survive its
sensors being down. Future adapters (web exploration, AI-to-AI) implement
the same one-method protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agency.observation import Observation
from agency.state import AgentState

_RECENT_EVENTS_SQL = """
SELECT event_id, magnitude, place, event_time, longitude, latitude, depth
FROM events
ORDER BY event_time DESC
LIMIT :limit
"""


class PerceptionAdapter(ABC):
    """Strategy interface for the OBSERVE stage."""

    #: Component name recorded on the run row.
    name: str = "perception"

    @abstractmethod
    async def observe(self, state: AgentState) -> list[Observation]:
        """Return the observations for this run's OBSERVE stage."""


class NullPerceptionAdapter(PerceptionAdapter):
    """Produces no observations (deterministic demo runs, tests)."""

    name = "none"

    async def observe(self, state: AgentState) -> list[Observation]:
        del state
        return []


class GodsEyePerceptionAdapter(PerceptionAdapter):
    """Reads the latest God's Eye events and normalizes them.

    Read-only: a single SELECT against the existing ``events`` table —
    no God's Eye code is imported or modified.
    """

    name = "gods_eye"

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        limit: int = 20,
    ) -> None:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        self._session_factory = session_factory
        self._limit = limit

    async def observe(self, state: AgentState) -> list[Observation]:
        try:
            async with self._session_factory() as session:
                rows = (
                    await session.execute(text(_RECENT_EVENTS_SQL), {"limit": self._limit})
                ).mappings().all()
        except Exception as exc:
            return [
                Observation(
                    source="gods_eye",
                    kind="error",
                    summary=f"God's Eye perception unavailable: {type(exc).__name__}: {exc}",
                    confidence=0.0,
                    run_id=state.run_id,
                    goal_id=state.goal_id,
                )
            ]

        events = [
            {
                "event_id": row["event_id"],
                "magnitude": row["magnitude"],
                "place": row["place"],
                "event_time": row["event_time"].isoformat() if row["event_time"] else None,
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "depth": row["depth"],
            }
            for row in rows
        ]
        return [
            Observation(
                source="gods_eye:usgs",
                kind="json",
                summary=f"{len(events)} recent God's Eye events",
                confidence=1.0,
                content={"events": events},
                run_id=state.run_id,
                goal_id=state.goal_id,
                metadata={"adapter": self.name, "upstream": "usgs_earthquakes"},
            )
        ]


def build_perception_adapter(
    name: str, session_factory: async_sessionmaker[AsyncSession] | None = None
) -> PerceptionAdapter:
    """Select a perception adapter by configuration name."""
    if name == "gods_eye":
        if session_factory is None:
            raise ValueError(
                "the 'gods_eye' perception adapter requires a database session factory"
            )
        return GodsEyePerceptionAdapter(session_factory)
    if name == "none":
        return NullPerceptionAdapter()
    raise ValueError(f"unknown perception adapter {name!r}")
