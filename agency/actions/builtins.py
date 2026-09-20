"""Built-in actions shipped with the Skynet core foundation.

Only the minimum safe set is implemented (see docs/SKYNET_CORE.md):

- ``echo`` — deterministic demo action; proves result capture end-to-end.
- ``gods_eye_latest_events`` — read-only query of the existing God's Eye
  ``events`` table, demonstrating the God's Eye → Skynet integration
  boundary without touching any upstream code.

Web/comms/lab actions are deliberately absent in this phase; the registry
categories and feature flags they will hang off already exist.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agency.actions.base import Action, ActionContext, ActionSpec
from agency.observation import Observation

_RECENT_EVENTS_SQL = """
SELECT event_id, magnitude, place, event_time, longitude, latitude, depth
FROM events
ORDER BY event_time DESC
LIMIT :limit
"""


class EchoAction(Action):
    """Return the given text. Deterministic, harmless, always available."""

    name = "echo"
    category = "core"
    description = "Returns the 'text' parameter unchanged (deterministic demo action)."

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        value = spec.params.get("text", "")
        if not isinstance(value, str):
            raise ValueError("'text' parameter must be a string")
        return {"text": value, "length": len(value)}


class GodsEyeLatestEventsAction(Action):
    """Fetch the most recent God's Eye events (read-only).

    This is the action-side twin of the God's Eye perception adapter: both
    consume the existing ``events`` table and never write to it. Requires a
    session factory; the loop turns the output into an Observation.
    """

    name = "gods_eye_latest_events"
    category = "core"
    description = "Read-only fetch of the latest God's Eye events from the events table."

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        raw_limit = spec.params.get("limit", 20)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("'limit' parameter must be an integer") from exc
        if not 1 <= limit <= 200:
            raise ValueError("'limit' parameter must be between 1 and 200")

        async with self._session_factory() as session:
            rows = (
                await session.execute(text(_RECENT_EVENTS_SQL), {"limit": limit})
            ).mappings().all()

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
        return {
            "count": len(events),
            "events": events,
            "observation": Observation(
                source="gods_eye:usgs",
                kind="json",
                summary=f"{len(events)} recent God's Eye events",
                confidence=1.0,
                content={"events": events},
                run_id=ctx.run_id,
                goal_id=ctx.goal_id,
            ).model_dump(mode="json"),
        }
