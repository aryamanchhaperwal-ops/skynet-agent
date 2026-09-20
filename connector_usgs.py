"""USGS earthquake feed ingestion."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import text

from database.session import session_scope

logger = logging.getLogger(__name__)

USGS_FEED_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
)

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    magnitude DOUBLE PRECISION,
    place TEXT,
    event_time TIMESTAMPTZ NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    depth DOUBLE PRECISION,
    geom geometry(PointZ, 4326) NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _event_values(feature: Mapping[str, Any]) -> dict[str, Any] | None:
    event_id = feature.get("id")
    properties = feature.get("properties")
    geometry = feature.get("geometry")
    if not isinstance(event_id, str) or not isinstance(properties, Mapping):
        return None
    coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None

    timestamp = properties.get("time")
    if not isinstance(timestamp, (int, float)):
        return None
    longitude, latitude = coordinates[0], coordinates[1]
    depth = coordinates[2] if len(coordinates) > 2 else None
    if not all(isinstance(value, (int, float)) for value in (longitude, latitude)):
        return None

    return {
        "event_id": event_id,
        "magnitude": properties.get("mag"),
        "place": properties.get("place"),
        "event_time": datetime.fromtimestamp(timestamp / 1000, tz=UTC),
        "longitude": longitude,
        "latitude": latitude,
        "depth": depth if isinstance(depth, (int, float)) else None,
        "properties": dict(properties),
    }


async def ingest_usgs() -> int:
    """Fetch the current USGS feed and upsert its valid features.

    Returns the number of features successfully upserted. HTTP and database
    errors are intentionally allowed to propagate to the scheduler caller.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(USGS_FEED_URL)
        response.raise_for_status()
        payload = response.json()

    features = payload.get("features", []) if isinstance(payload, Mapping) else []
    if not isinstance(features, list):
        raise ValueError("USGS response has an invalid features collection")

    rows = [
        values
        for feature in features
        if isinstance(feature, Mapping)
        and (values := _event_values(feature)) is not None
    ]
    if not rows:
        return 0

    async with session_scope() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        await session.execute(text(CREATE_EVENTS_TABLE))
        await session.execute(
            text(
                """
                INSERT INTO events (
                    event_id, magnitude, place, event_time, longitude, latitude,
                    depth, geom, properties, updated_at
                ) VALUES (
                    :event_id, :magnitude, :place, :event_time, :longitude, :latitude,
                    :depth, ST_SetSRID(
                        ST_MakePoint(
                            CAST(:longitude AS double precision),
                            CAST(:latitude AS double precision),
                            COALESCE(CAST(:depth AS double precision), 0.0)
                        ), 4326
                    ),
                    CAST(:properties AS jsonb), now()
                )
                ON CONFLICT (event_id) DO UPDATE SET
                    magnitude = EXCLUDED.magnitude,
                    place = EXCLUDED.place,
                    event_time = EXCLUDED.event_time,
                    longitude = EXCLUDED.longitude,
                    latitude = EXCLUDED.latitude,
                    depth = EXCLUDED.depth,
                    geom = EXCLUDED.geom,
                    properties = EXCLUDED.properties,
                    updated_at = now()
                """
            ),
            [
                {
                    **row,
                    "properties": json.dumps(row["properties"]),
                }
                for row in rows
            ],
        )

    logger.info("Upserted %d USGS earthquake events", len(rows))
    return len(rows)
