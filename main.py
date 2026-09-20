"""FastAPI application and scheduled USGS earthquake ingestion."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from connector_usgs import ingest_usgs
from database.session import dispose_engine, session_scope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


async def _ingestion_loop() -> None:
    while True:
        try:
            await ingest_usgs()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("USGS ingestion failed")
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ingestion_task = asyncio.create_task(_ingestion_loop(), name="usgs-ingestion")
    try:
        yield
    finally:
        ingestion_task.cancel()
        with suppress(asyncio.CancelledError):
            await ingestion_task
        await dispose_engine()


app = FastAPI(title="SKYNET API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    try:
        async with session_scope() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            await session.execute(text(CREATE_EVENTS_TABLE))
            count = await session.scalar(text("SELECT count(*) FROM events"))
        return {"status": "ONLINE", "database": "connected", "event_count": int(count or 0)}
    except Exception as exc:
        logger.warning("Health check database probe failed: %s", exc)
        return {"status": "ONLINE", "database": "disconnected", "event_count": 0}


@app.get("/api/v1/events")
async def events() -> dict[str, Any]:
    try:
        async with session_scope() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            await session.execute(text(CREATE_EVENTS_TABLE))
            result = await session.execute(
                text(
                    """
                    SELECT event_id, magnitude, place, event_time, longitude,
                           latitude, depth, ST_AsGeoJSON(geom)::json AS geometry,
                           properties
                    FROM events
                    ORDER BY event_time DESC
                    """
                )
            )
            rows = result.mappings().all()
    except Exception as exc:
        logger.exception("Failed to query earthquake events")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": row["event_id"],
                "geometry": row["geometry"],
                "properties": {
                    "id": row["event_id"],
                    "mag": row["magnitude"],
                    "place": row["place"],
                    "time": row["event_time"].isoformat(),
                    "longitude": row["longitude"],
                    "latitude": row["latitude"],
                    "depth": row["depth"],
                    **(row["properties"] or {}),
                },
            }
            for row in rows
        ],
    }
