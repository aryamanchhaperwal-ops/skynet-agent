-- USGS earthquake events consumed by the API and map clients.
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
);

CREATE INDEX IF NOT EXISTS ix_events_event_time ON events (event_time DESC);
CREATE INDEX IF NOT EXISTS ix_events_geom ON events USING GIST (geom);
