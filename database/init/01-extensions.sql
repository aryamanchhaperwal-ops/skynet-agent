-- ---------------------------------------------------------------------------
-- SKYNET database bootstrap
--
-- Executed once by the PostGIS container on first start (docker-entrypoint-
-- initdb.d). Alembic also ensures these extensions exist so that the schema
-- can be created against any PostgreSQL+PostGIS instance, not just this one.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Convenience schema for analytical helpers used by the intelligence layer.
CREATE SCHEMA IF NOT EXISTS skynet;

COMMENT ON SCHEMA skynet IS
  'Analytical helpers used by the SKYNET intelligence layer.';
