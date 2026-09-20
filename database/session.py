"""Async SQLAlchemy engine, session factory and health probes.

The engine is created lazily. Importing this module must never open a
connection: Alembic, the test suite and the FastAPI app all import it in
contexts where the database may legitimately be unreachable.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from database.settings import DatabaseSettings, get_database_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    settings: DatabaseSettings = get_database_settings()

    connect_args: dict[str, object] = {}
    if settings.db_statement_timeout_ms > 0:
        # Enforced server-side so a runaway analytical query cannot pin a
        # pooled connection indefinitely. 0 disables the guard entirely.
        connect_args["server_settings"] = {
            "statement_timeout": str(settings.db_statement_timeout_ms),
            "application_name": "skynet",
        }

    engine = create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle_seconds,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    logger.debug(
        "Created async engine (pool_size=%d, max_overflow=%d, statement_timeout_ms=%d)",
        settings.db_pool_size,
        settings.db_max_overflow,
        settings.db_statement_timeout_ms,
    )
    return engine


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session scope.

    Commits on clean exit, rolls back on any exception, and always closes.
    This is the entry point for background jobs (ingestion, anomaly scans).
    """
    factory = get_sessionmaker()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session.

    Read-only endpoints commit nothing but still benefit from a bounded,
    pooled connection. A failed request rolls back rather than leaking a
    half-applied write.
    """
    factory = get_sessionmaker()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def dispose_engine() -> None:
    """Dispose of pooled connections. Called on application shutdown."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        logger.debug("Disposed database engine")
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


async def database_health() -> dict[str, object]:
    """Probe connectivity and report server/PostGIS capabilities.

    Never raises: the health endpoint must be able to report a broken
    database rather than failing to respond.
    """
    try:
        async with get_engine().connect() as connection:
            row = (
                await connection.execute(
                    text(
                        "SELECT current_database() AS db,"
                        "       version() AS server_version,"
                        "       (SELECT extversion FROM pg_extension WHERE extname = 'postgis')"
                        "           AS postgis_version,"
                        "       now() AS server_time"
                    )
                )
            ).mappings().one()
            return {
                "reachable": True,
                "database": row["db"],
                # The full version banner is noisy; keep the first clause only.
                "server_version": str(row["server_version"]).split(",")[0],
                "postgis_version": row["postgis_version"],
                "server_time": row["server_time"].isoformat()
                if row["server_time"]
                else None,
            }
    except Exception as exc:  # noqa: BLE001 - health must never propagate
        logger.warning("Database health probe failed: %s", exc)
        return {"reachable": False, "error": str(exc)}
