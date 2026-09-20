"""Configuration owned by the persistence layer.

Every SKYNET layer reads the same root ``.env`` file but only declares the
settings it actually uses. This keeps the dependency graph acyclic (nothing
here imports ``connectors``, ``intelligence`` or ``backend``) while still
giving each layer validation on its own inputs.

Environment variables are prefixed ``SKYNET_``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root - ``database/settings.py`` -> parents[1].
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ENV_FILE: Path = REPO_ROOT / ".env"

DEFAULT_DATABASE_URL = "postgresql+asyncpg://skynet:skynet_local_dev@localhost:5433/skynet"


class DatabaseSettings(BaseSettings):
    """PostgreSQL/PostGIS connection and pooling configuration."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="SKYNET_",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(
        default=DEFAULT_DATABASE_URL,
        description="Async SQLAlchemy URL. Must use the asyncpg driver.",
    )
    db_echo: bool = Field(default=False, description="Log every emitted SQL statement.")
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_recycle_seconds: int = Field(
        default=1800,
        ge=-1,
        description="Recycle pooled connections after this many seconds (-1 disables).",
    )
    db_statement_timeout_ms: int = Field(
        default=30000,
        ge=0,
        description="Server-side statement timeout; 0 disables the guard.",
    )

    #: Ingested events older than this are eligible for pruning. Keeps the
    #: globe's working set bounded without deleting provenance.
    event_retention_days: int = Field(default=30, ge=1, le=3650)

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        if value.startswith("postgres://"):
            # Heroku-style URLs are common in the wild; transparently upgrade.
            return value.replace("postgres://", "postgresql+asyncpg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "SKYNET_DATABASE_URL must be a PostgreSQL URL using the asyncpg "
                f"driver (postgresql+asyncpg://...); received {value.split(':', 1)[0]!r}."
            )
        return value

    @property
    def sync_database_url(self) -> str:
        """URL variant for tooling that cannot use asyncpg (Alembic offline mode)."""
        return self.database_url.replace("+asyncpg", "+psycopg2")


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Return the process-wide, cached database settings instance."""
    return DatabaseSettings()
