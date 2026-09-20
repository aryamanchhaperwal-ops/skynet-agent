"""Shared fixtures for the Skynet core test suite.

Tests are DB-free unless marked ``db`` (see pyproject markers). Settings
fixtures pin explicit values so a developer's local ``.env`` cannot skew
assertions (explicit kwargs win over env files in pydantic-settings).
"""

from __future__ import annotations

from typing import Any

import pytest

from agency.config import SkynetSettings


@pytest.fixture
def memory_settings() -> SkynetSettings:
    """DB-free settings: memory storage, no perception, generous budget."""
    return SkynetSettings(
        storage_backend="memory",
        perception_adapter="none",
        max_run_seconds=30,
        max_steps_per_run=10,
    )


class FakeRow:
    """Mimics one row from ``session.execute(...).mappings().all()``."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


class FakeResult:
    def __init__(self, rows: list[FakeRow]) -> None:
        self._rows = rows

    def mappings(self) -> Any:
        return self

    def all(self) -> list[FakeRow]:
        return self._rows


class FakeSession:
    """Minimal async session stand-in for perception normalization tests."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [FakeRow(row) for row in rows]

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, *_args: object, **_kwargs: object) -> FakeResult:
        return FakeResult(self._rows)


class FakeSessionFactory:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls = 0

    def __call__(self) -> FakeSession:
        self.calls += 1
        return FakeSession(self._rows)


class BrokenSessionFactory:
    """Factory whose sessions always explode — simulates a dead database."""

    def __call__(self) -> Any:
        class _Broken:
            async def __aenter__(self) -> None:
                raise RuntimeError("database is down")

            async def __aexit__(self, *args: object) -> None:
                return None

        return _Broken()


@pytest.fixture
def fake_event_rows() -> list[dict[str, Any]]:
    from datetime import UTC, datetime

    moment = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
    return [
        {
            "event_id": "usgs-test-1",
            "magnitude": 5.4,
            "place": "Test Place A",
            "event_time": moment,
            "longitude": 10.0,
            "latitude": 20.0,
            "depth": 8.0,
        },
        {
            "event_id": "usgs-test-2",
            "magnitude": 2.1,
            "place": "Test Place B",
            "event_time": moment,
            "longitude": -10.0,
            "latitude": -20.0,
            "depth": None,
        },
    ]


@pytest.fixture
def fake_event_factory(fake_event_rows: list[dict[str, Any]]) -> FakeSessionFactory:
    """Session factory serving the canned God's Eye event rows."""
    return FakeSessionFactory(fake_event_rows)


@pytest.fixture
async def db_session_factory():
    """Real session factory for db-marked tests; skips when PostgreSQL is
    unreachable so the suite stays green on machines without the container."""
    from sqlalchemy import text

    from database.session import get_sessionmaker

    session_factory = get_sessionmaker()
    try:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("PostgreSQL unavailable (db-marked test)")
    return session_factory
