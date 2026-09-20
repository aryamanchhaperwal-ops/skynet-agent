"""SKYNET ORM models.

Every model lives in its own module and imports only from ``database.base``
plus SQLAlchemy. The God's Eye ``events`` table is intentionally **not**
modeled here: it predates the ORM and keeps its raw-SQL management path
until the Alembic phase.

Public surface
--------------
:class:`database.models.skynet_core.Goal`
    Goal lifecycle records (Skynet core).
:class:`database.models.skynet_core.Run`
    One execution of the Skynet core loop.
:class:`database.models.skynet_core.RunEvent`
    Structured per-run trace events.
:class:`database.models.skynet_core.Experience`
    Structured experience records for future learning.
"""

from database.models.skynet_core import Experience, Goal, Run, RunEvent

__all__ = ["Experience", "Goal", "Run", "RunEvent"]
