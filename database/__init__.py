"""SKYNET persistence layer.

This package is the bottom of the SKYNET dependency graph. It knows about
PostgreSQL, PostGIS and SQLAlchemy, and nothing else. It must never import
from ``connectors``, ``intelligence`` or ``backend``.

Public surface
--------------
:class:`database.base.Base`
    Declarative base class shared by every ORM model. Importing it here is
    cheap and does not open a database connection.
:func:`database.session.session_scope`
    Async context manager yielding a transactional :class:`AsyncSession`.
:mod:`database.models`
    ORM models: sources, observations, events, relationships, anomalies,
    investigations, assessments, predictions, outcomes, evaluations,
    improvement candidates, backtest runs and model versions.
:mod:`database.repositories`
    Query objects that keep SQL out of the API and intelligence layers.
"""

from database.base import Base

__all__ = ["Base"]
__version__ = "0.1.0"
