"""Standardized observation structure for everything Skynet perceives.

An observation is the single currency of the perception boundary: God's Eye
connectors, web exploration (Phase P4), AI-to-AI conversations (Phase P5)
and tool outputs are all eventually normalized into this shape. ``content``
is intentionally ``Any`` — structured JSON, GeoJSON, text, images (by
reference) and tool outputs are all supported without schema churn.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ObservationKind = Literal["text", "json", "geojson", "tool_output", "system", "error"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class Observation(BaseModel):
    """One thing Skynet observed, normalized and serializable."""

    id: str = Field(default_factory=_new_id)
    #: Provenance, e.g. ``gods_eye:usgs``, ``web:example.com``, ``tool:echo``.
    source: str
    kind: ObservationKind = "text"
    #: Raw payload (dict for structured data, str for text, ...).
    content: Any = None
    #: One-line human-readable digest.
    summary: str = ""
    #: 0.0–1.0 if the source quantifies confidence; None otherwise.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_at: datetime = Field(default_factory=_utcnow)
    #: Set by the loop once the owning run/goal exist.
    goal_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
