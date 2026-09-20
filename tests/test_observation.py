"""Observation model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agency.observation import Observation


def test_defaults_created() -> None:
    observation = Observation(source="gods_eye:usgs", content={"a": 1})
    assert observation.id
    assert observation.kind == "text"
    assert observation.observed_at.tzinfo is not None
    assert observation.metadata == {}


def test_structured_and_text_content() -> None:
    structured = Observation(source="tool:x", kind="json", content={"events": [1, 2]})
    textual = Observation(source="web:example.com", kind="text", content="hello")
    assert structured.content["events"] == [1, 2]
    assert textual.content == "hello"


def test_confidence_bounds_enforced() -> None:
    Observation(source="s", confidence=0.0)
    Observation(source="s", confidence=1.0)
    with pytest.raises(ValidationError):
        Observation(source="s", confidence=1.5)
    with pytest.raises(ValidationError):
        Observation(source="s", confidence=-0.1)
    assert Observation(source="s", confidence=None).confidence is None


def test_serialization_round_trip() -> None:
    observation = Observation(
        source="gods_eye:usgs",
        kind="geojson",
        content={"type": "Feature"},
        summary="one feature",
        confidence=0.9,
        metadata={"upstream": "usgs"},
    )
    clone = Observation.model_validate(observation.model_dump(mode="json"))
    assert clone.id == observation.id
    assert clone.kind == "geojson"
    assert clone.confidence == 0.9
    assert clone.metadata == {"upstream": "usgs"}
    assert clone.observed_at == observation.observed_at


def test_error_kind_observation() -> None:
    observation = Observation(
        source="gods_eye", kind="error", summary="sensor down", confidence=0.0
    )
    assert observation.kind == "error"
