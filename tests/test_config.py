"""Configuration layer tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agency.config import SkynetSettings, get_skynet_settings


def test_defaults_are_safe() -> None:
    settings = SkynetSettings()
    assert settings.storage_backend == "database"
    assert settings.default_provider == "deterministic"
    assert settings.perception_adapter == "gods_eye"
    assert settings.max_steps_per_run == 10
    assert settings.max_run_seconds == 120


def test_capability_flags_ship_dark() -> None:
    settings = SkynetSettings()
    assert settings.enable_web_tools is False
    assert settings.enable_external_comms is False
    assert settings.enable_self_improvement is False


def test_action_allowlist_parsing_tolerates_whitespace() -> None:
    settings = SkynetSettings(enabled_actions=" echo , gods_eye_latest_events ,, ")
    assert settings.action_allowlist == frozenset({"echo", "gods_eye_latest_events"})


def test_action_allowlist_empty_means_refuse_everything() -> None:
    settings = SkynetSettings(enabled_actions="")
    assert settings.action_allowlist == frozenset()


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        SkynetSettings(log_level="NOT_A_LEVEL")


def test_valid_log_levels_accepted_uppercase() -> None:
    assert SkynetSettings(log_level="debug").log_level == "DEBUG"


def test_invalid_storage_backend_rejected() -> None:
    with pytest.raises(ValidationError):
        SkynetSettings(storage_backend="spreadsheet")  # type: ignore[arg-type]


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYNET_MAX_STEPS_PER_RUN", "3")
    monkeypatch.setenv("SKYNET_ENABLE_WEB_TOOLS", "true")
    settings = SkynetSettings()
    assert settings.max_steps_per_run == 3
    assert settings.enable_web_tools is True


def test_explicit_kwargs_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYNET_MAX_STEPS_PER_RUN", "3")
    assert SkynetSettings(max_steps_per_run=7).max_steps_per_run == 7


def test_log_level_names_mapping_used() -> None:
    for level in ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"):
        assert SkynetSettings(log_level=level).log_level == level


def test_cached_accessor_returns_same_instance() -> None:
    assert get_skynet_settings() is get_skynet_settings()
