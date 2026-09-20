"""Configuration owned by the Skynet core (agency layer).

Same conventions as ``database.settings``: pydantic-settings, ``SKYNET_``
environment prefix, single root ``.env`` file, cached process-wide accessor.
Secrets never live here — provider keys stay in the environment and are
consumed by the (future) intelligence layer, not by the core.

Every capability flag below ships **dark**: a flag that unlocks a new class
of capability (web tools, external AI communication, self-improvement)
defaults to ``False`` and must be enabled explicitly by the operator.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from database.settings import ENV_FILE


class SkynetSettings(BaseSettings):
    """Runtime configuration for the Skynet core."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="SKYNET_",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Runtime ------------------------------------------------------------
    env: Literal["development", "test", "staging", "production"] = "development"
    log_level: str = Field(default="INFO", description="Root log level for skynet.* loggers.")
    storage_backend: Literal["database", "memory"] = Field(
        default="database",
        description="'database' persists goals/runs/events/experiences in PostgreSQL; "
        "'memory' keeps everything in-process (tests, dry runs).",
    )
    default_provider: Literal["deterministic", "anthropic", "openai"] = Field(
        default="deterministic",
        description="Reasoning provider selection. Only 'deterministic' is implemented; "
        "the LLM provider layer arrives with the intelligence phase.",
    )

    # -- Perception ----------------------------------------------------------
    perception_adapter: Literal["gods_eye", "none"] = Field(
        default="gods_eye",
        description="Initial observation source. 'gods_eye' reads the existing events table "
        "read-only via the perception adapter.",
    )

    # -- Tool availability ----------------------------------------------------
    #: Comma-separated action names the loop may execute. Actions present in
    #: the registry but missing here are refused at selection time.
    enabled_actions: str = "echo,gods_eye_latest_events"

    # -- Execution limits (budgets) -------------------------------------------
    max_steps_per_run: int = Field(default=10, ge=1, le=100)
    max_run_seconds: int = Field(default=120, ge=1, le=3600)

    # -- Capability flags (all default OFF) ------------------------------------
    enable_web_tools: bool = Field(
        default=False,
        description="Registers actions with category 'web' (web search / fetch). Phase P4.",
    )
    enable_external_comms: bool = Field(
        default=False,
        description="Registers actions with category 'comms' (AI-to-AI communication). Phase P5.",
    )
    enable_self_improvement: bool = Field(
        default=False,
        description="Allows the loop to submit improvement proposals to the lab. Phase P7.",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        candidate = value.strip().upper()
        if candidate not in logging.getLevelNamesMapping():
            allowed = ", ".join(sorted(logging.getLevelNamesMapping()))
            raise ValueError(f"SKYNET_LOG_LEVEL must be one of: {allowed}")
        return candidate

    @property
    def action_allowlist(self) -> frozenset[str]:
        """Parsed view of ``enabled_actions`` (whitespace tolerated)."""
        return frozenset(
            item.strip() for item in self.enabled_actions.split(",") if item.strip()
        )


@lru_cache(maxsize=1)
def get_skynet_settings() -> SkynetSettings:
    """Return the process-wide, cached Skynet settings instance."""
    return SkynetSettings()
