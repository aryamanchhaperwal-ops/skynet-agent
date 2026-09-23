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
    default_provider: Literal["deterministic", "llm"] = Field(
        default="deterministic",
        description="Reasoning strategy for planning/evaluation: 'deterministic' uses the "
        "built-in non-LLM strategies (default, always available); 'llm' uses the "
        "configured intelligence provider with deterministic fallback on failure.",
    )

    # -- Perception ----------------------------------------------------------
    perception_adapter: Literal["gods_eye", "none"] = Field(
        default="gods_eye",
        description="Initial observation source. 'gods_eye' reads the existing events table "
        "read-only via the perception adapter.",
    )

    # -- Tool availability ----------------------------------------------------
    #: Comma-separated action names the loop may execute. Actions present in
    #: the registry but missing here are refused at selection time. Enabling
    #: SKYNET_ENABLE_WEB_TOOLS without adding the web_* names here means the
    #: actions register but are refused at selection time — a deliberate
    #: second gate.
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

    # -- Web exploration (agency.web, Phase P4) --------------------------------
    #: 'wikipedia' is the keyless default (open MediaWiki API, automation-
    #: friendly). 'duckduckgo' exists but currently serves an anomaly/202 page
    #: to scripted clients; keyed commercial providers slot in later.
    search_provider: Literal["wikipedia", "duckduckgo", "none"] = "wikipedia"
    #: Descriptive UA: some sites 403 generic clients. Not a spoof — it names
    #: this agent and points at a contact path per robots-ethics convention.
    web_user_agent: str = (
        "SKYNET-ResearchBot/0.1 (+https://github.com/skynet-intel; research agent, "
        "respectful of robots.txt)"
    )
    web_timeout_seconds: float = Field(default=15.0, ge=1.0, le=120.0)
    web_max_pages_per_research: int = Field(default=8, ge=1, le=50)
    web_max_results_per_query: int = Field(default=10, ge=1, le=100)
    web_max_content_chars: int = Field(
        default=20000, ge=1000, le=1_000_000,
        description="Per-page extracted text cap (bounded payloads, no huge HTML dumps).",
    )
    web_max_bytes: int = Field(
        default=2_000_000, ge=10_000, le=50_000_000,
        description="Per-response download cap; larger bodies are truncated, not parsed.",
    )
    web_max_redirects: int = Field(default=5, ge=0, le=10)
    web_min_request_interval_seconds: float = Field(
        default=1.0, ge=0.0, le=60.0,
        description="Politeness delay between requests to the same host (rate-limit respect).",
    )
    web_respect_robots_txt: bool = Field(
        default=True,
        description="Fetch robots.txt and refuse disallowed paths. Disable only for tests.",
    )
    web_blocked_hosts: str = Field(
        default="localhost,127.0.0.1,0.0.0.0,::1,169.254.169.254,metadata.google.internal",
        description="Comma-separated hosts the fetcher always refuses (SSRF guard).",
    )
    web_offline_mode: bool = Field(
        default=False,
        description="When true, web tools fail fast with a structured offline error "
        "(CI, deterministic demos). Search results are never synthesized.",
    )

    # -- AI ↔ AI communication (agency.comms, Phase P5) -------------------------
    #: Enabled external AI participants. ``mock`` is a deterministic, offline
    #: provider for tests and demos. Real providers (e.g. ``anthropic``) activate
    #: only when the required key exists in the environment AND the operator
    #: lists them here — no key discovery, no silent fallback to paid APIs.
    ai_providers: str = Field(
        default="mock",
        description="Comma-separated external AI provider names to activate."
        " 'mock' is offline and deterministic; real providers need env keys.",
    )
    ai_conversation_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=600.0,
        description="Per-provider-request timeout for external AI calls.",
    )
    ai_max_turns_per_conversation: int = Field(
        default=5, ge=1, le=50,
        description="Hard cap on Skynet→AI turns; prevents runaway conversations.",
    )
    ai_max_retries: int = Field(
        default=1, ge=0, le=10,
        description="Retries per failed AI request (exponential backoff not needed at this scale).",
    )
    ai_max_request_cost_usd: float | None = Field(
        default=None,
        description="Optional per-request cost ceiling (USD). Providers reporting usage "
        "metadata above this refuse to send.",
    )
    ai_request_interval_seconds: float = Field(
        default=1.0, ge=0.0, le=60.0,
        description="Politeness delay between requests to the same provider.",
    )

    # -- Intelligence / LLM provider (agency.intelligence, Phase P2) -------------
    #: Skynet's own reasoning provider for planning, evaluation, synthesis and
    #: memory processing. 'mock' is deterministic and offline; 'openai'
    #: denotes any OpenAI-compatible endpoint (OPENAI_API_KEY + optional
    #: SKYNET_LLM_BASE_URL for Ollama/vLLM/OpenRouter). Keys stay in the
    #: environment; providers without their key refuse to activate.
    llm_provider: Literal["mock", "openai"] = Field(
        default="mock",
        description="Intelligence provider implementation. 'mock' is deterministic/offline.",
    )
    llm_model: str = Field(
        default="skynet-mock-1",
        description="Model identifier handed to the provider.",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL (Ollama: http://localhost:11434/v1).",
    )
    llm_temperature: float = Field(
        default=0.2, ge=0.0, le=2.0,
        description="Sampling temperature for generation requests.",
    )
    llm_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=600.0,
        description="Per-request timeout for intelligence calls.",
    )
    llm_max_tokens: int = Field(
        default=1024, ge=16, le=32_000,
        description="Default maximum completion tokens per request.",
    )
    llm_max_retries: int = Field(
        default=1, ge=0, le=5,
        description="Retries for transient provider failures (timeouts/connection errors).",
    )
    #: How the loop picks planner/evaluator implementations (see default_provider).
    planner_strategy: Literal["deterministic", "llm"] = "deterministic"
    evaluator_strategy: Literal["deterministic", "llm"] = "deterministic"

    # -- Long-term memory (agency.memory, Phase P2) -----------------------------
    #: 'memory' is process-local (tests); 'sqlite' persists across restarts
    #: with zero infrastructure (stdlib sqlite3); 'postgres' will reuse the
    #: existing async stack in a later phase. No vector DB is introduced yet —
    #: the store protocol is the future seam.
    memory_backend: Literal["memory", "sqlite"] = Field(
        default="memory",
        description="Long-term memory persistence. 'sqlite' survives process restarts.",
    )
    memory_sqlite_path: str = Field(
        default="data/skynet_memory.db",
        description="SQLite database file for the 'sqlite' memory backend.",
    )
    memory_max_results: int = Field(
        default=10, ge=1, le=100,
        description="Default cap on memories returned by recall/search.",
    )
    memory_importance_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Minimum importance for a memory candidate to be stored automatically.",
    )
    memory_recall_max_chars: int = Field(
        default=4000, ge=500, le=100_000,
        description="Total character budget for memories injected into planning context.",
    )

    # -- Experimentation (agency.experiments, Phase P6) ---------------------------
    #: Experiments compare *registered strategy artifacts* through a sandbox;
    #: they can never reference or modify source code. Actions ride the dark
    #: ``enable_self_improvement`` gate (category 'lab').
    experiments_registry_path: str = Field(
        default="data/experiments.jsonl",
        description="JSONL file where experiment history is persisted.",
    )
    experiment_default_trials: int = Field(
        default=3, ge=1, le=100,
        description="Default number of trials per arm when an experiment omits it.",
    )
    experiment_per_trial_timeout_seconds: float = Field(
        default=60.0, ge=0.1, le=3600.0,
        description="Default per-trial wall-clock timeout.",
    )
    experiment_max_total_seconds: float = Field(
        default=600.0, ge=1.0, le=86_400.0,
        description="Default total wall-clock budget for one experiment.",
    )

    @property
    def web_blocked_host_set(self) -> frozenset[str]:
        return frozenset(
            host.strip().lower() for host in self.web_blocked_hosts.split(",") if host.strip()
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
