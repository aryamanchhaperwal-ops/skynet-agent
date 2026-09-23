"""Strategy registry — the versioned-artifact seam experiments compare.

A **strategy** is a named, versioned, *data* artifact: a callable procedure
plus frozen configuration, registered explicitly by an operator or by
Skynet's (future) improvement engine through the same public API. The
registry is the structural safety property of the whole experimentation
phase:

- Experiments reference ``name:version`` pairs (:class:`TargetRef`), never
  code paths or objects — so an experiment record is reproducible from
  data alone and can never name arbitrary code to run.
- Registered procedures receive a sandboxed :class:`TrialContext` — they
  cannot touch the host process, source files, or secrets.
- ``capability`` is a closed set (planner_policy, research_strategy,
  tool_config, retrieval_params, benchmark_def — blueprint §7). Unknown
  capabilities are rejected at registration.

Registration is an explicit, auditable act; the runner refuses to execute
unregistered targets rather than inventing behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agency.experiments.sandbox import TrialContext

logger = logging.getLogger("skynet.experiments.registry")

#: Closed capability vocabulary (blueprint: artifact-typed improvements only).
STRATEGY_CAPABILITIES: frozenset[str] = frozenset(
    {
        "planner_policy",
        "research_strategy",
        "tool_config",
        "retrieval_params",
        "benchmark_def",
        "evaluation_policy",
    }
)

#: Async procedure signature: context in, metrics dict out. Raising marks
#: the trial failed; returning metrics records them.
Procedure = Callable[[TrialContext], Awaitable[dict[str, float]]]


@dataclass
class Strategy:
    """One registered, versioned strategy artifact."""

    name: str
    version: str
    capability: str
    description: str = ""
    #: Frozen configuration recorded for reproducibility (data only).
    config: dict[str, Any] = field(default_factory=dict)
    procedure: Procedure | None = None
    registered_at: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"

    def describe(self) -> dict[str, Any]:
        """Serializable description (procedure excluded — it is code)."""
        return {
            "ref": self.ref,
            "capability": self.capability,
            "description": self.description,
            "config": self.config,
            "registered_at": self.registered_at,
        }


class StrategyRegistry:
    """Name+version → strategy registry with duplicate protection."""

    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}

    def register(
        self,
        *,
        name: str,
        version: str,
        capability: str,
        procedure: Procedure,
        description: str = "",
        config: dict[str, Any] | None = None,
    ) -> Strategy:
        """Register one strategy; refuses unknown capabilities and duplicates."""
        if capability not in STRATEGY_CAPABILITIES:
            raise ValueError(
                f"unknown capability {capability!r}; allowed: "
                f"{', '.join(sorted(STRATEGY_CAPABILITIES))}"
            )
        ref = f"{name}:{version}"
        if ref in self._strategies:
            raise ValueError(f"strategy {ref!r} is already registered")
        strategy = Strategy(
            name=name,
            version=version,
            capability=capability,
            description=description,
            config=dict(config or {}),
            procedure=procedure,
            registered_at=datetime.now(UTC).isoformat(),
        )
        self._strategies[ref] = strategy
        logger.info("strategy registered: %s (%s)", ref, capability)
        return strategy

    def unregister(self, ref: str) -> bool:
        return self._strategies.pop(ref, None) is not None

    def get(self, ref: str) -> Strategy:
        """Fetch by ``name:version``; unknown refs raise (no invention)."""
        strategy = self._strategies.get(ref)
        if strategy is None:
            known = ", ".join(sorted(self._strategies)) or "(none)"
            raise KeyError(f"strategy {ref!r} is not registered; known: {known}")
        return strategy

    def list(self, capability: str | None = None) -> list[Strategy]:
        strategies = list(self._strategies.values())
        if capability:
            strategies = [s for s in strategies if s.capability == capability]
        return sorted(strategies, key=lambda s: s.ref)

    def __len__(self) -> int:
        return len(self._strategies)


__all__ = [
    "STRATEGY_CAPABILITIES",
    "Procedure",
    "Strategy",
    "StrategyRegistry",
]
