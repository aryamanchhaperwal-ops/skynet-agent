"""Action abstractions: specs, results, the Action protocol, and the registry.

An *action* is the only way the Skynet core changes the world. Every action:

- is registered by name in an :class:`ActionRegistry`,
- is invoked through :func:`run_action`, which centralizes timing and
  exception capture (an action that raises yields a failed
  :class:`ActionResult`, never a crashed run),
- carries an allow-list category so future tool classes (web, comms, lab)
  can be gated by configuration.

Future actions (web search, browser navigation, ask-another-AI, memory
writes, knowledge queries, experiments) plug in by subclassing
:class:`Action` and registering an instance — no core changes required.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field

logger = logging.getLogger("skynet.actions")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class ActionSpec(BaseModel):
    """A requested action: what to do and with which parameters."""

    id: str = Field(default_factory=_new_id)
    #: Registry name, e.g. ``echo`` or ``gods_eye_latest_events``.
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class ActionResult(BaseModel):
    """Outcome of one action execution, including failure information."""

    action_id: str
    action_type: str
    success: bool
    output: Any = None
    error: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime = Field(default_factory=_utcnow)
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionRecord(BaseModel):
    """State-log entry pairing a spec with its result."""

    spec: ActionSpec
    result: ActionResult | None = None


class ActionContext(BaseModel):
    """Read-only execution context handed to actions."""

    run_id: str
    goal_id: str | None = None
    #: Serializable snapshot of the agent state at selection time.
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    #: The run's :class:`agency.trace.Trace` facade (optional). Actions that
    #: perform multi-part work (e.g. web search / fetch / research) emit their
    #: own lifecycle events through it, keeping one event stream per run.
    #: Excluded from serialization: it is a live object, not data.
    tracer: Any = Field(default=None, exclude=True, repr=False)

    async def emit_trace(self, event_type: str, **payload: Any) -> None:
        """Emit a trace event if a tracer is attached; never raises."""
        if self.tracer is None:
            return
        try:
            await self.tracer.emit(event_type, **payload)
        except Exception:  # pragma: no cover - tracer failure must not fail actions
            logger.exception("action %s failed to emit trace event %s", self.run_id, event_type)


class Action(ABC):
    """Protocol every tool/action implements.

    ``execute`` returns the action's *output* (any JSON-serializable value).
    Raise any exception (conventionally :class:`ActionError`) to produce a
    failed :class:`ActionResult`; :func:`run_action` performs the capture.
    """

    #: Registry name (unique).
    name: ClassVar[str]
    #: Gate category: ``core`` | ``web`` | ``comms`` | ``lab``.
    category: ClassVar[str] = "core"
    description: ClassVar[str] = ""

    @abstractmethod
    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        """Execute the action and return its output."""


class ActionError(RuntimeError):
    """Raised by actions to signal a clean, expected failure."""


async def run_action(action: Action, spec: ActionSpec, ctx: ActionContext) -> ActionResult:
    """Execute one action with centralized timing and exception capture.

    This is the only place the core invokes actions: a raising action can
    never crash the run — it becomes a failed ``ActionResult`` instead.
    """
    started_at = _utcnow()
    clock = time.perf_counter()
    try:
        output = await action.execute(spec, ctx)
    except Exception as exc:
        return ActionResult(
            action_id=spec.id,
            action_type=spec.type,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            finished_at=_utcnow(),
            duration_ms=int((time.perf_counter() - clock) * 1000),
        )
    return ActionResult(
        action_id=spec.id,
        action_type=spec.type,
        success=True,
        output=output,
        started_at=started_at,
        finished_at=_utcnow(),
        duration_ms=int((time.perf_counter() - clock) * 1000),
    )


class ActionRegistry:
    """Name → action registry. Duplicate names must be registered with
    ``override=True`` explicitly; silent clobbering is a bug factory."""

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action, *, override: bool = False) -> None:
        name = action.name
        if not name:
            raise ValueError("Action.name must be a non-empty string")
        if name in self._actions and not override:
            raise ValueError(f"action {name!r} is already registered")
        self._actions[name] = action

    def get(self, name: str) -> Action | None:
        return self._actions.get(name)

    def names(self) -> list[str]:
        return sorted(self._actions)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._actions

    def __len__(self) -> int:
        return len(self._actions)
