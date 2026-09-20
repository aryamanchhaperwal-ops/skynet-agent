"""Action system tests (registration, execution, failure capture)."""

from __future__ import annotations

from typing import Any

import pytest

from agency.actions.base import (
    Action,
    ActionContext,
    ActionError,
    ActionRegistry,
    ActionSpec,
    run_action,
)
from agency.actions.builtins import EchoAction, GodsEyeLatestEventsAction
from tests.conftest import BrokenSessionFactory, FakeSessionFactory


def make_ctx(**kwargs: Any) -> ActionContext:
    defaults: dict[str, Any] = {"run_id": "run-1", "state_snapshot": {}}
    defaults.update(kwargs)
    return ActionContext(**defaults)


class BoomAction(Action):
    name = "boom"
    description = "always raises ActionError"

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        raise ActionError("kaboom")


class UnexpectedAction(Action):
    name = "unexpected"
    description = "raises a non-ActionError"

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
        del spec, ctx
        raise RuntimeError("surprise")


# ---------------------------------------------------------------- registry


def test_register_and_get() -> None:
    registry = ActionRegistry()
    action = EchoAction()
    registry.register(action)
    assert registry.get("echo") is action
    assert "echo" in registry
    assert len(registry) == 1
    assert registry.names() == ["echo"]


def test_duplicate_registration_requires_override() -> None:
    registry = ActionRegistry()
    registry.register(EchoAction())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoAction())
    registry.register(EchoAction(), override=True)
    assert len(registry) == 1


def test_empty_name_rejected() -> None:
    class NoName(Action):
        name = ""
        async def execute(self, spec: ActionSpec, ctx: ActionContext) -> Any:
            return None

    with pytest.raises(ValueError, match="non-empty"):
        ActionRegistry().register(NoName())


# --------------------------------------------------------------- run_action


@pytest.mark.asyncio
async def test_run_action_success_path() -> None:
    result = await run_action(EchoAction(), ActionSpec(type="echo", params={"text": "hi"}), make_ctx())
    assert result.success is True
    assert result.output == {"text": "hi", "length": 2}
    assert result.error is None
    assert result.duration_ms >= 0
    assert result.finished_at >= result.started_at


@pytest.mark.asyncio
async def test_run_action_captures_action_error() -> None:
    result = await run_action(BoomAction(), ActionSpec(type="boom"), make_ctx())
    assert result.success is False
    assert result.output is None
    assert "ActionError: kaboom" in result.error


@pytest.mark.asyncio
async def test_run_action_captures_unexpected_exceptions() -> None:
    result = await run_action(UnexpectedAction(), ActionSpec(type="unexpected"), make_ctx())
    assert result.success is False
    assert "RuntimeError: surprise" in result.error


@pytest.mark.asyncio
async def test_run_action_records_timing_fields() -> None:
    result = await run_action(EchoAction(), ActionSpec(type="echo", params={"text": "x" * 100}), make_ctx())
    assert result.duration_ms >= 0
    assert result.action_type == "echo"


# ------------------------------------------------------------------ builtins


@pytest.mark.asyncio
async def test_echo_requires_string_text() -> None:
    result = await run_action(
        EchoAction(), ActionSpec(type="echo", params={"text": 123}), make_ctx()
    )
    assert result.success is False
    assert "must be a string" in result.error


@pytest.mark.asyncio
async def test_gods_eye_action_normalizes_rows(fake_event_rows: list[dict[str, Any]]) -> None:
    factory = FakeSessionFactory(fake_event_rows)
    action = GodsEyeLatestEventsAction(factory)  # type: ignore[arg-type]
    result = await run_action(action, ActionSpec(type="gods_eye_latest_events", params={}), make_ctx())
    assert result.success is True
    payload = result.output
    assert payload["count"] == 2
    assert payload["events"][0]["event_id"] == "usgs-test-1"
    assert payload["events"][0]["magnitude"] == 5.4
    embedded = payload["observation"]
    assert embedded["source"] == "gods_eye:usgs"
    assert embedded["kind"] == "json"
    assert embedded["content"]["events"][0]["event_id"] == "usgs-test-1"


@pytest.mark.asyncio
async def test_gods_eye_action_limit_validation() -> None:
    action = GodsEyeLatestEventsAction(BrokenSessionFactory())  # type: ignore[arg-type]
    for bad in (0, 201, "many"):
        result = await run_action(action, ActionSpec(type="gods_eye_latest_events", params={"limit": bad}), make_ctx())
        assert result.success is False
        assert "limit" in result.error


@pytest.mark.asyncio
async def test_gods_eye_action_degrades_on_database_failure() -> None:
    action = GodsEyeLatestEventsAction(BrokenSessionFactory())  # type: ignore[arg-type]
    result = await run_action(action, ActionSpec(type="gods_eye_latest_events"), make_ctx())
    assert result.success is False
    assert "database is down" in result.error


def test_gods_eye_action_category_is_core() -> None:
    assert EchoAction.category == "core"
    assert GodsEyeLatestEventsAction.category == "core"
