"""Bootstrap / composition root tests."""

from __future__ import annotations

import pytest

from agency.actions.base import Action, ActionContext, ActionRegistry, ActionSpec
from agency.bootstrap import build_core, register_action
from agency.config import SkynetSettings
from agency.experience import DatabaseExperienceSink, InMemoryExperienceSink
from agency.perception import GodsEyePerceptionAdapter, NullPerceptionAdapter
from agency.storage import DatabaseStorage
from agency.trace import DatabaseTraceSink, LoggingTraceSink


class WebProbeAction(Action):
    name = "web_probe"
    category = "web"

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> object:
        return None


class CommsProbeAction(Action):
    name = "comms_probe"
    category = "comms"

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> object:
        return None


class LabProbeAction(Action):
    name = "lab_probe"
    category = "lab"

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> object:
        return None


class MysteryProbeAction(Action):
    name = "mystery_probe"
    category = "mystery"  # not in the gate map

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> object:
        return None


def test_build_core_memory_backend_is_db_free(memory_settings) -> None:
    core = build_core(memory_settings)
    assert core is not None
    # Memory backend: no DB action registered, null perception.
    assert "echo" in core._actions
    assert "gods_eye_latest_events" not in core._actions
    assert isinstance(core._perception, NullPerceptionAdapter)


def test_build_core_database_backend_wires_db_components(memory_settings) -> None:
    settings = memory_settings.model_copy(update={"storage_backend": "database"})
    core = build_core(settings, session_factory=None)
    assert isinstance(core._actions, ActionRegistry)
    # DB-backed: God's Eye action + DB sinks, constructed lazily
    # (session_factory=None is tolerated at build time).
    assert "gods_eye_latest_events" in core._actions
    # The fixture pins perception_adapter='none', so the adapter stays null
    # even on the database backend (perception choice is orthogonal).
    assert isinstance(core._perception, NullPerceptionAdapter)
    sinks = core._trace_sinks
    assert isinstance(sinks[0], LoggingTraceSink)
    assert isinstance(sinks[1], DatabaseTraceSink)


def test_build_core_database_backend_gods_eye_perception() -> None:
    settings = SkynetSettings(storage_backend="database", perception_adapter="gods_eye")
    core = build_core(settings, session_factory=None)
    assert isinstance(core._perception, GodsEyePerceptionAdapter)
    assert "gods_eye_latest_events" in core._actions


def test_web_actions_gated_by_flag(memory_settings) -> None:
    settings = memory_settings.model_copy(update={"enable_web_tools": False})
    registry = ActionRegistry()
    assert register_action(registry, WebProbeAction(), settings) is False
    assert "web_probe" not in registry

    settings_on = memory_settings.model_copy(update={"enable_web_tools": True})
    registry2 = ActionRegistry()
    assert register_action(registry2, WebProbeAction(), settings_on) is True
    assert "web_probe" in registry2


def test_comms_and_lab_actions_gated(memory_settings) -> None:
    off = memory_settings.model_copy(
        update={"enable_external_comms": False, "enable_self_improvement": False}
    )
    registry = ActionRegistry()
    assert register_action(registry, CommsProbeAction(), off) is False
    assert register_action(registry, LabProbeAction(), off) is False

    on = memory_settings.model_copy(
        update={"enable_external_comms": True, "enable_self_improvement": True}
    )
    registry2 = ActionRegistry()
    assert register_action(registry2, CommsProbeAction(), on) is True
    assert register_action(registry2, LabProbeAction(), on) is True


def test_unknown_category_registers_freely(memory_settings) -> None:
    registry = ActionRegistry()
    assert register_action(registry, MysteryProbeAction(), memory_settings) is True
    assert "mystery_probe" in registry


def test_extra_actions_flow_into_core(memory_settings) -> None:
    settings = memory_settings.model_copy(update={"enable_web_tools": True})
    core = build_core(settings, extra_actions=(WebProbeAction(),))
    assert "web_probe" in core._actions


def test_gated_action_not_registered_even_if_requested(memory_settings) -> None:
    core = build_core(memory_settings, extra_actions=(CommsProbeAction(),))
    assert "comms_probe" not in core._actions


def test_database_components_lazy_construction() -> None:
    # These constructors must not open connections.
    assert DatabaseExperienceSink(session_factory=None) is not None  # type: ignore[arg-type]
    assert DatabaseTraceSink(session_factory=None) is not None  # type: ignore[arg-type]
    assert DatabaseStorage(session_factory=None) is not None  # type: ignore[arg-type]
    assert InMemoryExperienceSink() is not None


def test_memory_backend_rejects_gods_eye_perception(memory_settings) -> None:
    """Consistency gate: the God's Eye adapter reads the events table and
    therefore requires database storage."""
    settings = memory_settings.model_copy(update={"perception_adapter": "gods_eye"})
    with pytest.raises(ValueError, match="requires database storage"):
        build_core(settings)
