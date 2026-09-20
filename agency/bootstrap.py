"""Composition root — the one place default implementations are wired.

``build_core`` assembles a :class:`agency.loop.SkynetCore` from the
configured backend (database or memory) and the deterministic placeholder
strategies. Future phases replace individual pieces (LLM planner, real
evaluators, new actions) either by passing components explicitly or by
extending the category registry below — never by editing the loop.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker

from agency.actions import (
    Action,
    ActionRegistry,
    EchoAction,
    GodsEyeLatestEventsAction,
)
from agency.config import SkynetSettings
from agency.evaluator import DeterministicEvaluator, Evaluator
from agency.experience import (
    DatabaseExperienceSink,
    ExperienceRecorder,
    ExperienceSink,
    InMemoryExperienceSink,
)
from agency.goals import GoalManager
from agency.loop import SkynetCore
from agency.perception import build_perception_adapter
from agency.planner import DeterministicPlanner, Planner
from agency.storage import DatabaseStorage, MemoryStorage, Storage
from agency.trace import DatabaseTraceSink, LoggingTraceSink, TraceSink

#: Which feature flag unlocks which action category. Actions with a
#: category absent from this mapping are always allowed to register.
_CATEGORY_FLAGS: dict[str, str] = {
    "web": "enable_web_tools",
    "comms": "enable_external_comms",
    "lab": "enable_self_improvement",
}


def _flag_for_category(settings: SkynetSettings, action: Action) -> bool:
    flag_name = _CATEGORY_FLAGS.get(action.category)
    return flag_name is None or bool(getattr(settings, flag_name))


def register_action(registry: ActionRegistry, action: Action, settings: SkynetSettings) -> bool:
    """Register ``action`` unless its category is gated off by feature flags.

    Returns True when registered. Unknown categories register freely;
    gated categories (web/comms/lab) require their flag to be enabled.
    """
    if not _flag_for_category(settings, action):
        return False
    registry.register(action)
    return True


def build_core(
    settings: SkynetSettings | None = None,
    *,
    session_factory: async_sessionmaker | None = None,
    storage: Storage | None = None,
    planner: Planner | None = None,
    evaluator: Evaluator | None = None,
    experience_sink: ExperienceSink | None = None,
    extra_actions: tuple[Action, ...] = (),
    extra_trace_sinks: tuple[TraceSink, ...] = (),
) -> SkynetCore:
    """Assemble a SkynetCore from settings.

    ``session_factory`` defaults to the process-wide database session
    factory. Pass ``storage='memory'`` via settings (or a custom storage)
    for DB-free runs.
    """
    settings = settings or SkynetSettings()

    if storage is None:
        if settings.storage_backend == "database":
            if session_factory is None:
                from database.session import get_sessionmaker

                # Lazy: creates the engine object, opens no connection.
                session_factory = get_sessionmaker()
            storage = DatabaseStorage(session_factory)
        else:
            storage = MemoryStorage()

    database_enabled = isinstance(storage, DatabaseStorage)
    if not database_enabled and settings.perception_adapter == "gods_eye":
        raise ValueError(
            "SKYNET_PERCEPTION_ADAPTER=gods_eye requires database storage; "
            "set SKYNET_PERCEPTION_ADAPTER=none or SKYNET_STORAGE_BACKEND=database"
        )

    # -- Actions --------------------------------------------------------------
    registry = ActionRegistry()
    register_action(registry, EchoAction(), settings)
    if database_enabled:
        register_action(registry, GodsEyeLatestEventsAction(session_factory), settings)
    for action in extra_actions:
        register_action(registry, action, settings)

    # -- Perception -------------------------------------------------------------
    session_for_perception = session_factory if database_enabled else None
    perception = build_perception_adapter(settings.perception_adapter, session_for_perception)

    # -- Strategies ---------------------------------------------------------------
    planner = planner or DeterministicPlanner()
    evaluator = evaluator or DeterministicEvaluator()

    # -- Experience + trace sinks ---------------------------------------------------
    if experience_sink is None:
        experience_sink = (
            DatabaseExperienceSink(session_factory) if database_enabled
            else InMemoryExperienceSink()
        )
    trace_sinks: list[TraceSink] = [LoggingTraceSink()]
    if database_enabled:
        trace_sinks.append(DatabaseTraceSink(session_factory))
    trace_sinks.extend(extra_trace_sinks)

    return SkynetCore(
        settings=settings,
        storage=storage,
        goal_manager=GoalManager(storage),
        actions=registry,
        planner=planner,
        evaluator=evaluator,
        perception=perception,
        experiences=ExperienceRecorder(experience_sink),
        trace_sinks=trace_sinks,
    )
