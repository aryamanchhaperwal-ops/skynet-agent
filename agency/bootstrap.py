"""Composition root — the one place default implementations are wired.

``build_core`` assembles a :class:`agency.loop.SkynetCore` from the
configured backend (database or memory) and the deterministic placeholder
strategies. Future phases replace individual pieces (LLM planner, real
evaluators, new actions) either by passing components explicitly or by
extending the category registry below — never by editing the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - annotation-only import
    from agency.orchestrator.orchestrator import AutonomousOrchestrator

from sqlalchemy.ext.asyncio import async_sessionmaker

from agency.actions import (
    Action,
    ActionRegistry,
    EchoAction,
    GodsEyeLatestEventsAction,
)
from agency.comms.actions import (
    AIAskAction,
    AICompareAction,
    AIListParticipantsAction,
)
from agency.comms.compare import ComparisonService
from agency.comms.manager import ConversationManager
from agency.comms.providers import AIProviderRegistry, build_providers_from_list
from agency.config import SkynetSettings
from agency.evaluator import DeterministicEvaluator, Evaluator
from agency.experience import (
    DatabaseExperienceSink,
    ExperienceRecorder,
    ExperienceSink,
    InMemoryExperienceSink,
)
from agency.experiments.actions import (
    ExperimentInspectAction,
    ExperimentListAction,
    ExperimentRunAction,
)
from agency.experiments.experiments_store import ExperimentRegistry
from agency.experiments.registry import StrategyRegistry
from agency.experiments.runner import ExperimentRunner
from agency.goals import GoalManager
from agency.improve.actions import (
    ImproveHistoryAction,
    ImproveInspectAction,
    ImproveProposeAction,
)
from agency.improve.detector import DetectorThresholds, WeaknessDetector
from agency.improve.pipeline import ImprovementPipeline
from agency.improve.store import ImprovementStore
from agency.intelligence import (
    IntelligenceService,
    LLMPlanner,
    build_llm_provider,
)
from agency.loop import SkynetCore
from agency.memory import MemoryManager
from agency.memory.actions import (
    MemoryExtractAction,
    MemorySearchAction,
    MemoryStoreAction,
)
from agency.memory.stores import build_memory_store
from agency.perception import build_perception_adapter
from agency.planner import DeterministicPlanner, Planner
from agency.storage import DatabaseStorage, MemoryStorage, Storage
from agency.trace import DatabaseTraceSink, LoggingTraceSink, TraceSink
from agency.web.actions import (
    WebExtractAction,
    WebFetchAction,
    WebResearchAction,
    WebSearchAction,
)
from agency.web.fetcher import SafeFetcher
from agency.web.research import ResearchService
from agency.web.search import SearchProvider, build_search_provider

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


class WebStack(NamedTuple):
    """The web exploration pieces, exposed for wiring and testing."""

    provider: SearchProvider
    fetcher: SafeFetcher
    service: ResearchService


class MemoryStack(NamedTuple):
    """The long-term memory pieces, exposed for wiring and testing."""

    manager: MemoryManager


class CommsStack(NamedTuple):
    """The AI-communication pieces, exposed for wiring and testing."""

    registry: AIProviderRegistry
    manager: ConversationManager
    comparison: ComparisonService


class LabStack(NamedTuple):
    """The experimentation + self-improvement pieces, exposed for wiring."""

    strategies: StrategyRegistry
    experiments: ExperimentRegistry
    runner: ExperimentRunner
    improvement: ImprovementPipeline


def build_web_stack(settings: SkynetSettings) -> WebStack:
    """Assemble the web exploration stack from settings.

    Kept separate from ``build_core`` so tests and future callers can build
    the web layer without a full core (and so its lifetime — especially the
    owned HTTP client — stays explicit).
    """
    provider = build_search_provider(
        settings.search_provider, user_agent=settings.web_user_agent
    )
    fetcher = SafeFetcher(
        user_agent=settings.web_user_agent,
        timeout=settings.web_timeout_seconds,
        max_bytes=settings.web_max_bytes,
        max_redirects=settings.web_max_redirects,
        max_chars=settings.web_max_content_chars,
        respect_robots=settings.web_respect_robots_txt,
        blocked_hosts=settings.web_blocked_host_set,
    )
    service = ResearchService(
        provider,
        fetcher,
        max_results_per_query=settings.web_max_results_per_query,
        max_sources=settings.web_max_pages_per_research,
    )
    return WebStack(provider=provider, fetcher=fetcher, service=service)


def build_comms_stack(
    settings: SkynetSettings,
    *,
    providers: tuple[object, ...] | None = None,
) -> CommsStack:
    """Assemble the AI-communication stack from settings.

    Only providers named in ``SKYNET_AI_PROVIDERS`` are activated; the factory
    raises for unknown names rather than inventing behavior. ``providers``
    (optional) replaces name-based construction — used by tests and the
    ``ai-demo`` command to inject provider instances with distinct
    participants/scenarios. Kept separate from ``build_core`` so tests can
    build comms without a full core.
    """
    registry = AIProviderRegistry()
    if providers is not None:
        for provider in providers:
            registry.register(provider)  # type: ignore[arg-type]
    else:
        # Repeated names become independent participants (mock#2, mock#3…).
        for provider in build_providers_from_list(settings.ai_providers.split(",")):
            registry.register(provider)
    manager = ConversationManager(
        registry,
        max_turns=settings.ai_max_turns_per_conversation,
        request_timeout=settings.ai_conversation_timeout_seconds,
        max_retries=settings.ai_max_retries,
        request_interval=settings.ai_request_interval_seconds,
    )
    comparison = ComparisonService(manager)
    return CommsStack(registry=registry, manager=manager, comparison=comparison)


def build_intelligence_service(settings: SkynetSettings) -> IntelligenceService:
    """Build the intelligence provider/service from settings.

    ``openai`` denotes any OpenAI-compatible endpoint (OpenAI, Ollama via
    SKYNET_LLM_BASE_URL, vLLM, OpenRouter…). Credentials are read from the
    environment at call time — never from settings or code. The factory
    raises loudly on missing credentials rather than silently degrading.
    """
    from agency.intelligence.service import IntelligenceService

    provider = build_llm_provider(
        settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    return IntelligenceService(
        provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def build_memory_stack(settings: SkynetSettings) -> MemoryStack:
    """Assemble the long-term memory stack from settings."""
    store = build_memory_store(
        settings.memory_backend, sqlite_path=settings.memory_sqlite_path
    )
    manager = MemoryManager(
        store,
        importance_threshold=settings.memory_importance_threshold,
        max_results=settings.memory_max_results,
        recall_max_chars=settings.memory_recall_max_chars,
    )
    return MemoryStack(manager=manager)


def build_lab_stack(
    settings: SkynetSettings,
    *,
    strategy_registry: StrategyRegistry | None = None,
    services: dict[str, object] | None = None,
    memory: MemoryManager | None = None,
    improvement_store: ImprovementStore | None = None,
    emit: object | None = None,
) -> LabStack:
    """Assemble the experimentation + self-improvement stack from settings.

    ``strategy_registry`` (optional) injects pre-registered strategies —
    used by tests and the demo; by default an empty registry is created
    and only explicitly registered strategies can ever run. ``services``
    is the sandbox's explicitly-injected capability map. Kept separate
    from ``build_core`` so tests can build the lab without a full core.
    """
    strategies = strategy_registry or StrategyRegistry()
    experiments = ExperimentRegistry(settings.experiments_registry_path)
    runner = ExperimentRunner(strategies, services=services)
    store = improvement_store or ImprovementStore(settings.improvements_registry_path)
    detector = WeaknessDetector(
        DetectorThresholds(
            min_frequency=settings.improvement_min_frequency,
            low_score=settings.improvement_low_score_threshold,
            duplicate_rate=settings.improvement_duplicate_rate_threshold,
        )
    )
    pipeline = ImprovementPipeline(
        strategies=strategies,
        runner=runner,
        experiments=experiments,
        memory=memory,
        emit=emit,
        store=store,
        detector=detector,
    )
    return LabStack(
        strategies=strategies,
        experiments=experiments,
        runner=runner,
        improvement=pipeline,
    )


def build_llm_evaluator(service: IntelligenceService) -> Evaluator:
    """Build the LLM-backed evaluator (kept separate for test injection)."""
    from agency.intelligence.evaluator import LLMEvaluator

    return LLMEvaluator(service)


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
    memory: MemoryManager | None = None,
    emit: object | None = None,
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
    # Explicitly injected actions bypass the category gate: passing an action
    # to ``build_core`` IS the operator's deliberate choice (tests, custom
    # deployments, fake transports). Only bootstrap's own built-in wiring
    # below is flag-gated.
    for action in extra_actions:
        registry.register(action)

    # -- Web exploration actions (flag-gated, category 'web') -----------------
    if settings.enable_web_tools:
        web = build_web_stack(settings)
        register_action(registry, WebSearchAction(web.provider), settings)
        register_action(registry, WebFetchAction(web.fetcher), settings)
        register_action(registry, WebExtractAction(web.fetcher), settings)
        register_action(registry, WebResearchAction(web.service), settings)

    # -- AI communication actions (flag-gated, category 'comms') ---------------
    if settings.enable_external_comms:
        comms = build_comms_stack(settings)
        register_action(registry, AIListParticipantsAction(comms.manager), settings)
        register_action(registry, AIAskAction(comms.manager), settings)
        register_action(registry, AICompareAction(comms.manager, comms.comparison), settings)

    # -- Experimentation + self-improvement actions (flag-gated, 'lab') ----------
    if settings.enable_self_improvement:
        lab = build_lab_stack(settings, memory=memory)
        run_action = ExperimentRunAction(lab.runner, lab.experiments, memory=memory)
        register_action(registry, run_action, settings)
        register_action(registry, ExperimentListAction(lab.experiments), settings)
        register_action(registry, ExperimentInspectAction(lab.experiments), settings)
        register_action(registry, ImproveProposeAction(lab.improvement), settings)
        register_action(registry, ImproveInspectAction(lab.improvement), settings)
        register_action(registry, ImproveHistoryAction(lab.improvement), settings)

    # -- Perception -------------------------------------------------------------
    session_for_perception = session_factory if database_enabled else None
    perception = build_perception_adapter(settings.perception_adapter, session_for_perception)

    # -- Long-term memory (always assembled; backend from settings) -------------
    memory = memory or build_memory_stack(settings).manager
    register_action(registry, MemoryStoreAction(memory), settings)
    register_action(registry, MemorySearchAction(memory), settings)
    register_action(registry, MemoryExtractAction(memory), settings)

    # -- Intelligence + strategies ------------------------------------------------
    use_llm_planner = planner is None and settings.planner_strategy == "llm"
    use_llm_evaluator = evaluator is None and settings.evaluator_strategy == "llm"
    service: IntelligenceService | None = None
    if use_llm_planner or use_llm_evaluator:
        try:
            service = build_intelligence_service(settings)
        except Exception:
            # Missing credentials etc. — deterministic strategies keep the
            # core working (LLM failure must not destroy the core).
            service = None
    if planner is None:
        # The actions this assembly may actually execute: the intersection of
        # the configured allow-list (execution gate, loop._execute_step) and
        # the actions registered here. An empty configuration means "allow
        # all registered actions". Used *only* for the LLM path: its fallback
        # planner must never name an unrunnable action.
        executable_actions = (
            settings.action_allowlist & frozenset(registry.names())
            if settings.action_allowlist
            else frozenset(registry.names())
        )
        planner = (
            LLMPlanner(
                service,  # type: ignore[arg-type] - guarded by use_llm_planner
                # Model-failure fallback is action-aware: it can only emit a
                # registered, allowed default. The deterministic-only path
                # below keeps the historical unfiltered default (its refusal
                # path is a documented, tested guarantee).
                fallback=DeterministicPlanner(allowed_actions=executable_actions),
                allowed_actions=executable_actions,
                max_steps=settings.max_steps_per_run,
            )
            if service is not None
            else DeterministicPlanner()
        )
    if evaluator is None:
        evaluator = (
            build_llm_evaluator(service) if service is not None else DeterministicEvaluator()
        )

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
    if emit is not None:
        # Caller-supplied async emit(event, **payload) hook (P9): the same
        # facade the orchestrator traces through, so a real run's core
        # cycles share one observable event stream.
        from agency.trace import CallableTraceSink

        trace_sinks.append(CallableTraceSink(emit))

    core = SkynetCore(
        settings=settings,
        storage=storage,
        goal_manager=GoalManager(storage),
        actions=registry,
        planner=planner,
        evaluator=evaluator,
        perception=perception,
        experiences=ExperienceRecorder(experience_sink),
        trace_sinks=trace_sinks,
        memory=memory,
    )
    # Read-only observability seam (P9): expose the intelligence service on
    # the core so the orchestrator can trace LLM request lifecycle events.
    # ``None`` when the run is deterministic-only.
    core._intelligence = service
    return core


def build_orchestrator(
    settings: SkynetSettings | None = None,
    *,
    core: SkynetCore | None = None,
    memory: MemoryManager | None = None,
    improvement_pipeline: ImprovementPipeline | None = None,
    research_adapter: object | None = None,
    comms_adapter: object | None = None,
    emit: object | None = None,
    trace_store: object | None = None,
) -> AutonomousOrchestrator:
    """Assemble the autonomous orchestrator from settings.

    Reuses ``build_core`` (or the caller's core), the P7 improvement
    pipeline, and the P4/P5-backed adapters. Pass explicit adapters for
    tests/demos (local deterministic ones by default offline).
    """
    from agency.orchestrator.adapters import (
        build_comms_adapter,
        build_research_adapter,
    )
    from agency.orchestrator.orchestrator import AutonomousOrchestrator
    from agency.orchestrator.store import RunStore
    from agency.orchestrator.trace_store import RunTraceStore

    settings = settings or SkynetSettings()
    core = core or build_core(settings, memory=memory, emit=emit)
    memory = memory or core.memory_manager
    pipeline = improvement_pipeline
    detector = None
    if pipeline is None and settings.enable_self_improvement:
        lab = build_lab_stack(settings, memory=memory)
        pipeline = lab.improvement
        detector = lab.improvement._detector  # the pipeline's own thresholds
    if trace_store is None and settings.autonomous_traces_path:
        trace_store = RunTraceStore(settings.autonomous_traces_path)
    return AutonomousOrchestrator(
        core=core,
        store=RunStore(settings.autonomous_runs_path),
        research=research_adapter or build_research_adapter(settings),
        comms=comms_adapter or build_comms_adapter(settings),
        memory=memory,
        detector=detector,
        pipeline=pipeline,
        emit=emit,
        trace_store=trace_store,
    )
