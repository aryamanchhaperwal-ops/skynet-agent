"""End-to-end validation gap closures (Phase P10 hardening).

Focused tests for the four gaps the mission validation left open:

1. **Action registration / fallback safety** — a planner fallback can only
   name a registered, allow-listed action.
2. **Plan validation** — model-derived plans are validated before execution;
   an invalid plan falls back deterministically once, and terminates the run
   safely if even the fallback is invalid.
3. **Multi-source verification** — evidence corroboration is classified
   (single-source / multi-source / conflicting / unavailable) without ever
   fabricating a second source.
4. **Persisted mission trace** — trace events survive a store reload
   (process-restart stand-in) and never carry secrets.

Nothing here reaches the network: every path is deterministic and offline.
"""

from __future__ import annotations

from typing import Any

from agency.actions.base import ActionSpec
from agency.bootstrap import build_core
from agency.config import SkynetSettings
from agency.experiments.demo import _CORPUS
from agency.goals import Goal, GoalSpec
from agency.intelligence.planner import LLMPlanner, LLMPlanPayload
from agency.orchestrator.adapters import LocalCommsAdapter, LocalResearchAdapter
from agency.orchestrator.models import Evidence, OrchestratorBudgets, RunStatus
from agency.orchestrator.orchestrator import AutonomousOrchestrator
from agency.orchestrator.store import RunStore
from agency.orchestrator.trace_store import RunTraceStore, redact_payload
from agency.orchestrator.verification import verify_evidence
from agency.plan_validation import validate_plan
from agency.planner import DeterministicPlanner, Plan, Planner
from agency.state import AgentState
from agency.trace import MemoryTraceSink
from tests.test_orchestrator_integration import GOOD_CONFIG, make_orchestrator


def _settings(**update: Any) -> SkynetSettings:
    base: dict[str, Any] = {"storage_backend": "memory", "perception_adapter": "none"}
    base.update(update)
    return SkynetSettings(**base)


def _goal(title: str = "Research something", params: dict | None = None) -> Goal:
    return Goal.from_spec(GoalSpec(title=title, params=params or {}))


# -- 1. action registration / fallback safety -------------------------------------------


async def test_deterministic_default_substituted_when_not_allowed() -> None:
    """A configured default outside the allowed set is replaced, not emitted."""
    planner = DeterministicPlanner(allowed_actions=frozenset({"echo"}))
    plan = await planner.plan(_goal(), AgentState())
    assert plan.step_types == ["echo"]
    assert "safe fallback" in plan.rationale


async def test_deterministic_default_honoured_when_allowed() -> None:
    planner = DeterministicPlanner(allowed_actions=frozenset({"echo", "gods_eye_latest_events"}))
    plan = await planner.plan(_goal(), AgentState())
    assert plan.step_types == ["gods_eye_latest_events"]
    assert plan.rationale == "deterministic default action"


async def test_deterministic_default_prefers_research_action() -> None:
    planner = DeterministicPlanner(allowed_actions=frozenset({"web_research", "echo"}))
    plan = await planner.plan(_goal("Ollama repository language"), AgentState())
    assert plan.step_types == ["web_research"]
    assert plan.steps[0].params["goal"]


async def test_deterministic_fallback_empty_when_nothing_allowed() -> None:
    planner = DeterministicPlanner(allowed_actions=frozenset())
    plan = await planner.plan(_goal(), AgentState())
    assert plan.steps == []


async def test_llm_fallback_only_emits_allowed_actions() -> None:
    """The LLM planner's deterministic net never names an unrunnable action."""

    class NoModel:
        async def structured(self, *args: Any, **kwargs: Any) -> Any:
            return None

        async def generate(self, *args: Any, **kwargs: Any) -> str:
            return ""

        def info(self) -> dict[str, Any]:
            return {"provider": "none", "model": "none"}

    planner = LLMPlanner(
        NoModel(),  # type: ignore[arg-type]
        fallback=DeterministicPlanner(allowed_actions=frozenset({"echo"})),
        allowed_actions=frozenset({"echo"}),
    )
    assert planner.plans_are_untrusted is True
    plan = await planner.plan(_goal(), AgentState())
    assert plan.step_types == ["echo"]


# -- 2. plan validation -----------------------------------------------------------------


def _specs(*pairs: tuple[str, dict[str, Any]]) -> list[ActionSpec]:
    return [ActionSpec(type=t, params=p) for t, p in pairs]


def test_validate_plan_rejects_empty_and_unregistered() -> None:
    allowed = frozenset({"echo"})
    registered = frozenset({"echo", "memory_store"})

    empty = validate_plan(Plan(steps=[], rationale="x"), allowed_actions=allowed)
    assert not empty.valid
    assert "empty_plan" in empty.reasons[0]

    bad = validate_plan(
        Plan(steps=_specs(("not_real", {}))),
        allowed_actions=allowed,
        registered_actions=registered,
    )
    assert not bad.valid
    codes = {issue.code for issue in bad.issues}
    assert {"action_not_registered", "action_not_allowed"} <= codes


def test_validate_plan_rejects_control_params_and_step_budget() -> None:
    control = validate_plan(
        Plan(steps=_specs(("echo", {"command": "rm -rf /"}))),
        allowed_actions=frozenset({"echo"}),
    )
    assert not control.valid
    assert control.issues[0].code == "control_param"

    over = validate_plan(
        Plan(steps=_specs(("echo", {}), ("echo", {}))),
        allowed_actions=frozenset({"echo"}),
        max_steps=1,
    )
    assert not over.valid
    assert over.issues[0].code == "step_budget"


def test_validate_plan_accepts_registered_plan() -> None:
    good = validate_plan(
        Plan(steps=_specs(("echo", {"text": "hi"}))),
        allowed_actions=frozenset({"echo"}),
        registered_actions=frozenset({"echo"}),
        max_steps=10,
    )
    assert good.valid and good.step_count == 1


class _UntrustedPlanner(Planner):
    """A planner whose output pretends to be model-derived (untrusted)."""

    name = "untrusted-test"
    plans_are_untrusted = True

    def __init__(self, steps: list[ActionSpec]) -> None:
        self._steps = steps

    async def plan(self, goal: Goal, state: AgentState) -> Plan:
        del goal, state
        return Plan(steps=list(self._steps), rationale="model proposal")


async def _run_with_planner(
    planner: Planner, sink: MemoryTraceSink
) -> tuple[Any, list[str]]:
    core = build_core(_settings(), planner=planner, extra_trace_sinks=(sink,))
    summary = await core.run_goal(GoalSpec(title="validated plan run"))
    return summary, sink.types


async def test_invalid_plan_falls_back_and_executes_safely() -> None:
    """Unregistered model action → rejection, deterministic fallback, success."""
    sink = MemoryTraceSink()
    planner = _UntrustedPlanner(_specs(("totally_made_up", {})))
    summary, event_types = await _run_with_planner(planner, sink)

    assert "PLAN_REJECTED" in event_types
    assert "PLAN_FALLBACK" in event_types
    assert summary.actions and summary.actions[0]["type"] == "echo"
    assert summary.actions[0]["success"] is True
    assert "totally_made_up" not in {a["type"] for a in summary.actions}


async def test_control_param_plan_is_rejected_before_execution() -> None:
    sink = MemoryTraceSink()
    planner = _UntrustedPlanner(_specs(("echo", {"shell": "rm -rf /"})))
    summary, _ = await _run_with_planner(planner, sink)

    rejected = [e for e in sink.events if e.event_type == "PLAN_REJECTED"]
    assert rejected, "expected a plan rejection"
    assert any("control_param" in issue for issue in rejected[0].payload["issues"])
    # The rejected step never executed; the safe fallback did.
    assert summary.actions[0]["type"] == "echo"
    assert "shell" not in str(summary.actions)


async def test_no_safe_fallback_terminates_run_safely() -> None:
    """No executable action at all → nothing runs and the run fails safely."""
    sink = MemoryTraceSink()
    # Allow-list names an action this (memory-storage) build never registers,
    # so allowlist ∩ registry is empty and no fallback can be valid.
    settings = _settings(enabled_actions="gods_eye_latest_events")
    core = build_core(settings, planner=_UntrustedPlanner(_specs(("nope", {}))), extra_trace_sinks=(sink,))
    summary = await core.run_goal(GoalSpec(title="no executable actions"))

    assert summary.status == "failed"
    assert summary.steps_executed == 0
    assert summary.error and "plan rejected" in summary.error
    assert "PLAN_REJECTED" in sink.types


async def test_deterministic_plans_are_not_validated() -> None:
    """A scripted (non-model) plan is honoured; the execution gate still refuses."""
    sink = MemoryTraceSink()
    core = build_core(_settings(), extra_trace_sinks=(sink,))
    summary = await core.run_goal(
        GoalSpec(title="default deterministic plan")
    )
    # Historical refusal path intact: the deterministic default names the
    # God's Eye action, which is unregistered here → refused at execution,
    # not by plan validation (no PLAN_REJECTED event).
    assert summary.actions[0]["type"] == "gods_eye_latest_events"
    assert summary.actions[0]["success"] is False
    assert "PLAN_REJECTED" not in sink.types


# -- 3. multi-source verification -------------------------------------------------------


def _web(
    url: str,
    content: str,
    *,
    query: str = "ollama language",
) -> Evidence:
    return Evidence(
        source=url,
        source_type="web",
        content=content,
        confidence=0.55,
        provenance={
            "query": query,
            "url": url,
            "evidence_class": "LIVE_EXTERNAL_EVIDENCE",
            "retrieved_at": "2026-01-01T00:00:00Z",
        },
    )


def test_verification_unavailable_without_external_evidence() -> None:
    verdict = verify_evidence([Evidence(source="memory:x", source_type="memory")])
    assert verdict["status"] == "NO_EXTERNAL_EVIDENCE"
    assert verdict["independent_sources"] == []


def test_verification_single_source() -> None:
    verdict = verify_evidence(
        [_web("https://one.example/a", "The Ollama project is written in Go and ships models.")]
    )
    assert verdict["status"] == "SINGLE_SOURCE"
    assert verdict["independent_sources"] == ["one.example"]


def test_verification_multi_source_corroborated() -> None:
    text_a = "Ollama is an open source project that runs large language models locally in Go."
    text_b = "Ollama runs large language models locally and is an open source Go project."
    verdict = verify_evidence(
        [
            _web("https://one.example/a", text_a),
            _web("https://two.example/b", text_b),
        ]
    )
    assert verdict["status"] == "MULTI_SOURCE_CORROBORATED"
    assert verdict["independent_sources"] == ["one.example", "two.example"]


def test_verification_detects_candidate_conflict() -> None:
    verdict = verify_evidence(
        [
            _web("https://alpha.example/a", "banana orchard harvesting calendars regional produce"),
            _web("https://beta.example/b", "quantum tunnelling diagrams photolithography substrates"),
        ]
    )
    assert verdict["status"] == "CONFLICTING"
    assert verdict["topics"][0]["conflicts"]


def test_same_host_is_not_two_sources() -> None:
    verdict = verify_evidence(
        [
            _web("https://same.example/a", "Ollama is written in Go and is open source software."),
            _web("https://same.example/b", "Ollama is written in Go and ships local model runners."),
        ]
    )
    assert verdict["status"] == "SINGLE_SOURCE"


async def test_orchestrator_records_verification_and_result(tmp_path: Any) -> None:
    class TwoHosts:
        async def search(self, query: str, *, max_results: int, run_id: str) -> list[Evidence]:
            del run_id
            return [
                _web("https://one.example/a", "Ollama is a Go project running models locally.", query=query),
                _web("https://two.example/b", "Ollama runs models locally and is a Go project.", query=query),
            ]

    orch = make_orchestrator(tmp_path=tmp_path, research=TwoHosts())
    run = await orch.start(
        "Research the Ollama project language",
        budgets=OrchestratorBudgets(max_iterations=5, max_web_requests=10),
        configuration=dict(GOOD_CONFIG),
    )
    run = await orch.execute(run.id)

    assert run.verification["status"] == "MULTI_SOURCE_CORROBORATED"
    assert run.result["verification"]["status"] == "MULTI_SOURCE_CORROBORATED"


# -- 4. persisted mission trace ---------------------------------------------------------


async def test_trace_store_survives_reload(tmp_path: Any) -> None:
    path = tmp_path / "trace.jsonl"
    store = RunTraceStore(path)
    await store.append(
        "AUTONOMOUS_STAGE_STARTED", {"run_id": "run-1", "stage": "research"}
    )
    await store.append(
        "SOURCE_RETRIEVED",
        {"run_id": "run-1", "source": "https://x.example", "status": "LIVE_EXTERNAL_EVIDENCE"},
    )

    # A brand-new instance (process-restart stand-in) reads the same file.
    fresh = RunTraceStore(path)
    records = await fresh.read("run-1")
    assert [r.event_type for r in records] == ["AUTONOMOUS_STAGE_STARTED", "SOURCE_RETRIEVED"]
    assert records[0].state == "research"
    assert records[1].run_id == "run-1"
    assert records[0].event_id != records[1].event_id
    assert await fresh.runs() == ["run-1"]


def test_trace_redaction_masks_keys_and_values() -> None:
    sentinel = "sk-SENTINEL-DO-NOT-LEAK-1234567890"
    redacted = redact_payload(
        {
            "objective": f"use {sentinel}",
            "api_key": sentinel,
            "nested": {"authorization": f"Bearer {sentinel}", "note": "safe"},
        }
    )
    assert redacted["api_key"] == "[REDACTED]"
    assert sentinel not in redacted["objective"]
    assert sentinel not in redacted["nested"]["authorization"]
    assert redacted["nested"]["note"] == "safe"


async def test_orchestrator_persists_trace_and_never_leaks_secrets(tmp_path: Any) -> None:
    sentinel = "sk-SENTINEL-PERSIST-0987654321"
    trace_path = tmp_path / "trace.jsonl"
    settings = _settings()
    core = build_core(settings)
    orch = AutonomousOrchestrator(
        core=core,
        store=RunStore(tmp_path / "runs.jsonl"),
        research=LocalResearchAdapter(_CORPUS),
        comms=LocalCommsAdapter(),
        trace_store=RunTraceStore(trace_path),
    )
    run = await orch.start(
        f"objective carrying {sentinel}",
        budgets=OrchestratorBudgets(max_iterations=5, max_web_requests=10),
        configuration=dict(GOOD_CONFIG),
    )
    run = await orch.execute(run.id)
    assert run.status is RunStatus.COMPLETED

    fresh = RunTraceStore(trace_path)
    records = await fresh.read(run.id)
    types = [r.event_type for r in records]
    assert "AUTONOMOUS_RUN_STARTED" in types
    assert "AUTONOMOUS_RUN_COMPLETED" in types
    assert all(r.run_id == run.id for r in records)
    raw = trace_path.read_text(encoding="utf-8")
    assert sentinel not in raw


def test_cli_real_trace_reads_persisted_stream(tmp_path: Any, monkeypatch: Any, capsys: Any) -> None:
    import asyncio

    from agency.cli import main

    trace_path = tmp_path / "cli_trace.jsonl"
    monkeypatch.setenv("SKYNET_AUTONOMOUS_TRACES_PATH", str(trace_path))
    monkeypatch.setenv("SKYNET_AUTONOMOUS_RUNS_PATH", str(tmp_path / "runs.jsonl"))

    store = RunTraceStore(trace_path)
    asyncio.run(
        store.append("AUTONOMOUS_RUN_STARTED", {"run_id": "abc123def456", "objective": "x"})
    )

    rc = main(["real", "trace", "abc123"])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "AUTONOMOUS_RUN_STARTED" in captured
    assert "abc123" in captured


async def test_save_preserves_cross_process_pause(tmp_path: Any) -> None:
    """A pause applied by another process is never clobbered by a stage save."""
    orch = make_orchestrator(tmp_path=tmp_path)
    store = RunStore(tmp_path / "orchestrator-tests.jsonl")
    run = await orch.start("objective", budgets=OrchestratorBudgets())

    await store.pause(run.id)  # operator, second process
    await orch._save(run)  # driver, in-memory copy still RUNNING

    stored = await store.get(run.id)
    assert stored is not None and stored.status is RunStatus.PAUSED


def test_cli_real_pause_persists_and_resume_requires_paused(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    """CLI pause persists cross-process; resume refuses non-paused runs."""
    import asyncio

    from agency.cli import main

    runs_path = tmp_path / "runs.jsonl"
    monkeypatch.setenv("SKYNET_AUTONOMOUS_RUNS_PATH", str(runs_path))
    monkeypatch.setenv("SKYNET_AUTONOMOUS_TRACES_PATH", str(tmp_path / "trace.jsonl"))

    async def seed() -> str:
        orch = AutonomousOrchestrator(
            core=build_core(_settings()),
            store=RunStore(runs_path),
            research=LocalResearchAdapter(_CORPUS),
            comms=LocalCommsAdapter(),
        )
        run = await orch.start(
            "Research long-term memory architectures",
            budgets=OrchestratorBudgets(max_iterations=5, max_web_requests=10),
            configuration=dict(GOOD_CONFIG),
        )
        return run.id

    run_id = asyncio.run(seed())

    assert main(["real", "pause", run_id]) == 0
    stored = asyncio.run(RunStore(runs_path).list())
    assert len(stored) == 1
    assert stored[0].status is RunStatus.PAUSED

    # Resuming an unknown/non-paused run is an error, never a silent new run.
    assert main(["real", "resume", "does-not-exist"]) == 1
    assert len(asyncio.run(RunStore(runs_path).list())) == 1


def test_llm_plan_payload_schema_is_strict() -> None:
    """The structured schema the planner relies on rejects malformed JSON."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMPlanPayload.model_validate({"steps": [{"params": {}}]})
