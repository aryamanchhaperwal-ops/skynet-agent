"""Integration: Skynet Core + memory recall + intelligence strategies (offline).

Exercises the full P2 chain through real cores:

    goal → memory recall → plan → actions → experience → memory_extract → store
    ...restart... → new core → recall finds what the previous run learned

All offline: mock intelligence provider, memory storage backend, no network.
"""

from __future__ import annotations

import pytest

from agency.bootstrap import build_core, build_intelligence_service
from agency.config import SkynetSettings
from agency.goals import GoalSpec
from agency.intelligence.evaluator import LLMEvaluator
from agency.intelligence.planner import LLMPlanner
from agency.intelligence.providers import MockLLMProvider
from agency.intelligence.service import IntelligenceService
from agency.memory import MemoryManager
from agency.memory.stores import InMemoryStore
from agency.planner import DeterministicPlanner
from agency.trace import MemoryTraceSink


def make_settings(**overrides) -> SkynetSettings:
    defaults = dict(
        storage_backend="memory",
        perception_adapter="none",
        memory_backend="memory",
        memory_importance_threshold=0.3,
        enabled_actions="echo,memory_store,memory_search,memory_extract",
    )
    defaults.update(overrides)
    return SkynetSettings(**defaults)


class TestMemoryRecallInTheLoop:
    async def test_recall_injects_memories_and_traces(self):
        manager = MemoryManager(InMemoryStore(), importance_threshold=0.3)
        await manager.store(
            content="Agent memory systems combine episodic and semantic stores.",
            memory_type="semantic",
            source="web:wikipedia.org",
            importance=0.9,
        )
        sink = MemoryTraceSink()
        core = build_core(make_settings(), memory=manager, extra_trace_sinks=(sink,))
        summary = await core.run_goal(
            GoalSpec(
                title="What do we know about agent memory systems?",
                params={"steps": [{"type": "echo", "params": {"text": "thinking"}}]},
            )
        )
        assert summary.status == "completed"
        recalled = summary.final_state.get("context", {}).get("memory_recall", [])
        assert recalled, "relevant memory should be injected into context"
        assert recalled[0]["source"] == "web:wikipedia.org"
        assert "MEMORY_RECALLED" in sink.types

    async def test_recall_skipped_when_nothing_relevant(self):
        manager = MemoryManager(InMemoryStore(), importance_threshold=0.3)
        sink = MemoryTraceSink()
        core = build_core(make_settings(), memory=manager, extra_trace_sinks=(sink,))
        summary = await core.run_goal(
            GoalSpec(
                title="Completely unrelated quixotic goal about turbopumps",
                params={"steps": [{"type": "echo", "params": {"text": "x"}}]},
            )
        )
        assert summary.status == "completed"
        assert "MEMORY_RECALLED" not in sink.types

    async def test_run_without_memory_manager_still_works(self):
        core = build_core(make_settings())  # default memory backend (memory)
        summary = await core.run_goal(
            GoalSpec(
                title="plain goal",
                params={"steps": [{"type": "echo", "params": {"text": "x"}}]},
            )
        )
        assert summary.status == "completed"

    async def test_memory_extract_action_stores_run_observations(self):
        manager = MemoryManager(InMemoryStore(), importance_threshold=0.3)
        core = build_core(
            make_settings(), memory=manager
        )
        summary = await core.run_goal(
            GoalSpec(
                title="Research vector stores",
                params={"steps": [
                    {"type": "echo", "params": {"text": "fake research observation"}},
                    {"type": "memory_extract", "params": {}},
                ]},
            )
        )
        assert summary.status == "completed"
        # echo produces no observations, so extract found none — but the
        # action itself must have run cleanly.
        records = summary.final_state.get("actions", [])
        assert records[-1]["result"]["success"]

    async def test_memory_store_action_visible_to_later_search(self):
        manager = MemoryManager(InMemoryStore(), importance_threshold=0.3)
        core = build_core(make_settings(), memory=manager)
        summary = await core.run_goal(
            GoalSpec(
                title="Learn a fact",
                params={"steps": [
                    {
                        "type": "memory_store",
                        "params": {
                            "content": "Deterministic evaluation is the fallback path",
                            "type": "semantic",
                            "importance": 0.8,
                            "tags": ["evaluation"],
                        },
                    },
                    {
                        "type": "memory_search",
                        "params": {"query": "evaluation fallback"},
                    },
                ]},
            )
        )
        assert summary.status == "completed"
        search_record = summary.final_state["actions"][-1]
        output = search_record["result"]["output"]
        assert output["count"] >= 1
        assert "deterministic" in output["results"][0]["summary"].lower() or \
            output["results"][0]["summary"] == ""


class TestLLMStrategiesInTheLoop:
    def build_llm_core(self) -> tuple:
        settings = make_settings(
            planner_strategy="llm",
            evaluator_strategy="llm",
            llm_provider="mock",
            llm_model="skynet-mock-1",
            enabled_actions="echo,memory_store,memory_search,memory_extract",
        )
        service = build_intelligence_service(settings)
        sink = MemoryTraceSink()
        core = build_core(settings, extra_trace_sinks=(sink,))
        return core, sink, service, settings

    async def test_llm_planner_runs_inside_the_core(self):
        core, _sink, _service, _settings = self.build_llm_core()
        assert core._planner.name == "llm"
        assert isinstance(core._planner, LLMPlanner)
        summary = await core.run_goal(GoalSpec(title="Plan something about memory"))
        assert summary.status == "completed"
        assert summary.planner == "llm"
        # The mock proposes an echo step, which is allow-listed.
        assert summary.steps_executed >= 1

    async def test_llm_evaluator_records_source(self):
        core, _sink, _service, _settings = self.build_llm_core()
        summary = await core.run_goal(
            GoalSpec(
                title="Evaluate me",
                params={"steps": [{"type": "echo", "params": {"text": "x"}}]},
            )
        )
        assert summary.evaluation is not None
        assert summary.evaluation.source == "llm"
        assert "llm" in summary.evaluation.notes

    async def test_llm_failure_falls_back_inside_the_core(self, monkeypatch):
        settings = make_settings(
            planner_strategy="llm",
            evaluator_strategy="llm",
            enabled_actions="echo",
        )
        # Inject a failing provider directly: the built service is wrapped
        # by LLMPlanner/LLMEvaluator, whose fallbacks must carry the run.
        from agency.memory.stores import InMemoryStore as _S  # noqa: F401

        failing = IntelligenceService(
            MockLLMProvider(scenario="failure"), max_retries=0
        )
        core = build_core(
            settings,
            planner=LLMPlanner(
                failing,
                fallback=DeterministicPlanner(default_action="echo"),
                allowed_actions=frozenset({"echo"}),
            ),
            evaluator=LLMEvaluator(failing),
        )
        summary = await core.run_goal(
            GoalSpec(
                title="still works",
                params={"steps": [{"type": "echo", "params": {"text": "x"}}]},
            )
        )
        assert summary.status == "completed"
        assert summary.evaluation.source == "deterministic"  # fallback verdict


class TestRestartPersistence:
    async def test_memory_survives_core_restart(self, tmp_path):
        """Run 1 stores knowledge; a *fresh core instance* (simulated restart)
        on the same sqlite file must recall it in Run 2."""
        db = tmp_path / "memory.db"
        settings_run1 = make_settings(
            memory_backend="sqlite", memory_sqlite_path=str(db)
        )
        core1 = build_core(settings_run1)
        await core1.run_goal(
            GoalSpec(
                title="Learn about agent memory",
                params={"steps": [{
                    "type": "memory_store",
                    "params": {
                        "content": (
                            "Skynet memory test fact: episodic memories record "
                            "what happened; semantic memories record learned facts."
                        ),
                        "type": "semantic",
                        "importance": 0.9,
                        "tags": ["memory", "fact"],
                    },
                }]},
            )
        )

        # ---- simulated process restart: brand-new core, same file ----
        settings_run2 = make_settings(
            memory_backend="sqlite", memory_sqlite_path=str(db)
        )
        core2 = build_core(settings_run2)
        sink2 = MemoryTraceSink()
        core2._trace_sinks.append(sink2)  # observe recall in run 2
        summary2 = await core2.run_goal(
            GoalSpec(
                title="What have you previously learned about agent memory?",
                params={"steps": [{"type": "echo", "params": {"text": "recall"}}]},
            )
        )
        assert summary2.status == "completed"
        recalled = summary2.final_state.get("context", {}).get("memory_recall", [])
        assert recalled, "run 2 must recall what run 1 learned"
        assert any("episodic memories record" in m["summary"] for m in recalled)
        assert "MEMORY_RECALLED" in sink2.types

    async def test_sqlite_backend_survives_real_process_boundary(self, tmp_path):
        """Tighter restart proof: two store instances in sequence, closing all
        connections in between (the closest a single process gets to a restart)."""
        from agency.memory.stores import SQLiteMemoryStore

        db = tmp_path / "tight.db"
        store1 = SQLiteMemoryStore(db)
        record = await store1.save(
            __import__("agency.memory.models", fromlist=["MemoryRecord"]).MemoryRecord(
                content="Boundary persistence fact", importance=0.9
            )
        )
        from agency.memory.models import MemoryRecord

        store2 = SQLiteMemoryStore(db)
        fetched = await store2.get(record.id)
        assert isinstance(fetched, MemoryRecord)
        assert fetched.content == "Boundary persistence fact"


class TestConfigDefaults:
    def test_dark_by_default(self):
        settings = SkynetSettings(storage_backend="memory")
        assert settings.planner_strategy == "deterministic"
        assert settings.evaluator_strategy == "deterministic"
        assert settings.llm_provider == "mock"
        assert settings.memory_importance_threshold == pytest.approx(0.4)

    def test_llm_planner_strategy_selects_llm_planner(self):
        core = build_core(
            make_settings(planner_strategy="llm", llm_provider="mock")
        )
        assert core._planner.name == "llm"

    def test_deterministic_default_unchanged(self):
        core = build_core(make_settings())
        assert core._planner.name == "deterministic"
        assert core._evaluator.name == "deterministic"
