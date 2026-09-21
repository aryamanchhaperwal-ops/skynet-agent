"""Long-term memory tests: stores, manager, extraction, consolidation, actions."""

from __future__ import annotations

import pytest

from agency.actions.base import ActionContext
from agency.memory.extraction import (
    Consolidator,
    memories_from_conversation,
    memories_from_observations,
)
from agency.memory.manager import MemoryManager
from agency.memory.models import MemoryRecord, MemoryType
from agency.memory.stores import InMemoryStore, SQLiteMemoryStore, build_memory_store
from agency.observation import Observation


def make_manager(**overrides) -> MemoryManager:
    return MemoryManager(
        InMemoryStore(),
        importance_threshold=overrides.pop("threshold", 0.4),
        max_results=overrides.pop("max_results", 10),
        **overrides,
    )


class TestStores:
    async def test_save_and_get_roundtrip(self):
        store = InMemoryStore()
        record = await store.save(
            MemoryRecord(
                content="Vector stores win for recency",
                type=MemoryType.SEMANTIC,
                source="web:wikipedia.org",
                tags=["memory"],
            )
        )
        fetched = await store.get(record.id)
        assert fetched is not None
        assert fetched.content == record.content
        assert fetched.type is MemoryType.SEMANTIC

    async def test_delete_forgets(self):
        store = InMemoryStore()
        record = await store.save(
            MemoryRecord(content="x")
        )
        assert await store.delete(record.id) is True
        assert await store.get(record.id) is None
        assert await store.delete(record.id) is False

    async def test_sqlite_roundtrip_and_restart_persistence(self, tmp_path):
        db = tmp_path / "memory.db"
        store = SQLiteMemoryStore(db)
        record = await store.save(
            MemoryRecord(
                content="SQLite survives restarts",
                type=MemoryType.EPISODIC,
                source="test",
                tags=["persistence"],
                importance=0.9,
            )
        )
        # Simulate a process restart: a brand-new store instance over the file.
        fresh = SQLiteMemoryStore(db)
        fetched = await fresh.get(record.id)
        assert fetched is not None
        assert fetched.content == "SQLite survives restarts"

    async def test_sqlite_search_filters(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path / "m.db")
        for i, (tag, importance) in enumerate(
            [("alpha", 0.9), ("beta", 0.3), ("alpha", 0.5)]
        ):
            await store.save(
                MemoryRecord(
                    content=f"memory {i} about {tag}",
                    tags=[tag],
                    importance=importance,
                )
            )
        results = await store.search(min_importance=0.4, limit=10)
        assert len(results) == 2
        assert all(r.importance >= 0.4 for r, _ in results)

    async def test_factory_builds_configured_backend(self, tmp_path):
        assert isinstance(build_memory_store("memory", sqlite_path="x"), InMemoryStore)
        store = build_memory_store("sqlite", sqlite_path=str(tmp_path / "m.db"))
        assert isinstance(store, SQLiteMemoryStore)
        with pytest.raises(ValueError, match="unknown memory backend"):
            build_memory_store("weird", sqlite_path="x")


class TestMemoryManager:
    async def test_store_below_threshold_rejected(self):
        manager = make_manager(threshold=0.6)
        record = await manager.store(content="low value", importance=0.3)
        assert record is None

    async def test_store_force_bypasses_threshold(self):
        manager = make_manager(threshold=0.9)
        record = await manager.store(content="operator insert", importance=0.1, force=True)
        assert record is not None

    async def test_unknown_origin_normalized(self):
        manager = make_manager()
        record = await manager.store(content="x", origin="sneaky-web-injected")
        assert record.origin == "system"

    async def test_recall_respects_char_budget(self):
        manager = MemoryManager(
            InMemoryStore(), max_results=10, recall_max_chars=100
        )
        for i in range(5):
            await manager.store(
                content="x" * 80, importance=0.9, summary=f"memory {i}"
            )
        recalled = await manager.recall("memory")
        total = sum(len(m.content) + len(m.summary) for m in recalled)
        assert total <= 100 * 5  # sanity
        assert len(recalled) < 5  # budget cut something

    async def test_search_returns_scored_pairs(self):
        manager = make_manager()
        await manager.store(content="vector stores for agent memory", importance=0.9)
        await manager.store(content="prefrontal cortex maps to working memory", importance=0.5)
        results = await manager.search(query="vector stores memory")
        assert results
        assert results[0][0].content.startswith("vector stores")
        assert 0.0 <= results[0][1] <= 1.0

    async def test_forget_removes_memory(self):
        manager = make_manager()
        record = await manager.store(content="to be forgotten", force=True)
        assert await manager.forget(record.id) is True
        assert await manager.get(record.id) is None

    async def test_stats_counts_by_type(self):
        manager = make_manager()
        await manager.store(content="a", memory_type="semantic", force=True)
        await manager.store(content="b", memory_type="semantic", force=True)
        await manager.store(content="c", memory_type="procedural", force=True)
        stats = await manager.stats()
        assert stats.total == 3
        assert stats.by_type["semantic"] == 2


class TestExtraction:
    async def test_observations_become_semantic_memories(self):
        observation = Observation(
            source="web:en.wikipedia.org",
            kind="text",
            content="Agent memory systems combine episodic and semantic stores.",
            summary="Memory system overview",
            confidence=0.8,
        )
        candidates = memories_from_observations([observation])
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.type is MemoryType.SEMANTIC
        assert candidate.source == "web:en.wikipedia.org"
        assert candidate.provenance["observation_id"] == observation.id
        assert candidate.importance == pytest.approx(0.8)

    async def test_error_observations_become_episodic_with_high_importance(self):
        observation = Observation(
            source="tool:web_fetch", kind="error", content="HTTP 500", summary="fetch failed"
        )
        candidate = memories_from_observations([observation])[0]
        assert candidate.type is MemoryType.EPISODIC
        assert candidate.importance >= 0.7

    async def test_conversation_yields_transcript_and_claims(self):
        from types import SimpleNamespace

        outcome = SimpleNamespace(
            conversation=SimpleNamespace(
                id="conv-1",
                participant_id="mock:agent-1",
                goal_id=None,
                run_id=None,
            ),
            status=SimpleNamespace(value="completed"),
            turns=[
                SimpleNamespace(
                    question="q",
                    answer="a",
                    message_ids=("m1", "m2"),
                    analysis_claims=["Retrieval quality matters most."],
                )
            ],
        )
        candidates = memories_from_conversation(outcome, goal_id="g", run_id="r")
        assert len(candidates) == 2  # one transcript pointer + one claim
        transcript, claim = candidates
        assert transcript.type is MemoryType.EPISODIC
        assert transcript.provenance["conversation_id"] == "conv-1"
        assert claim.type is MemoryType.SEMANTIC
        assert claim.provenance["verified"] is False  # never auto-trust AI claims
        assert claim.provenance["message_id"] == "m2"


class TestConsolidation:
    async def test_recurring_tag_patterns_proposed(self):
        manager = make_manager()
        for i in range(4):
            await manager.store(
                content=f"web research run {i} failed on robots",
                memory_type="episodic",
                tags=["web", "error"],
                importance=0.8,
                force=True,
            )
        existing = await manager.search(limit=20)
        consolidator = Consolidator(min_group_size=3)
        candidates = await consolidator.identify_candidates(existing)
        assert candidates, "recurring pattern should be detected"
        proposal = candidates[0]
        assert proposal.type is MemoryType.SEMANTIC
        assert proposal.origin == "consolidation"
        assert len(proposal.provenance["source_memory_ids"]) >= 3

    async def test_no_pattern_no_proposal(self):
        manager = make_manager()
        await manager.store(content="one off", tags=["unique"], force=True)
        existing = await manager.search(limit=20)
        assert await Consolidator().identify_candidates(existing) == []


class TestMemoryActions:
    async def make_ctx(self):
        from agency.actions.base import ActionContext

        return ActionContext(run_id="run-1", goal_id="goal-1", state_snapshot={})

    async def test_store_and_search_through_actions(self):
        from agency.actions.base import ActionSpec
        from agency.memory.actions import MemorySearchAction, MemoryStoreAction

        manager = make_manager()
        store_action = MemoryStoreAction(manager)
        output = await store_action.execute(
            ActionSpec(
                type="memory_store",
                params={
                    "content": "Wikipedia is a good first source for concepts",
                    "type": "semantic",
                    "importance": 0.8,
                    "tags": ["research", "sources"],
                },
            ),
            await self.make_ctx(),
        )
        assert output["stored"] is True
        search = MemorySearchAction(manager)
        results = await search.execute(
            ActionSpec(type="memory_search", params={"query": "wikipedia sources"}),
            await self.make_ctx(),
        )
        assert results["count"] == 1

    async def test_store_below_threshold_reports_rejection(self):
        from agency.actions.base import ActionSpec
        from agency.memory.actions import MemoryStoreAction

        action = MemoryStoreAction(make_manager(threshold=0.9))
        output = await action.execute(
            ActionSpec(type="memory_store", params={"content": "x", "importance": 0.2}),
            await self.make_ctx(),
        )
        assert output["stored"] is False
        assert "below threshold" in output["reason"]

    async def test_extract_from_state_snapshot_observations(self):
        from agency.actions.base import ActionSpec
        from agency.memory.actions import MemoryExtractAction
        from agency.observation import Observation

        manager = make_manager()
        observation = Observation(
            source="web:example.com",
            kind="text",
            content="Extracted finding about agent memory",
            confidence=0.85,
        ).model_dump(mode="json")
        action = MemoryExtractAction(manager)
        output = await action.execute(
            ActionSpec(type="memory_extract"),
            ActionContext(
                run_id="run-1",
                goal_id="goal-1",
                state_snapshot={"observations": [observation]},
            ),
        )
        assert output["candidates"] == 1
        assert output["stored"] == 1
        stored = (await manager.search(query="extracted finding"))[0][0]
        assert stored.provenance["observation_id"] == observation["id"]
