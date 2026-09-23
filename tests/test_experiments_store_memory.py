"""Experiments: persistence, memory bridge, loop/actions integration, security."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agency.experiments.experiments_store import ExperimentRegistry
from agency.experiments.memory_bridge import record_experiment
from agency.experiments.models import (
    Experiment,
    ExperimentStatus,
    Hypothesis,
    MetricDefinition,
    SuccessCriterion,
    TargetRef,
    Verdict,
)
from agency.memory.models import MemoryType
from agency.memory.stores import InMemoryStore


def _experiment(**overrides: object) -> Experiment:
    base: dict[str, object] = {
        "name": "persist-test",
        "hypothesis": Hypothesis(
            statement="v2 beats v1",
            success_criteria=[
                SuccessCriterion(kind="improves", metric="score", threshold=0.1)
            ],
        ),
        "baseline": TargetRef(name="strat", version="v1"),
        "candidate": TargetRef(name="strat", version="v2"),
        "metrics": [MetricDefinition(name="score", direction="maximize")],
        "trials": 3,
    }
    base.update(overrides)
    return Experiment(**base)  # type: ignore[arg-type]


def _finished_experiment(verdict: Verdict) -> Experiment:
    experiment = _experiment()
    experiment.criteria_frozen()
    experiment.status = ExperimentStatus.COMPLETED
    from agency.experiments.metrics import aggregate
    from agency.experiments.models import (
        ComparisonReport,
        ExperimentEvaluation,
        TrialResult,
    )

    trials = [
        TrialResult(arm="baseline", index=0, metrics={"score": 1.0}),
        TrialResult(arm="candidate", index=0, metrics={"score": 2.0}),
    ]
    experiment.trials_results = trials
    baseline = aggregate([t for t in trials if t.arm == "baseline"], ["score"])
    candidate = aggregate([t for t in trials if t.arm == "candidate"], ["score"])
    experiment.comparison = ComparisonReport(
        baseline=baseline,
        candidate=candidate,
        delta={"score": 1.0},
        relative={"score": 1.0},
        improvements=["score"] if verdict is Verdict.SUCCESS else [],
        regressions=[] if verdict is Verdict.SUCCESS else ["score"],
        significance_tested=False,
    )
    experiment.evaluation = ExperimentEvaluation(
        verdict=verdict, explanation=f"test {verdict.value}"
    )
    experiment.completed_at = experiment.created_at
    return experiment


# -- persistence -------------------------------------------------------------------


class TestExperimentRegistry:
    async def test_save_and_get_roundtrip(self, tmp_path: Path) -> None:
        store = ExperimentRegistry(tmp_path / "experiments.jsonl")
        experiment = _experiment()
        await store.save(experiment)
        loaded = await store.get(experiment.id)
        assert loaded is not None
        assert loaded.hypothesis.statement == "v2 beats v1"
        assert loaded.baseline.ref == "strat:v1"

    async def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "experiments.jsonl"
        store1 = ExperimentRegistry(path)
        experiment = _experiment()
        await store1.save(experiment)
        # New instance = fresh process simulation.
        store2 = ExperimentRegistry(path)
        loaded = await store2.get(experiment.id)
        assert loaded is not None
        assert loaded.id == experiment.id

    async def test_list_summaries(self, tmp_path: Path) -> None:
        store = ExperimentRegistry(tmp_path / "experiments.jsonl")
        await store.save(_finished_experiment(Verdict.SUCCESS))
        await store.save(_finished_experiment(Verdict.FAILURE))
        summaries = await store.list()
        assert len(summaries) == 2
        assert {s.verdict for s in summaries} == {Verdict.SUCCESS, Verdict.FAILURE}
        assert all(s.trials == 2 for s in summaries)  # fixture adds two trials

    async def test_list_status_filter(self, tmp_path: Path) -> None:
        store = ExperimentRegistry(tmp_path / "experiments.jsonl")
        done = _finished_experiment(Verdict.SUCCESS)
        created = _experiment()
        await store.save(done)
        await store.save(created)
        only_created = await store.list(status="created")
        assert len(only_created) == 1
        assert only_created[0].id == created.id

    async def test_search_keyword(self, tmp_path: Path) -> None:
        store = ExperimentRegistry(tmp_path / "experiments.jsonl")
        matching = _experiment(name="memory coverage trial",
                               hypothesis=Hypothesis(statement="more sources"))
        await store.save(matching)
        await store.save(_experiment(name="unrelated thing"))
        hits = await store.search("memory coverage")
        assert [e.id for e in hits] == [matching.id]

    async def test_prior_for_exact_pair(self, tmp_path: Path) -> None:
        store = ExperimentRegistry(tmp_path / "experiments.jsonl")
        experiment = _experiment()
        await store.save(experiment)
        prior = await store.prior_for("strat:v1", "strat:v2")
        assert [e.id for e in prior] == [experiment.id]
        # Order-independent.
        prior_reversed = await store.prior_for("strat:v2", "strat:v1")
        assert [e.id for e in prior_reversed] == [experiment.id]
        assert await store.prior_for("other:v1", "strat:v2") == []

    async def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "experiments.jsonl"
        good = _experiment()
        path.write_text(
            "not json\n" + json.dumps({"id": "nope"}) + "\n"
            + good.model_dump_json() + "\n",
            encoding="utf-8",
        )
        store = ExperimentRegistry(path)
        assert await store.get(good.id) is not None
        assert len(await store.list()) == 1


# -- memory bridge -------------------------------------------------------------------


class TestMemoryBridge:
    async def test_success_writes_episodic_and_procedural(self) -> None:
        manager = _manager()
        stored = await record_experiment(_finished_experiment(Verdict.SUCCESS), manager)
        assert len(stored) == 2
        types = {record.type for record in stored}
        assert types == {MemoryType.EPISODIC, MemoryType.PROCEDURAL}
        procedural = next(r for r in stored if r.type is MemoryType.PROCEDURAL)
        # Provenance is mandatory.
        assert procedural.provenance["experiment_id"]
        assert procedural.provenance["candidate"] == "strat:v2"
        assert procedural.provenance["baseline"] == "strat:v1"
        assert procedural.provenance["verdict"] == "success"
        # The claim is bounded, not eternal truth.
        assert "not a guarantee" in procedural.content

    @pytest.mark.parametrize(
        "verdict,count",
        [
            (Verdict.FAILURE, 1),
            (Verdict.INCONCLUSIVE, 1),
            (Verdict.ERROR, 1),
            (Verdict.CANCELLED, 1),
        ],
    )
    async def test_non_success_writes_episodic_only(
        self, verdict: Verdict, count: int
    ) -> None:
        manager = _manager()
        stored = await record_experiment(_finished_experiment(verdict), manager)
        assert len(stored) == count
        assert stored[0].type is MemoryType.EPISODIC

    async def test_no_memory_manager_is_noop(self) -> None:
        stored = await record_experiment(_finished_experiment(Verdict.SUCCESS), None)
        assert stored == []

    async def test_running_experiment_not_recorded(self) -> None:
        manager = _manager()
        experiment = _experiment()  # status=created (not terminal)
        stored = await record_experiment(experiment, manager)
        assert stored == []

    async def test_memory_failure_never_breaks_recording(self) -> None:
        class FailingStore(InMemoryStore):
            async def save(self, record):
                raise RuntimeError("disk on fire")

        from agency.memory.manager import MemoryManager

        manager = MemoryManager(FailingStore())
        stored = await record_experiment(_finished_experiment(Verdict.SUCCESS), manager)
        assert stored == []  # swallowed; experiment pipeline unaffected


def _manager() -> object:
    from agency.memory.manager import MemoryManager

    return MemoryManager(InMemoryStore())
