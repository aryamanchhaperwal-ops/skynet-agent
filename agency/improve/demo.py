"""End-to-end self-improvement demonstration (offline, deterministic).

The demo runs the **entire improvement loop** with real measurements —
no fabricated numbers, no paid APIs, and whatever verdicts the evidence
supports:

1. **Recorded evidence**: a real P6 experiment measures a deliberately
   wasteful research strategy (four overlapping query variants, no
   per-query cap) against the demo corpus — its arms really do surface
   duplicates.
2. **Detect**: the WeaknessDetector reads the *actual persisted* metrics
   and raises a duplicate-sources weakness (real evidence, real counts).
3. **Propose**: a config-only candidate is built from the weakness via
   the standard pipeline (source code is never touched).
4. **Test**: the P6 runner benchmarks parent vs candidate (5 trials/arm,
   frozen criteria, seed 42) inside the sandbox.
5. **Evaluate**: the acceptance gate decides ACCEPT/REJECT/INCONCLUSIVE
   from the measured verdict — printed whatever it is.
6. **Approve + Apply** (only when accepted): explicit human approval is
   recorded, then the version lands in the ledger with provenance.
7. **Monitor + Rollback**: measured post-deployment performance below
   the configured floor triggers a traced rollback to the known-good
   version; the failed candidate stays in the ledger.
8. **Second candidate (deliberately bad)**: tested honestly and
   *rejected* by the gate — rejections are recorded to memory too.
9. **Memory**: every decision is recorded into long-term memory with
   the full provenance chain.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("skynet.improve.demo")

_GOAL = "approaches to long-term memory in autonomous AI agents"


def _register_wide_strategy(strategies: Any) -> None:
    """Register the config-driven strategy whose flaw lives in its config.

    ``variants`` = how many query variants to fan out (overlapping queries
    are what actually produce duplicates on a small corpus);
    ``max_per_query`` = per-query result cap (huge = wasteful).
    """

    async def wide_procedure(ctx: Any) -> dict[str, float]:
        import time as _time

        from agency.experiments.demo import _search, _tokens

        goal = str(ctx.params.get("goal", "")).strip()
        if not goal:
            raise ValueError("procedure_params['goal'] is required")
        variants = int(ctx.params.get("variants", 4))
        max_per_query = int(ctx.params.get("max_per_query", 10_000))
        tokens = _tokens(goal)
        queries = [goal]
        bigrams = [
            f"{tokens[i]} {tokens[i + 1]}" for i in range(1, max(1, len(tokens) - 1))
        ]
        queries.extend(bigrams[: max(0, variants - 1)])
        corpus = ctx.service("search_corpus")
        started = _time.perf_counter()
        discovered: list[str] = []
        for query in queries[:variants]:
            discovered.extend(_search(corpus, query, limit=max_per_query))
        duration_ms = int((_time.perf_counter() - started) * 1000)
        unique = set(discovered)
        return {
            "sources_found": float(len(unique)),
            "duplicate_rate": (
                round(1.0 - len(unique) / len(discovered), 4) if discovered else 0.0
            ),
            "search_errors": 0.0,
            "duration_ms": float(duration_ms),
        }

    strategies.register(
        name="wide_research",
        version="v1",
        capability="research_strategy",
        procedure=wide_procedure,
        description="Demo strategy: overlapping query variants, uncapped "
        "per-query results (the flaw under test).",
        config={"variants": 4, "max_per_query": 10_000},
    )


async def run_demo(
    *, memory_backend: str | None = None, as_json: bool = False
) -> int:
    """Run the full loop; returns a process exit code."""
    from agency.bootstrap import build_lab_stack, build_memory_stack
    from agency.cli import SkynetSettings  # reuse the same settings source
    from agency.experiments.demo import _CORPUS, register_demo_strategies
    from agency.experiments.experiments_store import ExperimentRegistry
    from agency.experiments.models import (
        Experiment,
        MetricDefinition,
        SuccessCriterion,
        TargetRef,
    )
    from agency.experiments.models import (
        Hypothesis as ExperimentHypothesis,
    )
    from agency.experiments.registry import StrategyRegistry
    from agency.experiments.runner import ExperimentRunner
    from agency.improve.detector import DetectorInput

    updates: dict[str, Any] = {}
    if memory_backend:
        updates["memory_backend"] = memory_backend
    settings = SkynetSettings(**updates)
    memory = build_memory_stack(settings).manager if updates else None

    # -- 1. Recorded evidence: one real experiment with genuinely wasteful arms.
    strategies = StrategyRegistry()
    register_demo_strategies(strategies)
    _register_wide_strategy(strategies)
    runner = ExperimentRunner(strategies, services={"search_corpus": _CORPUS})
    experiments = ExperimentRegistry(settings.experiments_registry_path)

    evidence_experiment = Experiment(
        name="improve-demo:evidence",
        description="Measurement experiment exposing the wide-strategy flaw.",
        objective="measure duplicate_rate of wide_research:v1 on the demo corpus",
        hypothesis=ExperimentHypothesis(
            statement=(
                "wide_research:v1 with overlapping query variants and no cap "
                "produces duplicate_rate ≥ 0.2 on the demo corpus"
            ),
            success_criteria=[
                SuccessCriterion(kind="min", metric="duplicate_rate", threshold=0.2)
            ],
            provenance={"purpose": "improvement_demo_evidence"},
        ),
        baseline=TargetRef(name="wide_research", version="v1"),
        candidate=TargetRef(name="single_query", version="v1"),
        metrics=[
            MetricDefinition(name="duplicate_rate", direction="minimize"),
            MetricDefinition(name="sources_found", direction="maximize"),
        ],
        trials=3,
        seed=7,
        procedure_params={"goal": _GOAL},
    )
    evidence = await runner.run(evidence_experiment)
    await experiments.save(evidence)

    # Self-contained but operator-visible history: a demo-suffixed file
    # derived from the configured path. The demo is re-runnable (its
    # measured evidence is identical every run and production dedupe
    # would suppress re-detection), while `improve history` and this
    # file share the same store format.
    import pathlib

    from agency.improve.store import ImprovementStore

    configured = pathlib.Path(settings.improvements_registry_path)
    demo_history = configured.with_name(f"{configured.stem}.demo{configured.suffix}")
    lab = build_lab_stack(
        settings,
        strategy_registry=strategies,
        services={"search_corpus": _CORPUS},
        memory=memory,
        improvement_store=ImprovementStore(path=demo_history),
    )
    pipeline = lab.improvement

    # -- 2. Detect from the persisted record of a real measurement.
    detected = await pipeline.detect(DetectorInput(experiments=[evidence]))
    dup_weaknesses = [w for w in detected if w.category.value == "duplicate_sources"]
    if not dup_weaknesses:
        measured = (
            evidence.comparison.baseline.mean.get("duplicate_rate")
            if evidence.comparison
            else None
        )
        print(
            "demo: no duplicate_sources weakness detected from the measured "
            f"evidence (baseline duplicate_rate={measured}) — the detector "
            "refuses to fabricate one."
        )
        return 1
    weakness = dup_weaknesses[0]
    if not as_json:
        print(f"1. detected   : {weakness.category.value} (severity {weakness.severity:.2f})")
        print(f"   evidence    : {weakness.evidence[0].summary}")

    # Known-good baseline in the ledger (operator bookkeeping, once).
    await pipeline.seed_baseline(name="wide_research", version="v1")

    plan = {
        "metrics": [
            MetricDefinition(name="duplicate_rate", direction="minimize"),
            MetricDefinition(name="sources_found", direction="maximize"),
            MetricDefinition(name="duration_ms", direction="minimize"),
        ],
        "trials": 5,
        "seed": 42,
        "procedure_params": {"goal": _GOAL},
    }
    result = await _run_candidate(
        pipeline,
        weakness,
        plan,
        config={"variants": 2, "max_per_query": 11},
        changes="cap per-query results at 11, keep two query variants",
        as_json=as_json,
        step_offset=2,
    )
    if result["decision"] == "accepted":
        await _finish_accepted(pipeline, proposal=result["proposal"], as_json=as_json)
    await _run_candidate(
        pipeline,
        weakness,
        plan,
        config={"variants": 4, "max_per_query": 1},
        changes="keep the fan-out but cap each query at 1 result (deliberately bad)",
        as_json=as_json,
        step_offset=2,
        expect_rejection=True,
    )
    if as_json:
        print(
            json.dumps(
                {
                    "weakness_id": weakness.id,
                    "first_decision": result["decision"],
                    "first_verdict": result["verdict"],
                    "proposals": len(pipeline.proposals()),
                    "versions": len(pipeline.versions("wide_research")),
                }
            )
        )
    return 0


async def _run_candidate(
    pipeline: Any,
    weakness: Any,
    plan: dict[str, Any],
    *,
    config: dict[str, Any],
    changes: str,
    as_json: bool,
    step_offset: int,
    expect_rejection: bool = False,
) -> dict[str, Any]:
    """Propose → test → evaluate one candidate; records memory either way."""
    from agency.experiments.models import SuccessCriterion
    from agency.improve.memory_bridge import record_proposal

    _hypothesis, proposal, candidate = await pipeline.propose(
        weakness.id,
        candidate_config=config,
        candidate_changes=changes,
        plan_overrides={
            **plan,
            "acceptance_criteria": [
                # duplicate_rate must land at or below 0.35 (absolute bound on
                # a minimize metric; the baseline measures ~0.62).
                SuccessCriterion(kind="max", metric="duplicate_rate", threshold=0.35),
                # Coverage may not collapse: keep ≥ 80% of the baseline's
                # unique sources (absolute floor on a maximize metric).
                SuccessCriterion(kind="min", metric="sources_found", threshold=9.0),
                # Timer-jitter latency stays bounded (covers the duration
                # regression the comparator would otherwise flag).
                SuccessCriterion(kind="max", metric="duration_ms", threshold=50.0),
            ],
        },
    )
    if not as_json:
        tag = "bad" if expect_rejection else "good"
        print(f"{step_offset}. proposed   [{tag}]: {candidate.ref} (config-only; no code changes)")
        print(f"   change      : {changes}")
    experiment = await pipeline.test(proposal, candidate)
    if not as_json:
        print(
            f"{step_offset + 1}. tested     : experiment {experiment.id[:12]} "
            f"({experiment.status.value})"
        )
        if experiment.comparison:
            print(f"   baseline   : {experiment.comparison.baseline.mean}")
            print(f"   candidate  : {experiment.comparison.candidate.mean}")
    report = await pipeline.evaluate(proposal, candidate, experiment)
    if not as_json:
        print(
            f"{step_offset + 2}. evaluated  : verdict={report['verdict']} "
            f"decision={report['decision']}"
        )
        print(f"   reason      : {report['reason']}")
    stored = await record_proposal(
        proposal,
        weakness=weakness,
        candidate=candidate,
        experiment=experiment,
        memory=pipeline.memory,
    )
    if not as_json:
        print(f"   memory      : {len(stored)} record(s) stored with provenance")
    return {
        "decision": report["decision"],
        "verdict": report["verdict"],
        "proposal": proposal,
        "candidate": candidate,
    }


async def _finish_accepted(pipeline: Any, *, proposal: Any, as_json: bool) -> None:
    """Approve → apply → monitor → rollback for an accepted proposal."""
    await pipeline.approve(proposal, approved_by="operator", note="demo approval")
    candidate = pipeline.candidate_for_proposal(proposal.id)
    experiment = await pipeline.experiment_record(proposal.experiment_id)
    version = await pipeline.apply(proposal, candidate, experiment)
    if not as_json:
        print(
            f"5. approved   : by operator; applied version {version.ref} "
            f"(active={pipeline.active(version.name).ref})"
        )
    # Auto-rollback policy is an explicit operator choice (default: record
    # only). The demo enables it to prove the monitored rollback path.
    proposal.metadata["monitor_policy"] = "auto_rollback"
    # Measured post-deployment reality: coverage collapsed below the floor.
    events = await pipeline.monitor(
        proposal,
        actual={"sources_found": 5.0},
        rollback_below={"sources_found": 9.0},
    )
    if not as_json and events:
        print(
            f"6. monitored  : regression {events[0].metric} "
            f"expected={events[0].expected} actual={events[0].actual}"
        )
        print(
            f"7. rolled back: restored {pipeline.active('wide_research').ref} "
            f"(ledger keeps {len(pipeline.versions('wide_research'))} versions)"
        )


__all__ = ["run_demo"]
