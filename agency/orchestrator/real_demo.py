"""Bounded REAL-capability demonstration (Phase P9).

Proves one configurable objective can travel the full loop through **real
providers where they exist** and degrade honestly where they do not:

    OBJECTIVE → ORCHESTRATOR → PLANNER → REAL LLM (query proposal,
    optional) → RESEARCH/WEB → EVIDENCE (provenance-classified) →
    EVALUATOR → MEMORY (persisted + retrieval test) → COMPLETION/REPLAN

No fabrication rule: the exit code is 0 only when the run genuinely
completed on real (or clearly-labelled local) evidence; LLM- and web-
provider health are probed and reported as REAL / LOCAL / MOCK /
UNAVAILABLE — never silently substituted.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agency.config import SkynetSettings
from agency.orchestrator.health import check_all
from agency.orchestrator.models import OrchestratorBudgets
from agency.orchestrator.store import RunStore

DEFAULT_OBJECTIVE = (
    "Research a technical topic using available web sources, compare the "
    "evidence, identify uncertainties, and produce a concise evidence-backed report."
)

#: The ACT stage runs the real core loop; this explicit plan performs one
#: genuine research task through the P4 stack (the operator's steering seam
#: — the orchestrator never invents action choices itself).
REAL_RESEARCH_STEPS = [
    {
        "type": "web_research",
        "params": {"goal": "long-term memory architectures for autonomous AI agents"},
    },
    {
        "type": "memory_store",
        "params": {
            "content": "Autonomous research run completed a web_research task over "
            "the configured search provider; see evidence provenance for sources.",
            "type": "episodic",
            "importance": 0.65,
            "tags": ["autonomous", "research", "real-demo"],
            "summary": "Real-capability demo: one live web research task executed",
            "source": "action:web_research",
        },
    },
]

STEPS = ["INITIALIZE", "OBSERVE", "PLAN", "RESEARCH", "COMMUNICATE",
         "ACT", "EVALUATE", "LEARN", "COMPLETE"]


def _print_step(offset: int, name: str, detail: str) -> None:
    print(f"{offset:>2}. {name:12} {detail}")


async def run_real_demo(
    *,
    objective: str | None = None,
    queries: list[str] | None = None,
    questions: list[str] | None = None,
    max_iterations: int = 3,
    as_json: bool = False,
) -> int:
    """Run one bounded objective through real providers, print the stages."""
    from agency.bootstrap import build_orchestrator

    objective = objective or DEFAULT_OBJECTIVE
    started = datetime.now(UTC)

    # -- preflight: classify every provider seam (one real request each) ----
    settings = SkynetSettings(
        storage_backend="memory",
        perception_adapter="none",
        memory_backend="sqlite",
        memory_sqlite_path=str(Path(tempfile.mkdtemp(prefix="skynet-real-")) / "memory.db"),
    )
    settings = settings.model_copy(
        update={
            "enable_web_tools": True,
            "enable_external_comms": True,
            # Real execution needs the web/memory actions in the loop's
            # allow-list (registration gate + execution gate are separate).
            "enabled_actions": "echo,gods_eye_latest_events,web_research,"
            "web_search,web_fetch,web_extract,memory_store,memory_search,memory_extract",
        }
    )
    print("SKYNET real-capability demonstration")
    print(f"objective  : {objective}")
    health = await check_all(settings)
    for status in health:
        print(f"  provider : {status.summary_line()}")

    if queries is None:
        queries = ["long-term memory architectures for autonomous AI agents"]
    configuration: dict[str, Any] = {
        "research_queries": queries,
        "core_goal_params": {"steps": REAL_RESEARCH_STEPS},
    }
    if questions:
        configuration["ai_questions"] = questions

    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, **payload: Any) -> None:
        events.append((event, payload))

    runs_path = Path(tempfile.mkdtemp(prefix="skynet-real-runs-")) / "runs.jsonl"
    orchestrator = build_orchestrator(settings, emit=emit)
    orchestrator._store = RunStore(runs_path)

    budgets = OrchestratorBudgets(
        max_iterations=max_iterations,
        max_runtime_seconds=settings.autonomous_max_runtime_seconds,
        max_web_requests=settings.autonomous_max_web_requests,
        max_ai_requests=settings.autonomous_max_ai_requests,
        max_experiments=settings.autonomous_max_experiments,
    )
    run = await orchestrator.start(objective, budgets=budgets, configuration=configuration)
    run = await orchestrator.execute(run.id)

    # -- memory retrieval test (experience must be retrievable) --------------
    retrieval_test: dict[str, Any] = {"performed": False}
    if orchestrator._memory is not None and run.status.value == "completed":
        try:
            pairs = await orchestrator._memory.search(
                query=objective[:120], limit=3
            )
            records = (
                [p[0] for p in pairs]
                if pairs and isinstance(pairs[0], tuple)
                else list(pairs)
            )
            retrieval_test = {
                "performed": True,
                "hits": len(records),
                "top_source": records[0].source if records else "",
            }
        except Exception as exc:  # pragma: no cover - honesty over silence
            retrieval_test = {"performed": False, "error": str(exc)}

    elapsed = (datetime.now(UTC) - started).total_seconds()

    if as_json:
        import json

        print(
            json.dumps(
                {
                    "run_id": run.id,
                    "objective": run.objective,
                    "status": run.status.value,
                    "stages": [d.action for d in run.decisions],
                    "evidence": len(run.evidence),
                    "evidence_classes": sorted(
                        {
                            str(e.provenance.get("evidence_class", "UNLABELLED"))
                            for e in run.evidence
                        }
                    ),
                    "usage": run.usage,
                    "memory_retrieval_test": retrieval_test,
                    "providers": {h.component: h.label for h in health},
                    "security_report": run.security_report,
                    "elapsed_seconds": round(elapsed, 2),
                    "result": run.result,
                    "errors": run.errors,
                },
                indent=2,
            )
        )
        return 0 if run.status.value == "completed" else 1

    # Stage walk-through from the actual trace stream (not a script).
    seen: list[str] = []
    for event, payload in events:
        stage = payload.get("stage")
        if event == "AUTONOMOUS_STAGE_STARTED" and stage not in seen:
            seen.append(stage)
    print()
    for i, stage in enumerate(seen, start=1):
        detail = ""
        if stage == "plan":
            detail = f"queries={run.plan.research_queries}"
        elif stage == "research":
            detail = f"web_requests used={int(run.usage.get('web_requests', 0.0))}"
        elif stage == "communicate":
            detail = f"ai_requests used={int(run.usage.get('ai_requests', 0.0))}"
        elif stage == "act":
            oks = [s.ok for s in run.core_summaries]
            detail = f"core cycles={len(run.core_run_ids)} (ok={oks})"
        elif stage == "evaluate":
            detail = "; ".join(f"{d.action}: {d.reason[:60]}" for d in run.decisions[-2:])
        elif stage == "learn":
            detail = f"summary memory stored (provenance=autonomous_run:{run.id[:12]})"
        _print_step(i, stage.upper(), detail)

    print()
    print(f"status     : {run.status.value}")
    print(f"evidence   : {len(run.evidence)} item(s)")
    by_class: dict[str, int] = {}
    for e in run.evidence:
        ec = str(e.provenance.get("evidence_class", "UNLABELLED"))
        by_class[ec] = by_class.get(ec, 0) + 1
    for ec, count in sorted(by_class.items()):
        print(f"             {ec}: {count}")
    print(f"memory     : retrieval test {'passed' if retrieval_test.get('hits') else 'no hits'} "
          f"({retrieval_test.get('hits', 0)} hit(s))")
    if run.security_report:
        print(f"security   : {run.security_report.get('status')} — "
              f"{run.security_report.get('total_flags')} flag(s) recorded as data")
    print(f"elapsed    : {elapsed:.1f}s")
    if run.errors:
        print("errors     :")
        for error in run.errors:
            print(f"  - {error}")
    return 0 if run.status.value == "completed" else 1


__all__ = ["DEFAULT_OBJECTIVE", "run_real_demo"]
