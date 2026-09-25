"""End-to-end autonomous-loop demonstration (offline, deterministic).

Proves one objective travels the integrated architecture with real code and
real measurements — no fabricated results, no paid APIs, no network:

    OBJECTIVE → PLAN → OBSERVE(memory) → RESEARCH(local corpus)
    → COMMUNICATE(local mock AI) → ACT(SkynetCore cycle) → EVALUATE
    → LEARN(memory) → COMPLETE

The run is executed through :class:`AutonomousOrchestrator` exactly as the
CLI does, with a temp run store (re-runnable) and in-memory long-term memory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

OBJECTIVE = "Investigate the demo corpus and produce a supported answer"

CONFIGURATION: dict[str, Any] = {
    # The core agent cycle runs the echo action deterministically (the
    # memory-storage core has no web actions registered; steering the core
    # is the operator's job via configuration, not the orchestrator's).
    "core_goal_params": {
        "steps": [{"type": "echo", "params": {"message": "synthesize evidence"}}]
    },
    "ai_questions": ["What is known about the demo corpus topic?"],
}


def _print_step(offset: int, name: str, detail: str) -> None:
    print(f"{offset}. {name:12} {detail}")


async def run_demo(*, as_json: bool = False) -> int:
    """Execute the full autonomous loop offline and print what happened."""
    from agency.bootstrap import build_core
    from agency.config import SkynetSettings
    from agency.orchestrator.adapters import (
        build_comms_adapter,
        build_research_adapter,
    )
    from agency.orchestrator.models import OrchestratorBudgets
    from agency.orchestrator.orchestrator import AutonomousOrchestrator
    from agency.orchestrator.store import RunStore

    with tempfile.TemporaryDirectory(prefix="skynet-autonomous-demo-") as tmp:
        settings = SkynetSettings(storage_backend="memory", perception_adapter="none")
        run_store_path = Path(tmp) / "runs.jsonl"

        core = build_core(settings)
        memory = getattr(core, "memory_manager", None)
        orchestrator = AutonomousOrchestrator(
            core=core,
            store=RunStore(run_store_path),
            research=build_research_adapter(settings),
            comms=build_comms_adapter(settings),
            memory=memory,
            detector=None,  # improvement demo lives in `improve demo`
            pipeline=None,
        )

        run = await orchestrator.start(
            OBJECTIVE,
            budgets=OrchestratorBudgets(max_iterations=5),
            configuration=dict(CONFIGURATION),
        )

        # Cross-process pause check: a second store instance (as an operator
        # in another terminal would use) pauses the live run; we then resume.
        other = RunStore(run_store_path)
        await other.pause(run.id)
        paused = await RunStore(run_store_path).get(run.id)
        assert paused is not None and paused.status.value == "paused"
        await orchestrator.resume(run.id)

        run = await orchestrator.execute(run.id)

        if as_json:
            print(
                run.model_dump_json(
                    indent=2,
                    include={
                        "id",
                        "objective",
                        "status",
                        "current_stage",
                        "iteration",
                        "evidence",
                        "decisions",
                        "usage",
                        "result",
                        "errors",
                    },
                )
            )
            return 0 if run.result.get("completed") else 1

        print("SKYNET autonomous loop demonstration (offline, deterministic)")
        print(f"objective  : {run.objective}")
        print(f"budgets    : max_iterations={run.budgets.max_iterations}")
        stages = ["initialize", "observe", "plan", "research", "communicate", "act"]
        for i, stage in enumerate(stages, start=1):
            detail = {
                "initialize": f"run {run.id[:12]} created; state persisted to JSONL",
                "observe": (
                    f"memory recall first (memory={'on' if memory else 'off'})"
                ),
                "plan": (
                    f"queries={len(run.plan.research_queries)} "
                    f"ai_questions={len(run.plan.ai_questions)}"
                ),
                "research": (
                    f"web_requests used={int(run.usage.get('web_requests', 0.0))}"
                ),
                "communicate": (
                    f"ai_requests used={int(run.usage.get('ai_requests', 0.0))}"
                ),
                "act": (
                    f"core cycles={len(run.core_run_ids)} "
                    f"(all passed={all(s.ok for s in run.core_summaries)})"
                ),
            }[stage]
            _print_step(i, stage.upper(), detail)
        decision_lines = [f"{d.action}: {d.reason[:70]}" for d in run.decisions]
        _print_step(7, "EVALUATE", "; ".join(decision_lines) or "n/a")
        _print_step(
            8,
            "LEARN",
            f"summary memory stored (provenance=autonomous_run:{run.id[:12]})"
            if memory
            else "memory disabled in this demo config",
        )
        _print_step(
            9,
            "COMPLETE",
            f"status={run.status.value}; {run.result.get('reason', '')}",
        )
        print(f"evidence   : {len(run.evidence)} item(s) — by source type:")
        by_type: dict[str, int] = {}
        for e in run.evidence:
            by_type[e.source_type] = by_type.get(e.source_type, 0) + 1
        for source_type, count in sorted(by_type.items()):
            print(f"             {source_type}: {count}")
        if run.errors:
            print("errors     :")
            for error in run.errors:
                print(f"  - {error}")
        return 0 if run.result.get("completed") else 1


__all__ = ["run_demo"]
