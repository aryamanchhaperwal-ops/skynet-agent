"""Command-line entry point for the Skynet core.

Usage (from the repository root):

    .venv/Scripts/python.exe -m agency.cli run --goal "Report the strongest recent earthquake"

    # deterministic plan without the database (memory backend):
    .venv/Scripts/python.exe -m agency.cli run --goal "Demo" --storage memory \
        --steps '[{"type": "echo", "params": {"text": "hello"}}]'

After ``pip install -e .`` the shorter form is available: ``skynet run ...``.

Exit codes: 0 = run completed, 1 = run failed. ``--json`` prints only the
structured run summary (machine-readable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from agency.bootstrap import build_core
from agency.config import SkynetSettings
from agency.goals import GoalSpec
from agency.loop import RunSummary
from agency.storage import ensure_core_schema


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skynet", description="Run the Skynet core.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one goal through the Skynet core.")
    run.add_argument("--goal", required=True, help="Goal title (what Skynet should achieve).")
    run.add_argument("--description", default="", help="Optional longer goal description.")
    run.add_argument(
        "--storage",
        choices=("database", "memory"),
        default=None,
        help="Override SKYNET_STORAGE_BACKEND for this run.",
    )
    run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override SKYNET_MAX_STEPS_PER_RUN for this run.",
    )
    run.add_argument(
    "--steps",
    default=None,
    help="Explicit plan as a JSON array of {\"type\", \"params\"} steps "
    "(default: deterministic planner's built-in strategy).",
)
    run.add_argument(
        "--memory-backend",
        choices=("memory", "sqlite"),
        default=None,
        help="Override SKYNET_MEMORY_BACKEND for this run (sqlite persists across runs).",
    )
    run.add_argument("--json", action="store_true", help="Print only the JSON run summary.")

    research = subparsers.add_parser(
        "research",
        help="Run one web research task through the Skynet core "
        "(enables web tools for this run).",
    )
    research.add_argument("goal", help="Research goal, e.g. a question to investigate.")
    research.add_argument(
        "--query",
        action="append",
        default=None,
        help="Explicit search query (repeatable; overrides goal-derived queries).",
    )
    research.add_argument(
        "--storage",
        choices=("database", "memory"),
        default=None,
        help="Override SKYNET_STORAGE_BACKEND for this run.",
    )
    research.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Override SKYNET_WEB_MAX_PAGES_PER_RESEARCH for this run.",
    )
    research.add_argument("--json", action="store_true", help="Print only the JSON run summary.")

    ai = subparsers.add_parser(
        "ai-ask",
        help="Ask an external AI participant one research question "
        "through the Skynet core (enables comms for this run).",
    )
    ai.add_argument("participant", help="Participant id, e.g. mock:mock-agent-1.")
    ai.add_argument("question", help="Question to ask the external AI.")
    ai.add_argument(
        "--follow-up",
        action="append",
        default=None,
        help="Follow-up question (repeatable; each adds one conversation turn).",
    )
    ai.add_argument(
        "--provider",
        default="mock",
        help="Comma-separated providers to activate (default: mock; offline).",
    )
    ai.add_argument(
        "--storage",
        choices=("database", "memory"),
        default=None,
        help="Override SKYNET_STORAGE_BACKEND for this run.",
    )
    ai.add_argument("--json", action="store_true", help="Print only the JSON run summary.")

    ai_participants = subparsers.add_parser(
        "ai-participants",
        help="List external AI participants and their declared capabilities "
        "(capability-based discovery, offline with the mock provider).",
    )
    ai_participants.add_argument(
        "--provider",
        default="mock",
        help="Comma-separated providers to activate (default: mock; offline).",
    )
    ai_participants.add_argument(
        "--capability",
        default=None,
        help="Only list participants declaring this capability (e.g. research).",
    )
    ai_participants.add_argument("--json", action="store_true", help="Print only JSON.")

    ai_demo = subparsers.add_parser(
        "ai-demo",
        help="Compare the same question across available AI participants "
        "(mock provider, fully offline).",
    )
    ai_demo.add_argument("question", help="Question to compare answers on.")
    ai_demo.add_argument(
        "--providers",
        default="mock,mock,mock",
        help="Comma-separated providers; repeated names become independent "
        "participants (default: three mocks).",
    )
    ai_demo.add_argument(
        "--storage",
        choices=("database", "memory"),
        default=None,
        help="Override SKYNET_STORAGE_BACKEND for this run.",
    )
    ai_demo.add_argument("--json", action="store_true", help="Print only the JSON run summary.")

    remember = subparsers.add_parser(
        "remember",
        help="Store one memory directly (with provenance and importance).",
    )
    remember.add_argument("content", help="Memory content (plain text).")
    remember.add_argument(
        "--type",
        dest="memory_type",
        choices=("episodic", "semantic", "procedural"),
        default="semantic",
        help="Memory type (default: semantic).",
    )
    remember.add_argument("--source", default="human", help="Provenance source label.")
    remember.add_argument(
        "--importance", type=float, default=0.8, help="Importance 0.0-1.0 (default 0.8)."
    )
    remember.add_argument("--tag", action="append", default=None, help="Tag (repeatable).")
    remember.add_argument(
        "--storage",
        choices=("memory", "sqlite"),
        default=None,
        help="Override SKYNET_MEMORY_BACKEND for this command.",
    )

    recall = subparsers.add_parser(
        "recall",
        help="Search long-term memory (what has Skynet learned?).",
    )
    recall.add_argument("query", help="Keyword query.")
    recall.add_argument(
        "--type",
        dest="memory_type",
        choices=("episodic", "semantic", "procedural"),
        default=None,
        help="Filter by memory type.",
    )
    recall.add_argument("--limit", type=int, default=None, help="Max results.")
    recall.add_argument(
        "--storage",
        choices=("memory", "sqlite"),
        default=None,
        help="Override SKYNET_MEMORY_BACKEND for this command.",
    )
    recall.add_argument("--json", action="store_true", help="Print only JSON.")

    stats_parser = subparsers.add_parser(
        "memory-stats",
        help="Show long-term memory census (counts by type and origin).",
    )
    stats_parser.add_argument(
        "--storage",
        choices=("memory", "sqlite"),
        default=None,
        help="Override SKYNET_MEMORY_BACKEND for this command.",
    )
    stats_parser.add_argument("--json", action="store_true", help="Print only JSON.")

    # -- Experiments (agency.experiments, Phase P6) ------------------------------
    experiment = subparsers.add_parser(
        "experiment",
        help="Run, list or inspect controlled experiments (offline demo included).",
    )
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)

    exp_run = experiment_sub.add_parser(
        "run",
        help="Run one experiment: baseline vs candidate strategy with criteria.",
    )
    exp_run.add_argument("--name", required=True, help="Experiment name.")
    exp_run.add_argument(
        "--hypothesis", required=True, help="Testable claim (one sentence)."
    )
    exp_run.add_argument(
        "--baseline", required=True, help="Baseline strategy ref, e.g. single_query:v1."
    )
    exp_run.add_argument(
        "--candidate", required=True, help="Candidate strategy ref, e.g. multi_query:v1."
    )
    exp_run.add_argument(
        "--criterion",
        action="append",
        required=True,
        help="Frozen acceptance criterion as JSON: '{\"kind\": \"improves\", "
        "\"metric\": \"sources_found\", \"threshold\": 0.1}' (repeatable).",
    )
    exp_run.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric definition 'name[:direction]' (repeatable; direction maximize|minimize).",
    )
    exp_run.add_argument("--trials", type=int, default=3, help="Trials per arm (default 3).")
    exp_run.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    exp_run.add_argument(
        "--memory-backend",
        choices=("memory", "sqlite"),
        default=None,
        help="Where experiment memories are recorded (sqlite persists across runs).",
    )
    exp_run.add_argument(
        "--param",
        action="append",
        default=None,
        help="Procedure parameter key=value (repeatable), e.g. --param goal=agent memory.",
    )
    exp_run.add_argument("--json", action="store_true", help="Print only JSON.")

    exp_list = experiment_sub.add_parser(
        "list", help="List previously run experiments (have I tried this before?)."
    )
    exp_list.add_argument(
        "--status", default=None, help="Filter by status (completed/failed/inconclusive/...)."
    )
    exp_list.add_argument("--json", action="store_true", help="Print only JSON.")

    exp_inspect = experiment_sub.add_parser(
        "inspect", help="Show one experiment's full record (criteria, trials, verdict)."
    )
    exp_inspect.add_argument("experiment_id", help="Experiment id (from experiment list).")
    exp_inspect.add_argument("--json", action="store_true", help="Print only JSON.")

    experiment_sub.add_parser(
        "demo",
        help="Run the built-in offline demo: single-query vs multi-query "
        "research strategies over an injected corpus (deterministic).",
    )
    exp_demo = experiment_sub.choices["demo"]  # type: ignore[index]
    exp_demo.add_argument(
        "--goal", default="approaches to long-term memory in autonomous AI agents",
        help="Research goal the strategies search for.",
    )
    exp_demo.add_argument("--trials", type=int, default=5, help="Trials per arm (default 5).")
    exp_demo.add_argument(
        "--memory-backend",
        choices=("memory", "sqlite"),
        default=None,
        help="Where experiment memories are recorded (sqlite persists across runs).",
    )
    exp_demo.add_argument("--json", action="store_true", help="Print only JSON.")

    # -- Self-improvement (agency.improve, Phase P7) --------------------------------
    improve = subparsers.add_parser(
        "improve",
        help="Detect weaknesses, propose improvements, test, approve, "
        "apply, roll back and monitor (offline demo included).",
    )
    improve_sub = improve.add_subparsers(dest="improve_command", required=True)

    imp_detect = improve_sub.add_parser(
        "detect",
        help="Detect weaknesses from recorded experiences and experiment history.",
    )
    imp_detect.add_argument(
        "--memory-backend",
        choices=("memory", "sqlite"),
        default=None,
        help="Memory backend for weakness persistence (sqlite persists).",
    )
    imp_detect.add_argument("--json", action="store_true", help="Print only JSON.")

    imp_propose = improve_sub.add_parser(
        "propose", help="Build hypothesis + proposal + candidate for one weakness."
    )
    imp_propose.add_argument("weakness_id", help="Weakness id (from improve detect).")
    imp_propose.add_argument(
        "--origin",
        choices=("internal", "external"),
        default="internal",
        help="Internal detection or an external (web/AI) suggestion (same gates).",
    )
    imp_propose.add_argument(
        "--param",
        action="append",
        default=None,
        help="Candidate config key=value (repeatable), e.g. --param max_results=8.",
    )
    imp_propose.add_argument("--json", action="store_true", help="Print only JSON.")

    improve_sub.add_parser(
        "history",
        help="List recorded proposals (have I attempted this before?).",
    ).add_argument("--json", action="store_true", help="Print only JSON.")

    improve_sub.add_parser(
        "demo",
        help="Run the built-in end-to-end offline demonstration: detect → "
        "propose → test → evaluate → approve → apply → monitor → rollback.",
    )
    imp_demo = improve_sub.choices["demo"]  # type: ignore[index]
    imp_demo.add_argument(
        "--memory-backend",
        choices=("memory", "sqlite"),
        default=None,
        help="Where improvement memories are recorded (sqlite persists).",
    )
    imp_demo.add_argument("--json", action="store_true", help="Print only JSON.")

    # -- Autonomous orchestrator (agency.orchestrator, Phase P8) -------------------
    autonomous = subparsers.add_parser(
        "autonomous",
        help="Start, pause, resume, cancel and inspect resumable autonomous runs "
        "that coordinate research, AI communication, the agent core, memory "
        "and self-improvement (offline demo included).",
    )
    autonomous_sub = autonomous.add_subparsers(
        dest="autonomous_command", required=True
    )

    aut_start = autonomous_sub.add_parser(
        "start", help="Create and drive a new autonomous run toward an objective."
    )
    aut_start.add_argument(
        "--objective", required=True, help="What the run should achieve."
    )
    aut_start.add_argument(
        "--max-iterations", type=int, default=None,
        help="Iteration budget (default SKYNET_AUTONOMOUS_MAX_ITERATIONS).",
    )
    aut_start.add_argument(
        "--question", action="append", default=None,
        help="Question to put to available external AI participants (repeatable).",
    )
    aut_start.add_argument(
        "--steps", default=None,
        help="Core agent-cycle plan as JSON [{\"type\",\"params\"}] (executed "
        "each iteration; default: the core planner's own strategy).",
    )
    aut_start.add_argument("--json", action="store_true", help="Print only JSON.")

    aut_status = autonomous_sub.add_parser(
        "status", help="List autonomous runs and their current state."
    )

    # -- Real capabilities (P9): provider health + bounded real objective ----- 
    provider = subparsers.add_parser(
        "provider",
        help="Provider health checks for real LLM/web/comms capabilities.",
    )
    provider_sub = provider.add_subparsers(dest="provider_command", required=True)
    prov_status = provider_sub.add_parser(
        "status",
        help="Classify every provider seam (REAL/LOCAL/MOCK/UNAVAILABLE) without "
        "sending requests.",
    )
    prov_status.add_argument("--json", action="store_true", help="JSON output only.")
    prov_test = provider_sub.add_parser(
        "test",
        help="One minimal REAL request per configured provider (LLM, web).",
    )
    prov_test.add_argument(
        "--component",
        choices=("all", "llm", "web"),
        default="all",
        help="Which seam to probe (default: all configured).",
    )
    prov_test.add_argument("--json", action="store_true", help="JSON output only.")

    run_cmd = subparsers.add_parser(
        "real",
        help="Bounded real-capability runs (start/status/inspect). "
        "Named `real` because `run` is the core's single-goal command.",
    )
    run_sub = run_cmd.add_subparsers(dest="run_command", required=True)
    run_start = run_sub.add_parser(
        "start",
        help="Start one bounded autonomous run through real providers "
        "(offline adapters only when explicitly requested).",
    )
    run_start.add_argument(
        "--objective",
        default=(
            "Research a technical topic using available web sources, compare the "
            "evidence, identify uncertainties, and produce a concise evidence-backed report."
        ),
        help="The bounded research objective (configurable; nothing hard-coded).",
    )
    run_start.add_argument(
        "--query",
        action="append",
        default=None,
        help="Explicit research query (repeatable; skips LLM query proposal).",
    )
    run_start.add_argument(
        "--question",
        action="append",
        default=None,
        help="Question for external AI participants (repeatable).",
    )
    run_start.add_argument(
        "--max-iterations", type=int, default=None, help="Iteration budget override."
    )
    run_start.add_argument(
        "--offline",
        action="store_true",
        help="Force local deterministic adapters (never contacts real providers).",
    )
    run_start.add_argument("--json", action="store_true", help="JSON output only.")
    run_status = run_sub.add_parser("status", help="List bounded real runs.")
    run_status.add_argument("--json", action="store_true", help="Print only JSON.")
    run_status_arg = run_sub.add_parser("inspect", help="Inspect one run's full record.")
    run_status_arg.add_argument("run_id", help="Run id (from run start/status).")
    run_status_arg.add_argument("--json", action="store_true", help="Print only JSON.")
    real_demo = run_sub.add_parser(
        "demo",
        help="One bounded REAL-capability run: provider preflight, full stage "
        "walk-through, memory retrieval test (uses live web where configured).",
    )
    real_demo.add_argument(
        "--objective", default=None, help="Override the demonstration objective."
    )
    real_demo.add_argument(
        "--query", action="append", default=None, help="Explicit research query."
    )
    real_demo.add_argument("--json", action="store_true", help="JSON output only.")
    aut_status.add_argument("--json", action="store_true", help="Print only JSON.")

    for name, helptext in (
        ("pause", "Pause a running autonomous run (safe to resume later)."),
        ("resume", "Resume a paused autonomous run."),
        ("cancel", "Cancel an autonomous run (state is preserved)."),
        ("inspect", "Show one run's stage, usage, decisions and errors."),
    ):
        cmd = autonomous_sub.add_parser(name, help=helptext)
        cmd.add_argument("run_id", help="Autonomous run id (from autonomous start/status).")
        cmd.add_argument("--json", action="store_true", help="Print only JSON.")

    autonomous_sub.add_parser(
        "demo",
        help="Run the built-in deterministic offline demonstration of the full "
        "autonomous loop (research → communicate → act → evaluate → complete).",
    ).add_argument("--json", action="store_true", help="Print only JSON.")
    return parser


async def _run(args: argparse.Namespace) -> int:
    if getattr(args, "command", None) == "research":
        return await _run_research(args)
    if getattr(args, "command", None) == "ai-participants":
        return await _run_ai_participants(args)
    if getattr(args, "command", None) in {"ai-ask", "ai-demo"}:
        return await _run_ai(args)
    if getattr(args, "command", None) == "remember":
        return await _run_remember(args)
    if getattr(args, "command", None) == "recall":
        return await _run_recall(args)
    if getattr(args, "command", None) == "memory-stats":
        return await _run_memory_stats(args)
    if getattr(args, "command", None) == "experiment":
        return await _run_experiment(args)
    if getattr(args, "command", None) == "improve":
        return await _run_improve(args)
    if getattr(args, "command", None) == "autonomous":
        return await _run_autonomous(args)
    if getattr(args, "command", None) == "provider":
        return await _run_provider(args)
    if getattr(args, "command", None) == "real":
        return await _run_real(args)
    return await _run_goal(args)


def settings_default_actions() -> str:
    """Configured action allow-list (helper for run-command overrides)."""
    return SkynetSettings().enabled_actions


def _memory_settings(args: argparse.Namespace) -> SkynetSettings:
    """Memory settings for direct memory commands (storage override only)."""
    updates: dict[str, Any] = {}
    if getattr(args, "storage", None):
        updates["memory_backend"] = args.storage
    return SkynetSettings(**updates)


async def _run_remember(args: argparse.Namespace) -> int:
    from agency.bootstrap import build_memory_stack

    manager = build_memory_stack(_memory_settings(args)).manager
    record = await manager.store(
        content=args.content,
        memory_type=args.memory_type,
        summary=args.content[:120],
        source=args.source,
        origin="human",
        importance=args.importance,
        tags=args.tag or [],
        force=True,  # direct operator insertion bypasses the threshold gate
    )
    if record is None:
        print("error: memory rejected", file=sys.stderr)
        return 2
    print(f"memory stored: {record.id}")
    print(f"  type       : {record.type.value}")
    print(f"  importance : {record.importance:.2f}")
    print(f"  source     : {record.source}")
    if record.tags:
        print(f"  tags       : {', '.join(record.tags)}")
    return 0


async def _run_recall(args: argparse.Namespace) -> int:
    from agency.bootstrap import build_memory_stack

    manager = build_memory_stack(_memory_settings(args)).manager
    results = await manager.search(
        query=args.query,
        memory_type=args.memory_type,
        limit=args.limit,
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": record.id,
                        "type": record.type.value,
                        "score": score,
                        "summary": record.summary,
                        "source": record.source,
                        "provenance": record.provenance,
                        "created_at": record.created_at.isoformat(),
                    }
                    for record, score in results
                ],
                indent=2,
            )
        )
        return 0
    if not results:
        print("no memories found")
        return 1
    print(f"recalled {len(results)} memory(ies) for: {args.query}")
    for record, score in results:
        print(f"  - [{record.type.value}] ({score:.2f}) {record.summary or record.content[:100]}")
        print(f"      source: {record.source} · stored {record.created_at:%Y-%m-%d %H:%M} UTC")
    return 0


async def _run_memory_stats(args: argparse.Namespace) -> int:
    from agency.bootstrap import build_memory_stack

    manager = build_memory_stack(_memory_settings(args)).manager
    stats = await manager.stats()
    if args.json:
        print(stats.model_dump_json(indent=2))
        return 0
    print(f"long-term memory: {stats.total} record(s)")
    for memory_type, count in sorted(stats.by_type.items()):
        print(f"  {memory_type:12}: {count}")
    return 0


async def _run_ai_participants(args: argparse.Namespace) -> int:
    """Discovery-only: list participants and declared capabilities."""
    from agency.comms.providers import build_providers_from_list

    try:
        providers = build_providers_from_list(args.provider.split(","))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    participants = [p.identify() for p in providers]
    if args.capability:
        wanted = args.capability.strip().lower()
        participants = [
            p for p in participants if wanted in {c.value for c in p.capabilities}
        ]
    if args.json:
        print(json.dumps([p.model_dump(mode="json") for p in participants], indent=2))
        return 0
    if not participants:
        print("no AI participants available")
        return 1
    print(f"AI participants: {len(participants)}")
    for participant in participants:
        caps = ", ".join(sorted(c.value for c in participant.capabilities)) or "(none declared)"
        print(
            f"  - {participant.participant_id}  [{participant.provider}/{participant.model}]"
        )
        print(f"      capabilities : {caps}")
        print(f"      status       : {participant.status.value}")
    return 0


async def _run_goal(args: argparse.Namespace) -> int:
    updates: dict[str, Any] = {}
    if args.storage is not None:
        updates["storage_backend"] = args.storage
        if args.storage == "memory":
            # The memory backend cannot serve the God's Eye adapter (it reads
            # the events table); fall back to the null adapter for this run.
            updates["perception_adapter"] = "none"
    if args.max_steps is not None:
        updates["max_steps_per_run"] = args.max_steps
    if getattr(args, "memory_backend", None):
        updates["memory_backend"] = args.memory_backend
        # Opting into persistent memory for this run is explicit operator
        # intent: allow the memory actions through the second gate too.
        memory_actions = "memory_store,memory_search,memory_extract"
        updates["enabled_actions"] = (
            f"{settings_default_actions()},{memory_actions}"
        )
    settings = SkynetSettings(**updates)

    if args.steps:
        try:
            steps = json.loads(args.steps)
        except json.JSONDecodeError as exc:
            print(f"error: --steps is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(steps, list):
            print("error: --steps must be a JSON array", file=sys.stderr)
            return 2
        params: dict[str, Any] = {"steps": steps}
    else:
        params = {}

    core = build_core(settings)
    if settings.storage_backend == "database":
        await ensure_core_schema()

    summary = await core.run_goal(
        GoalSpec(title=args.goal, description=args.description, params=params)
    )
    _print_summary(summary, as_json=args.json)
    return 0 if summary.ok else 1


async def _run_research(args: argparse.Namespace) -> int:
    """One research goal through the core, with web tools enabled for this run."""
    updates: dict[str, Any] = {
        # Web tools on for this run, second gate opened with them.
        "enable_web_tools": True,
        "enabled_actions": "echo,web_search,web_fetch,web_extract,web_research",
        # The default 10-step / 120 s budgets are sized for the deterministic
        # foundation; research needs a little more headroom.
        "max_steps_per_run": 4,
        "max_run_seconds": 300,
        "perception_adapter": "none",
    }
    if args.storage is not None:
        updates["storage_backend"] = args.storage
        if args.storage == "memory":
            updates["perception_adapter"] = "none"
    if args.max_pages is not None:
        updates["web_max_pages_per_research"] = args.max_pages
    settings = SkynetSettings(**updates)

    # Explicit single-step plan: the deterministic planner honors goal
    # params['steps'] verbatim, so no planner changes are needed for research.
    step_params: dict[str, Any] = {"goal": args.goal}
    if args.query:
        step_params["queries"] = list(args.query)
    params: dict[str, Any] = {"steps": [{"type": "web_research", "params": step_params}]}

    core = build_core(settings)
    if settings.storage_backend == "database":
        await ensure_core_schema()

    summary = await core.run_goal(
        GoalSpec(
            title=args.goal,
            description=f"Web research task: {args.goal}",
            params=params,
        )
    )
    _print_research_summary(summary, as_json=args.json)
    return 0 if summary.ok else 1


async def _run_ai(args: argparse.Namespace) -> int:
    """AI-to-AI runs: one question (ai-ask) or a comparison (ai-demo)."""
    compare_mode = args.command == "ai-demo"
    raw_names = args.providers if compare_mode else args.provider
    provider_names = [n.strip() for n in raw_names.split(",") if n.strip()]
    updates: dict[str, Any] = {
        "enable_external_comms": True,
        "enabled_actions": "echo,ai_list_participants,ai_ask,ai_compare",
        # Repeated names are intentional: each becomes an independent
        # participant (mock#2, mock#3…) so comparisons have several parties.
        "ai_providers": ",".join(provider_names),
        "ai_request_interval_seconds": 0.0,  # offline mocks; keep demos snappy
        "max_steps_per_run": 3 if compare_mode else 2,
        "max_run_seconds": 120,
        "perception_adapter": "none",
    }
    if args.storage is not None:
        updates["storage_backend"] = args.storage
        if args.storage == "memory":
            updates["perception_adapter"] = "none"
    settings = SkynetSettings(**updates)

    if compare_mode:
        params: dict[str, Any] = {
            "steps": [{"type": "ai_compare", "params": {"prompt": args.question}}]
        }
        goal_title = f"Compare AI answers: {args.question}"
    else:
        step_params: dict[str, Any] = {
            "participant_id": args.participant,
            "question": args.question,
        }
        if args.follow_up:
            step_params["follow_ups"] = list(args.follow_up)
        params = {"steps": [{"type": "ai_ask", "params": step_params}]}
        goal_title = f"Ask AI ({args.participant}): {args.question}"

    core = build_core(settings)
    if settings.storage_backend == "database":
        await ensure_core_schema()
    summary = await core.run_goal(GoalSpec(title=goal_title, description=goal_title, params=params))
    _print_ai_summary(summary, compare_mode=compare_mode, as_json=args.json)
    return 0 if summary.ok else 1


def _ai_output_from(summary: RunSummary) -> dict[str, Any] | None:
    """Locate the ai_* action's output in the run's action records."""
    for record in summary.final_state.get("actions", []):
        if not isinstance(record, dict):
            continue
        spec = record.get("spec") or {}
        result = record.get("result") or {}
        if str(spec.get("type", "")).startswith("ai_") and result.get("success"):
            return result.get("output") or {}
    return None


def _print_ai_summary(summary: RunSummary, *, compare_mode: bool, as_json: bool) -> None:
    """Human-readable AI interaction output: participants → answers → provenance."""
    if as_json:
        print(json.dumps(summary.model_dump(mode="json"), indent=2))
        return

    mode = "comparison" if compare_mode else "conversation"
    print(f"SKYNET ai-{mode} {summary.run_id}")
    print(f"  goal        : {summary.goal_title}")
    print(f"  status      : {summary.status.upper()}")
    for action in summary.actions:
        marker = "ok" if action["success"] else f"FAILED: {action['error']}"
        print(f"    - {action['type']}: {marker} ({action['duration_ms']} ms)")
    output = _ai_output_from(summary) or {}
    if compare_mode and output.get("comparison"):
        comparison = output["comparison"]
        print(f"  participants: {len(comparison.get('participant_ids') or [])} asked, "
              f"{len(comparison.get('responded') or [])} responded")
        for pid, text in (comparison.get("responses") or {}).items():
            print(f"    - {pid}: {text[:140]}{'…' if len(text) > 140 else ''}")
        for pid, error in (comparison.get("failures") or {}).items():
            print(f"    - {pid}: FAILED — {error}")
        agreement = comparison.get("agreement_terms") or []
        divergence = comparison.get("divergence_terms") or []
        print(f"  agreement   : {', '.join(agreement[:12]) or '(none recorded)'}")
        print(f"  divergence  : {', '.join(divergence[:12]) or '(none recorded)'}")
        print("  note        : agreement is recorded, not treated as truth")
    elif output.get("conversation"):
        conversation = output["conversation"]
        print(f"  participant : {conversation.get('participant_id')}")
        print(f"  turns       : {len(output.get('turns') or [])} "
              f"(status {conversation.get('status')})")
        for turn in output.get("turns") or []:
            print(f"    Q: {turn.get('question')}")
            answer = turn.get("answer") or ""
            print(f"    A: {answer[:160]}{'…' if len(answer) > 160 else ''}")
            if turn.get("analysis_directives"):
                print(f"    [!] {len(turn['analysis_directives'])} directive-like "
                      "fragment(s) flagged — treated as data, never executed")
        observations = output.get("skynet_observations") or []
        if observations:
            provenance = (observations[0].get("metadata") or {}).get("provenance") or {}
            print(f"  provenance  : conversation {provenance.get('conversation_id', '?')[:8]}… "
                  f"→ participant {provenance.get('participant_id')} → message "
                  f"{str(provenance.get('message_id', '?'))[:8]}…")
        print(f"  observations: {len(observations)} structured")
    if summary.evaluation is not None:
        print(
            f"  evaluation  : {'PASSED' if summary.evaluation.passed else 'FAILED'} "
            f"(score {summary.evaluation.score:.2f}) — {summary.evaluation.notes}"
        )
    print(f"  experiences : {summary.experiences} recorded")
    print(f"  events      : {summary.events} traced")
    if summary.error:
        print(f"  error       : {summary.error}")


def _research_package_from(summary: RunSummary) -> dict[str, Any] | None:
    """Locate the web_research action's package in the run's action records."""
    for record in summary.final_state.get("actions", []):
        if not isinstance(record, dict):
            continue
        spec = record.get("spec") or {}
        result = record.get("result") or {}
        if spec.get("type") != "web_research" or not result.get("success"):
            continue
        output = result.get("output") or {}
        package = output.get("package")
        if isinstance(package, dict):
            return package
    return None


def _print_research_summary(summary: RunSummary, *, as_json: bool) -> None:
    """Human-readable research output: goal → queries → sources → package."""
    if as_json:
        print(json.dumps(summary.model_dump(mode="json"), indent=2))
        return

    print(f"SKYNET research {summary.run_id}")
    print(f"  goal        : {summary.goal_title}")
    print(f"  status      : {summary.status.upper()}")
    for action in summary.actions:
        marker = "ok" if action["success"] else f"FAILED: {action['error']}"
        print(f"    - {action['type']}: {marker} ({action['duration_ms']} ms)")
    package = _research_package_from(summary)
    if isinstance(package, dict):
        print(f"  queries     : {', '.join(package.get('queries') or []) or '(goal text)'}")
        sources = package.get("sources") or []
        print(f"  sources     : {len(sources)} retrieved, "
              f"{len(package.get('failures') or [])} failed")
        for source in sources:
            status = source.get("extraction_status", "?")
            chars = len(source.get("text") or "")
            print(
                f"    - [{status}] {source.get('title') or source.get('url')} "
                f"({chars} chars) — {source.get('url')}"
            )
        for failure in package.get("failures") or []:
            print(f"    - [failed] {failure.get('url')} — {failure.get('error')}")
        print(f"  observations: {len(package.get('observations') or [])} structured")
        for warning in package.get("warnings") or []:
            print(f"  warning     : {warning}")
    if summary.evaluation is not None:
        print(
            f"  evaluation  : {'PASSED' if summary.evaluation.passed else 'FAILED'} "
            f"(score {summary.evaluation.score:.2f}) — {summary.evaluation.notes}"
        )
    print(f"  experiences : {summary.experiences} recorded")
    print(f"  events      : {summary.events} traced")
    if summary.error:
        print(f"  error       : {summary.error}")


def _print_summary(summary: RunSummary, *, as_json: bool) -> None:
    payload = summary.model_dump(mode="json")
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    print(f"SKYNET run {summary.run_id}")
    print(f"  goal        : {summary.goal_title} (goal {summary.goal_id})")
    print(f"  status      : {summary.status.upper()}")
    print(f"  planner     : {summary.planner} / perception: {summary.perception}")
    print(f"  steps       : {summary.steps_executed}/{summary.steps_planned} executed")
    for action in summary.actions:
        marker = "ok" if action["success"] else f"FAILED: {action['error']}"
        print(f"    - {action['type']}: {marker} ({action['duration_ms']} ms)")
    print(f"  observations: {summary.observations}")
    if summary.evaluation is not None:
        print(
            f"  evaluation  : {'PASSED' if summary.evaluation.passed else 'FAILED'} "
            f"(score {summary.evaluation.score:.2f}) — {summary.evaluation.notes}"
        )
    print(f"  experiences : {summary.experiences} recorded")
    print(f"  events      : {summary.events} traced")
    if summary.error:
        print(f"  error       : {summary.error}")
    result = summary.result
    if result.get("strongest_event"):
        strongest = result["strongest_event"]
        print(f"  strongest   : M{strongest.get('magnitude')} — {strongest.get('place')}")
    if result.get("summary"):
        print(f"  summary     : {result['summary']}")


# -- Experiment commands (Phase P6) -----------------------------------------------


def _lab_stack(args: argparse.Namespace):
    """Build the lab stack with the demo strategies registered."""
    from agency.bootstrap import build_lab_stack
    from agency.config import SkynetSettings
    from agency.experiments.demo import register_demo_strategies

    settings = SkynetSettings()
    stack = build_lab_stack(settings)
    register_demo_strategies(stack.strategies)
    return stack


def _experiment_memory(args: argparse.Namespace):
    """Memory manager for experiment recording (backend per operator flag)."""
    from agency.bootstrap import build_memory_stack
    from agency.config import SkynetSettings

    updates: dict[str, Any] = {}
    backend = getattr(args, "memory_backend", None)
    if backend:
        updates["memory_backend"] = backend
    return build_memory_stack(SkynetSettings(**updates)).manager


def _parse_criterion(raw: str) -> dict[str, Any]:
    import json as _json

    try:
        parsed = _json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: --criterion is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("error: --criterion must be a JSON object")
    return parsed


async def _run_experiment(args: argparse.Namespace) -> int:
    from agency.experiments.experiments_store import ExperimentRegistry
    from agency.experiments.memory_bridge import record_experiment
    from agency.experiments.models import (
        Experiment,
        Hypothesis,
        MetricDefinition,
        SuccessCriterion,
        TargetRef,
    )
    from agency.experiments.runner import ExperimentRunner

    command = args.experiment_command

    if command == "list":
        store = ExperimentRegistry()
        summaries = await store.list(status=args.status)
        if args.json:
            print(json.dumps([s.model_dump(mode="json") for s in summaries], indent=2))
            return 0
        if not summaries:
            print("no experiments recorded")
            return 1
        print(f"experiments: {len(summaries)}")
        for item in summaries:
            verdict = item.verdict.value if item.verdict else "—"
            print(
                f"  - {item.id[:12]}  {item.status.value:12}  {verdict:12} "
                f"{item.baseline} vs {item.candidate} ({item.trials} trials)"
            )
            print(f"      created {item.created_at:%Y-%m-%d %H:%M} UTC")
        return 0

    if command == "inspect":
        store = ExperimentRegistry()
        record = await store.get(args.experiment_id)
        if record is None:
            print(f"error: unknown experiment {args.experiment_id!r}", file=sys.stderr)
            return 2
        if args.json:
            print(record.model_dump_json(indent=2))
            return 0
        _print_experiment_detail(record)
        return 0

    if command == "demo":
        return await _run_experiment_demo(args)

    # -- run ------------------------------------------------------------------
    stack = _lab_stack(args)
    criteria: list[SuccessCriterion] = []
    for raw in args.criterion or []:
        parsed = _parse_criterion(raw)
        try:
            criteria.append(
                SuccessCriterion(
                    kind=parsed["kind"],
                    metric=parsed["metric"],
                    threshold=float(parsed.get("threshold", 0.0)),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            print(f"error: invalid criterion {raw!r}: {exc}", file=sys.stderr)
            return 2
    metrics = []
    for raw in args.metric or []:
        name, _, direction = raw.partition(":")
        metrics.append(MetricDefinition(name=name, direction=direction or "maximize"))
    procedure_params: dict[str, Any] = {}
    for raw in args.param or []:
        key, _, value = raw.partition("=")
        procedure_params[key.strip()] = value

    experiment = Experiment(
        name=args.name,
        hypothesis=Hypothesis(
            statement=args.hypothesis,
            success_criteria=criteria,
            provenance={"proposed_by": "human"},
        ),
        baseline=TargetRef(
            name=args.baseline.split(":")[0], version=args.baseline.split(":")[1]
        ),
        candidate=TargetRef(
            name=args.candidate.split(":")[0], version=args.candidate.split(":")[1]
        ),
        metrics=metrics,
        trials=args.trials,
        seed=args.seed,
        procedure_params=procedure_params,
    )
    from agency.experiments.demo import _CORPUS

    runner = ExperimentRunner(stack.strategies, services={"search_corpus": _CORPUS})
    finished = await runner.run(experiment)
    store = ExperimentRegistry()
    await store.save(finished)
    stored = await record_experiment(finished, _experiment_memory(args))
    if args.json:
        print(finished.model_dump_json(indent=2))
        return 0 if finished.evaluation and finished.evaluation.verdict.value in {
            "success", "failure", "inconclusive"
        } else 1
    _print_experiment_detail(finished)
    if stored:
        print(f"  memories    : {len(stored)} stored from this experiment")
    return 0 if finished.evaluation else 1


async def _run_experiment_demo(args: argparse.Namespace) -> int:
    """Built-in offline experiment: real measurements, no network, no APIs."""
    from agency.experiments.experiments_store import ExperimentRegistry
    from agency.experiments.memory_bridge import record_experiment
    from agency.experiments.models import (
        Experiment,
        Hypothesis,
        MetricDefinition,
        SuccessCriterion,
        TargetRef,
    )
    from agency.experiments.runner import ExperimentRunner

    stack = _lab_stack(args)
    goal = args.goal
    experiment = Experiment(
        name="research-strategy-demo",
        objective=f"Compare research query strategies for: {goal}",
        description=(
            "Offline demo: single-query (baseline) vs multi-query (candidate) "
            "research strategies against an injected deterministic corpus."
        ),
        hypothesis=Hypothesis(
            statement=(
                "A multi-query research strategy discovers more unique sources "
                "than a single-query strategy, without a materially higher "
                "duplicate rate."
            ),
            rationale="more diverse queries should cover more of the corpus",
            success_criteria=[
                SuccessCriterion(
                    kind="improves", metric="sources_found", threshold=0.15
                ),
                # Absolute cap: duplicate_rate's baseline is 0 by construction,
                # so a relative criterion is unmeasurable there. Threshold
                # sized for the union-of-4-queries dedupe arithmetic.
                SuccessCriterion(
                    kind="max", metric="duplicate_rate", threshold=0.8
                ),
                SuccessCriterion(
                    kind="max", metric="duration_ms", threshold=50.0
                ),
            ],
            provenance={"proposed_by": "human", "demo": True},
        ),
        baseline=TargetRef(name="single_query", version="v1"),
        candidate=TargetRef(name="multi_query", version="v1"),
        metrics=[
            MetricDefinition(
                name="sources_found", direction="maximize",
                description="unique source URLs discovered",
            ),
            MetricDefinition(
                name="duplicate_rate", direction="minimize",
                description="1 − unique/discovered results",
            ),
            MetricDefinition(name="search_errors", direction="minimize"),
            MetricDefinition(name="duration_ms", direction="minimize"),
        ],
        trials=args.trials,
        seed=42,
        procedure_params={"goal": goal},
    )

    from agency.experiments.demo import _CORPUS

    runner = ExperimentRunner(
        stack.strategies, services={"search_corpus": _CORPUS}
    )
    finished = await runner.run(experiment)
    await ExperimentRegistry().save(finished)
    stored = await record_experiment(finished, _experiment_memory(args))

    if args.json:
        print(finished.model_dump_json(indent=2))
    else:
        _print_experiment_report(finished, goal=goal)
        if stored:
            print(f"  memories    : {len(stored)} stored "
                  "(episodic + procedural on success)")
    evaluation = finished.evaluation
    if evaluation is None or evaluation.verdict.value == "error":
        return 1
    return 0


def _print_experiment_report(experiment: Any, *, goal: str = "") -> None:
    comparison = experiment.comparison
    evaluation = experiment.evaluation
    print(f"SKYNET experiment {experiment.id[:12]}  [{experiment.status.value}]")
    if goal:
        print(f"  goal        : {goal}")
    print(f"  hypothesis  : {experiment.hypothesis.statement}")
    print(
        f"  arms        : {experiment.baseline.ref} (baseline) vs "
        f"{experiment.candidate.ref} (candidate), "
        f"{experiment.trials} trials each, seed={experiment.seed}"
    )
    print("  criteria    : "
          + "; ".join(c.description for c in experiment.success_criteria))
    if comparison is not None:
        print(f"  baseline    : {comparison.baseline.ref} mean {comparison.baseline.mean}")
        print(f"  candidate   : {comparison.candidate.ref} mean {comparison.candidate.mean}")
        for metric, delta in comparison.delta.items():
            rel = comparison.relative.get(metric)
            rel_text = f" ({rel:+.1%})" if rel is not None else ""
            print(f"    - {metric}: {delta:+.4g}{rel_text}")
        if comparison.improvements:
            print(f"  improved    : {', '.join(comparison.improvements)}")
        if comparison.regressions:
            print(f"  regressed   : {', '.join(comparison.regressions)}")
        print("  note        : descriptive comparison; no significance test performed")
    if evaluation is not None:
        print(f"  verdict     : {evaluation.verdict.value.upper()} — {evaluation.explanation}")
        for result in evaluation.criteria_results:
            measured = result.get("measured")
            measured_text = f" measured={measured}" if measured is not None else ""
            detail = result.get("detail")
            detail_text = f" ({detail})" if detail else ""
            print(
                f"    - [{result['outcome']}] {result['criterion']}"
                f"{measured_text}{detail_text}"
            )
        for line in evaluation.evidence:
            print(f"    · {line}")
    if experiment.error:
        print(f"  error       : {experiment.error}")


def _print_experiment_detail(experiment: Any) -> None:
    _print_experiment_report(experiment)
    print(f"  environment : {experiment.environment}")
    print(f"  created     : {experiment.created_at:%Y-%m-%d %H:%M:%S} UTC")
    if experiment.started_at and experiment.completed_at:
        seconds = (experiment.completed_at - experiment.started_at).total_seconds()
        print(f"  duration    : {seconds:.2f}s")
    for trial in experiment.trials_results:
        marker = "ok" if trial.success else f"FAILED: {trial.error}"
        print(
            f"    trial [{trial.arm}/{trial.index}] {marker} "
            f"{trial.metrics} ({trial.duration_ms} ms)"
        )


# -- Self-improvement (agency.improve, Phase P7) ----------------------------------


def _improve_memory(args: argparse.Namespace):
    """Memory manager for improvement recording (backend per operator flag)."""
    from agency.bootstrap import build_memory_stack

    if not getattr(args, "memory_backend", None):
        return None
    updates: dict[str, Any] = {"memory_backend": args.memory_backend}
    return build_memory_stack(SkynetSettings(**updates)).manager


def _build_improve_pipeline(args: argparse.Namespace):
    """Lab pipeline wired from settings + demo corpus for offline detection."""
    from agency.bootstrap import build_lab_stack
    from agency.experiments.demo import _CORPUS

    settings = SkynetSettings()
    memory = _improve_memory(args)
    stack = build_lab_stack(
        settings,
        memory=memory,
        services={"search_corpus": _CORPUS},
    )
    return stack.improvement


async def _run_improve_detect(args: argparse.Namespace) -> int:
    """Detect weaknesses from recorded history (experiences + experiments)."""
    from agency.experiments.experiments_store import ExperimentRegistry
    from agency.improve.detector import DetectorInput

    pipeline = _build_improve_pipeline(args)
    # Experiences: the operator's recorded run history (database backend in
    # production). The CLI works offline, so the experiments registry —
    # real persisted experiment history — is the data source here.
    experiments_store = ExperimentRegistry(SkynetSettings().experiments_registry_path)
    all_experiments = []
    for summary in await experiments_store.list(limit=100):
        record = await experiments_store.get(summary.id)
        if record is not None:
            all_experiments.append(record)
    detected = await pipeline.detect(
        DetectorInput(experiments=all_experiments)
    )
    known = pipeline.list_weaknesses()
    if args.json:
        print(json.dumps({"detected": len(detected), "known": len(known)}))
        return 0
    print(f"weaknesses detected: {len(detected)} (total known: {len(known)})")
    for weakness in known:
        print(
            f"  - [{weakness.severity:.2f}] {weakness.category.value} "
            f"({weakness.status.value}) freq={weakness.frequency} id={weakness.id[:12]}"
        )
        print(f"      {weakness.description}")
        print(f"      component: {weakness.affected_component or 'n/a'}")
    return 0


async def _run_improve_propose(args: argparse.Namespace) -> int:
    from agency.improve.pipeline import build_proposal_artifacts  # noqa: F401

    pipeline = _build_improve_pipeline(args)
    config: dict[str, Any] = {}
    for pair in args.param or []:
        key, _, raw = pair.partition("=")
        value: Any = raw
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        config[key.strip()] = value
    try:
        hypothesis, proposal, candidate = await pipeline.propose(
            args.weakness_id,
            origin=args.origin,
            candidate_config=config,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "hypothesis_id": hypothesis.id,
                    "proposal_id": proposal.id,
                    "candidate_ref": candidate.ref,
                    "statement": hypothesis.statement,
                }
            )
        )
        return 0
    print(f"SKYNET proposal {proposal.id[:12]}  [{proposal.status.value}]")
    print(f"  weakness    : {proposal.weakness_id[:12]}")
    print(f"  hypothesis  : {hypothesis.statement}")
    print(f"  target      : {proposal.target} -> {candidate.ref}")
    print(f"  change      : {proposal.proposed_change}")
    print("  criteria    : " + "; ".join(c.description for c in proposal.acceptance_criteria))
    print("  rollback    : " + proposal.rollback_plan)
    if proposal.metadata.get("prior_attempts"):
        print(
            f"  prior tries : {len(proposal.metadata['prior_attempts'])} "
            "(see improve history)"
        )
    return 0


async def _run_improve_history(args: argparse.Namespace) -> int:
    pipeline = _build_improve_pipeline(args)
    proposals = pipeline.proposals()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": p.id,
                        "status": p.status.value,
                        "decision": p.decision,
                        "target": p.target,
                        "candidate": p.candidate_ref,
                        "reason": p.decision_reason,
                    }
                    for p in proposals
                ]
            )
        )
        return 0
    if not proposals:
        print("no improvement proposals recorded")
        return 0
    print(f"improvement proposals: {len(proposals)}")
    for p in proposals:
        decision = p.decision or "-"
        print(
            f"  - {p.id[:12]}  [{p.status.value}]  {p.target} vs "
            f"{p.candidate_ref or '-'}  decision={decision}"
        )
        if p.decision_reason:
            print(f"      reason: {p.decision_reason}")
    return 0


async def _run_improve(args: argparse.Namespace) -> int:
    command = getattr(args, "improve_command", None)
    if command == "detect":
        return await _run_improve_detect(args)
    if command == "propose":
        return await _run_improve_propose(args)
    if command == "history":
        return await _run_improve_history(args)
    if command == "demo":
        from agency.improve.demo import run_demo

        return await run_demo(
            memory_backend=args.memory_backend, as_json=args.json
        )
    print(f"error: unknown improve command {command!r}", file=sys.stderr)
    return 1


def _build_orchestrator() -> Any:
    """Assemble the autonomous orchestrator (real subsystems, offline adapters)."""
    from agency.bootstrap import build_orchestrator

    # The orchestrator's DB-free default: memory-storage core, local corpus
    # research and local-mock comms unless settings enable the real stacks.
    settings = SkynetSettings(storage_backend="memory", perception_adapter="none")
    return build_orchestrator(settings)


def _print_run(run: Any, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "id": run.id,
                    "objective": run.objective,
                    "status": run.status.value,
                    "stage": run.current_stage.value,
                    "iteration": run.iteration,
                    "usage": run.usage,
                    "evidence": len(run.evidence),
                    "decisions": [d.action for d in run.decisions],
                    "result": run.result,
                    "errors": run.errors,
                },
                indent=2,
            )
        )
        return
    print(f"autonomous run {run.id[:12]}  [{run.status.value}]  stage={run.current_stage.value}")
    print(f"  objective : {run.objective}")
    print(
        f"  progress  : iteration {run.iteration}, {len(run.evidence)} evidence item(s), "
        f"{len(run.decisions)} decision(s)"
    )
    print(f"  usage     : {run.usage}")
    if run.pending_approval is not None:
        print(f"  APPROVAL  : {run.pending_approval.what}")
    if run.result:
        print(f"  result    : {run.result}")
    for error in run.errors[-3:]:
        print(f"  error     : {error}")


async def _run_autonomous_start(args: argparse.Namespace) -> int:
    from agency.orchestrator.models import OrchestratorBudgets

    settings = SkynetSettings()
    budgets = OrchestratorBudgets(
        max_iterations=(
            args.max_iterations
            if args.max_iterations is not None
            else settings.autonomous_max_iterations
        ),
        max_runtime_seconds=settings.autonomous_max_runtime_seconds,
        max_web_requests=settings.autonomous_max_web_requests,
        max_ai_requests=settings.autonomous_max_ai_requests,
        max_experiments=settings.autonomous_max_experiments,
    )
    configuration: dict[str, Any] = {}
    if args.question:
        configuration["ai_questions"] = list(args.question)
    if args.steps:
        try:
            steps = json.loads(args.steps)
        except json.JSONDecodeError as exc:
            print(f"error: --steps is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(steps, list):
            print("error: --steps must be a JSON array", file=sys.stderr)
            return 2
        configuration["core_goal_params"] = {"steps": steps}

    orchestrator = _build_orchestrator()
    run = await orchestrator.start(
        args.objective, budgets=budgets, configuration=configuration
    )
    run = await orchestrator.execute(run.id)
    _print_run(run, as_json=args.json)
    return 0 if run.status.value in {"completed", "waiting", "paused"} else 1


async def _run_autonomous_status(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    runs = await orchestrator._store.list()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "status": r.status.value,
                        "stage": r.current_stage.value,
                        "iteration": r.iteration,
                        "objective": r.objective,
                    }
                    for r in runs
                ],
                indent=2,
            )
        )
        return 0
    if not runs:
        print("no autonomous runs recorded")
        return 0
    print(f"autonomous runs: {len(runs)}")
    for r in runs:
        print(f"  - {r.id[:12]}  [{r.status.value}]  {r.current_stage.value}  {r.objective[:60]}")
    return 0


async def _run_autonomous_lifecycle(args: argparse.Namespace, command: str) -> int:
    orchestrator = _build_orchestrator()
    method = getattr(orchestrator, command)
    run = await method(args.run_id)
    if run is None:
        print(f"error: unknown run id {args.run_id!r}", file=sys.stderr)
        return 1
    _print_run(run, as_json=args.json)
    return 0


async def _run_autonomous_inspect(args: argparse.Namespace) -> int:
    orchestrator = _build_orchestrator()
    run = await orchestrator._store.get(args.run_id)
    if run is None:
        print(f"error: unknown run id {args.run_id!r}", file=sys.stderr)
        return 1
    _print_run(run, as_json=args.json)
    if not args.json:
        print("  decisions :")
        for d in run.decisions:
            print(f"    - {d.action}: {d.reason}")
        if run.evidence:
            print("  evidence  :")
            for e in run.evidence[-8:]:
                print(f"    - [{e.source_type}] {e.source} ({e.confidence:.2f})")
    return 0


async def _run_autonomous(args: argparse.Namespace) -> int:
    command = getattr(args, "autonomous_command", None)
    if command == "start":
        return await _run_autonomous_start(args)
    if command == "status":
        return await _run_autonomous_status(args)
    if command in {"pause", "resume", "cancel"}:
        return await _run_autonomous_lifecycle(args, command)
    if command == "inspect":
        return await _run_autonomous_inspect(args)
    if command == "demo":
        from agency.orchestrator.demo import run_demo

        return await run_demo(as_json=args.json)
    print(f"error: unknown autonomous command {command!r}", file=sys.stderr)
    return 1


# -- Real capabilities (Phase P9) ------------------------------------------------------


def _print_health(status: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(status.model_dump(mode="json"), indent=2))
        return
    print(status.summary_line())
    if not status.ok and status.detail:
        print(f"    detail: {status.detail}")


async def _run_provider(args: argparse.Namespace) -> int:
    """`skynet provider status|test` — classify or minimally probe providers."""
    from agency.orchestrator.health import check_llm_provider, check_web_stack

    settings = SkynetSettings(storage_backend="memory", perception_adapter="none")
    statuses: list[Any]
    if args.provider_command == "status":
        # Classification only: no requests are sent in `status`.
        statuses = []
        for component, provider, model in (
            ("llm", settings.llm_provider, settings.llm_model),
            ("web", settings.search_provider, ""),
            ("comms", settings.ai_providers, ""),
        ):
            if provider == "mock":
                kind, label = "mock", "MOCK"
            elif provider == "none":
                kind, label = "unavailable", "UNAVAILABLE"
            else:
                kind, label = "real", "REAL"
            from agency.orchestrator.health import HealthStatus

            statuses.append(
                HealthStatus(
                    component=component,
                    provider=provider,
                    kind=kind,  # type: ignore[arg-type]
                    model=model,
                    reachable=True,  # configured; liveness not probed here
                    ok=kind == "mock",
                    label=label,
                    detail=(
                        "classification only — always available offline"
                        if kind == "mock"
                        else "configured REAL — run `provider test` to verify liveness"
                    ),
                )
            )
    else:
        checks = []
        if args.component in {"all", "llm"}:
            checks.append(check_llm_provider(settings))
        if args.component in {"all", "web"}:
            checks.append(check_web_stack(settings))
        statuses = [await c for c in checks]

    # `status` is informational: rc 0 unless something is configured OFF.
    # `test` reflects real liveness: rc 0 only when every probe passed.
    if args.provider_command == "status":
        ok = all(s.kind != "unavailable" for s in statuses)
    else:
        ok = all(s.ok for s in statuses)
    if args.json:
        print(json.dumps([s.model_dump(mode="json") for s in statuses], indent=2))
        return 0 if ok else 1
    for status in statuses:
        _print_health(status, as_json=False)
    return 0 if ok else 1


async def _run_real(args: argparse.Namespace) -> int:
    """`skynet run start|status|inspect` — bounded real-capability runs."""
    from agency.bootstrap import build_orchestrator
    from agency.orchestrator.demo import OBJECTIVE as DEMO_OBJECTIVE

    settings = SkynetSettings(storage_backend="memory", perception_adapter="none")
    if getattr(args, "offline", False):
        settings = settings.model_copy(
            update={
                "enable_web_tools": False,
                "enable_external_comms": False,
                "web_offline_mode": True,
            }
        )
    else:
        # Real execution: web/comms seams follow settings (enable_web_tools,
        # enable_external_comms). Real LLM rides the llm_provider settings.
        settings = settings.model_copy(
            update={"enable_web_tools": True, "enable_external_comms": True}
        )

    if args.run_command == "status":
        orchestrator = build_orchestrator(settings)
        runs = await orchestrator._store.list()
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "id": r.id,
                            "status": r.status.value,
                            "stage": r.current_stage.value,
                            "objective": r.objective,
                        }
                        for r in runs
                    ],
                    indent=2,
                )
            )
            return 0
        if not runs:
            print("no runs recorded")
            return 0
        for r in runs:
            print(
                f"  - {r.id[:12]}  [{r.status.value}]  {r.current_stage.value}  "
                f"{r.objective[:60]}"
            )
        return 0

    if args.run_command == "inspect":
        orchestrator = build_orchestrator(settings)
        run = await orchestrator._store.get(args.run_id)
        if run is None:
            print(f"error: unknown run id {args.run_id!r}", file=sys.stderr)
            return 1
        _print_run(run, as_json=args.json)
        if not args.json:
            for e in run.evidence[-10:]:
                ec = e.provenance.get("evidence_class", "")
                print(f"    - [{e.source_type}/{ec}] {e.source} ({e.confidence:.2f})")
        return 0

    if args.run_command == "demo":
        from agency.orchestrator.real_demo import run_real_demo

        return await run_real_demo(
            objective=args.objective,
            queries=list(args.query) if args.query else None,
            as_json=args.json,
        )

    # -- start ------------------------------------------------------------------
    from agency.orchestrator.models import OrchestratorBudgets

    configuration: dict[str, Any] = {}
    if getattr(args, "query", None):
        configuration["research_queries"] = list(args.query)
    if getattr(args, "question", None):
        configuration["ai_questions"] = list(args.question)
    budgets = OrchestratorBudgets(
        max_iterations=(
            args.max_iterations
            if args.max_iterations is not None
            else settings.autonomous_max_iterations
        ),
        max_runtime_seconds=settings.autonomous_max_runtime_seconds,
        max_web_requests=settings.autonomous_max_web_requests,
        max_ai_requests=settings.autonomous_max_ai_requests,
        max_experiments=settings.autonomous_max_experiments,
    )
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, **payload: Any) -> None:
        events.append((event, payload))

    orchestrator = build_orchestrator(settings, emit=emit)
    run = await orchestrator.start(
        args.objective or DEMO_OBJECTIVE, budgets=budgets, configuration=configuration
    )
    run = await orchestrator.execute(run.id)

    if args.json:
        _print_run(run, as_json=True)
        return 0 if run.status.value in {"completed", "waiting", "paused"} else 1

    print(f"real run {run.id[:12]}  [{run.status.value}]  stage={run.current_stage.value}")
    print(f"  objective : {run.objective}")
    budget_line = (
        f"  budgets   : iterations={budgets.max_iterations}"
        f" web={budgets.max_web_requests} ai={budgets.max_ai_requests}"
    )
    print(budget_line)
    print(f"  usage     : {run.usage}")
    stages_seen: list[str] = []
    for event, payload in events:
        if event == "AUTONOMOUS_STAGE_STARTED" and payload.get("stage") not in stages_seen:
            stages_seen.append(payload.get("stage"))
    print(f"  stages    : {' -> '.join(str(s).upper() for s in stages_seen)}")
    print(f"  evidence  : {len(run.evidence)} item(s)")
    for e in run.evidence[-8:]:
        ec = e.provenance.get("evidence_class", "")
        print(f"    - [{e.source_type}/{ec}] {e.source} ({e.confidence:.2f})")
    by_class: dict[str, int] = {}
    for e in run.evidence:
        ec = str(e.provenance.get("evidence_class", "UNLABELLED"))
        by_class[ec] = by_class.get(ec, 0) + 1
    print(f"  classes   : {by_class}")
    if run.security_report:
        print(
            f"  security  : {run.security_report.get('status')} — "
            f"{run.security_report.get('total_flags')} flag(s)"
        )
    if run.result:
        print(f"  result    : {run.result}")
    for error in run.errors[-3:]:
        print(f"  error     : {error}")
    return 0 if run.status.value in {"completed", "waiting", "paused"} else 1


def main(argv: list[str] | None = None) -> int:
    """Synchronous entry point (also exposed as the ``skynet`` console script)."""
    # External AI/web content is arbitrary Unicode; Windows consoles default
    # to legacy codepages (e.g. cp1252) that cannot encode it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
