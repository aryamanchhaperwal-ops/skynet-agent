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
