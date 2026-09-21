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
    return parser


async def _run(args: argparse.Namespace) -> int:
    if getattr(args, "command", None) == "research":
        return await _run_research(args)
    return await _run_goal(args)


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
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
