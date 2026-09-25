"""Autonomous CLI commands: start/status/inspect/pause/resume/cancel/demo."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from agency import cli


def asyncio_run_isolated(coro: Any) -> None:
    """Run a coroutine on a fresh loop in a helper thread (pytest has none)."""
    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - surfaced below
            result["exc"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join(timeout=120)
    assert not thread.is_alive(), "scenario hung"
    if result.get("exc") is not None:
        raise result["exc"]


@pytest.fixture
def tmp_runs_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """Point the autonomous runs store at a temp file for every CLI call."""
    path = tmp_path / "cli-runs.jsonl"

    real_build = cli._build_orchestrator

    def build_with_tmp() -> Any:
        orch = real_build()
        from agency.orchestrator.store import RunStore

        orch._store = RunStore(path)
        return orch

    monkeypatch.setattr(cli, "_build_orchestrator", build_with_tmp)
    return path


def _main(argv: list[str]) -> int:
    return cli.main(argv)


GOOD_STEPS = '[{"type": "echo", "params": {"message": "synthesis"}}]'


def test_cli_start_runs_to_completion(
    tmp_runs_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _main(
        [
            "autonomous", "start",
            "--objective", "approaches to long-term memory in autonomous AI agents",
            "--max-iterations", "4",
            "--question", "What is known about memory architectures?",
            "--steps", GOOD_STEPS,
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["result"]["completed"] is True
    assert payload["evidence"] >= 3


def test_cli_status_lists_runs(
    tmp_runs_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(
        ["autonomous", "start", "--objective", "status listing check", "--json"]
    )
    capsys.readouterr()
    rc = _main(["autonomous", "status", "--json"])
    assert rc == 0
    runs = json.loads(capsys.readouterr().out)
    assert any(r["objective"] == "status listing check" for r in runs)


def test_cli_pause_resume_lifecycle(
    tmp_runs_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(
        [
            "autonomous", "start",
            "--objective", "lifecycle target",
            "--max-iterations", "3",
            "--steps", GOOD_STEPS,
            "--json",
        ]
    )
    run_id = json.loads(capsys.readouterr().out)["id"]

    # The run already finished (small budget); pause of a terminal run is
    # refused cleanly with exit code 1 and an error line.
    rc = _main(["autonomous", "pause", run_id])
    assert rc == 1
    assert "error" in capsys.readouterr().err

    # inspect works on the finished run
    rc = _main(["autonomous", "inspect", run_id, "--json"])
    assert rc == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["id"] == run_id


def test_cli_pause_active_run_then_resume(
    tmp_runs_path: Any, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pause an in-flight run through a second store, then resume it."""
    from agency.orchestrator.models import OrchestratorBudgets, RunStatus
    from agency.orchestrator.store import RunStore

    orch = cli._build_orchestrator()

    async def scenario() -> None:
        run = await orch.start(
            "pause target",
            budgets=OrchestratorBudgets(max_iterations=3),
        )
        other = RunStore(orch._store._path)  # "second process"
        await other.pause(run.id)
        resumed = await other.resume(run.id)
        assert resumed is not None and resumed.status is RunStatus.RUNNING
        final = await orch.execute(run.id)
        assert final.status in {RunStatus.COMPLETED, RunStatus.FAILED}

    asyncio_run_isolated(scenario())


def test_cli_demo(
    tmp_runs_path: Any, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demo subcommand runs the deterministic offline demonstration."""
    rc = _main(["autonomous", "demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "INITIALIZE" in out
    assert "COMPLETE" in out
    assert "status=completed" in out
