"""CLI tests (memory backend — no database required)."""

from __future__ import annotations

import json

import pytest

from agency.cli import _build_parser, _run, main


def test_parser_requires_goal() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["run"])


def test_parser_defaults() -> None:
    args = _build_parser().parse_args(["run", "--goal", "G"])
    assert args.goal == "G"
    assert args.storage is None
    assert args.steps is None
    assert args.json is False
    assert args.description == ""


@pytest.mark.asyncio
async def test_run_memory_happy_path_exit_zero(capsys: pytest.CaptureFixture) -> None:
    args = _build_parser().parse_args(
        ["run", "--goal", "CLI happy", "--storage", "memory",
         "--steps", '[{"type": "echo", "params": {"text": "hi"}}]']
    )
    code = await _run(args)
    assert code == 0
    out = capsys.readouterr().out
    assert "SKYNET run" in out
    assert "COMPLETED" in out
    assert "- echo: ok" in out


@pytest.mark.asyncio
async def test_run_json_output_is_valid_json(capsys: pytest.CaptureFixture) -> None:
    args = _build_parser().parse_args(
        ["run", "--goal", "CLI json", "--storage", "memory",
         "--steps", '[{"type": "echo", "params": {"text": "x"}}]', "--json"]
    )
    code = await _run(args)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["goal_title"] == "CLI json"
    assert payload["actions"][0]["type"] == "echo"


@pytest.mark.asyncio
async def test_run_failed_goal_exit_one(capsys: pytest.CaptureFixture) -> None:
    args = _build_parser().parse_args(
        ["run", "--goal", "CLI failing", "--storage", "memory",
         "--steps", '[{"type": "boom"}]']
    )
    code = await _run(args)
    assert code == 1
    out = capsys.readouterr().out
    assert "unknown action type" in out


@pytest.mark.asyncio
async def test_run_invalid_steps_json_reports_error(capsys: pytest.CaptureFixture) -> None:
    args = _build_parser().parse_args(
        ["run", "--goal", "Bad json", "--storage", "memory", "--steps", "{not json"]
    )
    code = await _run(args)
    assert code == 2
    err = capsys.readouterr().err
    assert "not valid JSON" in err


@pytest.mark.asyncio
async def test_run_steps_must_be_array(capsys: pytest.CaptureFixture) -> None:
    args = _build_parser().parse_args(
        ["run", "--goal", "Not array", "--storage", "memory", "--steps", '{"type": "echo"}']
    )
    code = await _run(args)
    assert code == 2
    assert "JSON array" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_max_steps_override_applies(capsys: pytest.CaptureFixture) -> None:
    steps = json.dumps([{"type": "echo"}] * 3)
    args = _build_parser().parse_args(
        ["run", "--goal", "Budgeted", "--storage", "memory",
         "--max-steps", "2", "--steps", steps]
    )
    code = await _run(args)
    assert code == 1  # budget exhaustion -> failed run
    out = capsys.readouterr().out
    assert "max_steps_per_run" in out


def test_main_sync_entry() -> None:
    code = main(["run", "--goal", "Sync entry", "--storage", "memory",
                 "--steps", '[{"type": "echo", "params": {"text": "y"}}]'])
    assert code == 0


def test_missing_command_exits_with_usage() -> None:
    with pytest.raises(SystemExit):
        main([])
