"""CLI `research` subcommand tests (core mocked; no network)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agency.cli import main
from agency.evaluator import Evaluation
from agency.loop import RunSummary


def _summary(**overrides) -> RunSummary:
    defaults = dict(
        run_id="run-x",
        goal_id="goal-x",
        goal_title="research goal",
        status="completed",
        planner="deterministic",
        perception="none",
        steps_planned=1,
        steps_executed=1,
        actions=[
            {"type": "web_research", "success": True, "duration_ms": 5, "error": None}
        ],
        observations=2,
        evaluation=Evaluation(subject="run", level="run", passed=True, score=1.0, notes="ok"),
        experiences=5,
        events=12,
        final_state={
            "actions": [
                {
                    "spec": {"type": "web_research"},
                    "result": {
                        "success": True,
                        "output": {
                            "package": {
                                "queries": ["q1"],
                                "sources": [
                                    {
                                        "title": "Alpha",
                                        "url": "https://a.example/1",
                                        "extraction_status": "succeeded",
                                        "text": "body",
                                    }
                                ],
                                "failures": [
                                    {"url": "https://d.example/", "error": "HTTP 500", "stage": "fetch"}
                                ],
                                "observations": [{}, {}],
                                "warnings": [],
                            }
                        },
                    },
                }
            ]
        },
        error=None,
    )
    defaults.update(overrides)
    return RunSummary(**defaults)


@pytest.fixture
def research_core_patch():
    """Patch build_core + ensure_core_schema so the CLI runs fully in-process."""
    fake_core = type("C", (), {})()
    fake_core.run_goal = AsyncMock(return_value=_summary())
    with (
        patch("agency.cli.build_core", return_value=fake_core),
        patch("agency.cli.ensure_core_schema", new=AsyncMock()),
    ):
        yield fake_core


class TestResearchCommand:
    def test_research_prints_structured_summary(self, research_core_patch, capsys):
        exit_code = main(["research", "How are agents using memory?"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "SKYNET research" in out
        assert "sources     : 1 retrieved, 1 failed" in out
        assert "https://a.example/1" in out
        assert "observations: 2 structured" in out

    def test_research_json_mode(self, research_core_patch, capsys):
        exit_code = main(["research", "goal", "--json"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["run_id"] == "run-x"
        assert payload["status"] == "completed"

    def test_research_passes_explicit_queries(self, research_core_patch, capsys):
        main(["research", "goal", "--query", "one", "--query", "two", "--storage", "memory"])
        call_args = research_core_patch.run_goal.call_args
        spec = call_args.args[0] if call_args.args else call_args.kwargs["spec"]
        step = spec.params["steps"][0]
        assert step["type"] == "web_research"
        assert step["params"]["queries"] == ["one", "two"]

    def test_research_enables_web_tools_for_the_run(self, research_core_patch):

        captured = {}

        def fake_build(settings):
            captured.update(settings.model_dump())
            return research_core_patch

        with patch("agency.cli.build_core", side_effect=fake_build):
            main(["research", "goal", "--storage", "memory"])
        assert captured["enable_web_tools"] is True
        assert "web_research" in captured["enabled_actions"]

    def test_failed_summary_exits_nonzero(self, capsys):
        fake_core = type("C", (), {})()
        failing = _summary(status="failed", error="boom")
        failing = failing.model_copy(
            update={
                "evaluation": Evaluation(subject="run", level="run", passed=False, score=0.0, notes="bad"),
                "actions": [
                    {"type": "web_research", "success": False, "duration_ms": 1, "error": "boom"}
                ],
            }
        )
        fake_core.run_goal = AsyncMock(return_value=failing)
        with (
            patch("agency.cli.build_core", return_value=fake_core),
            patch("agency.cli.ensure_core_schema", new=AsyncMock()),
        ):
            exit_code = main(["research", "goal", "--storage", "memory"])
        assert exit_code == 1
