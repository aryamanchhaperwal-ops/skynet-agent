"""CLI AI-communication subcommand tests (mock provider; fully offline)."""

from __future__ import annotations

from agency.cli import main


class TestAIParticipantsCommand:
    def test_lists_mock_participant_with_capabilities(self, capsys):
        exit_code = main(["ai-participants"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "AI participants: 1" in out
        assert "mock:mock-agent-1" in out
        assert "reasoning" in out
        assert "research" in out

    def test_capability_filter_excludes_undeclared(self, capsys):
        exit_code = main(["ai-participants", "--capability", "vision"])
        assert exit_code == 1  # no participant declares vision
        out = capsys.readouterr().out
        assert "no AI participants available" in out

    def test_json_mode(self, capsys):
        import json

        exit_code = main(["ai-participants", "--json"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["participant_id"] == "mock:mock-agent-1"
        assert "reasoning" in payload[0]["capabilities"]

    def test_unknown_provider_fails_cleanly(self, capsys):
        exit_code = main(["ai-participants", "--provider", "does-not-exist"])
        assert exit_code == 2
        err = capsys.readouterr().err
        assert "unknown AI provider" in err


class TestAIAskCommand:
    def test_ask_runs_offline_conversation(self, capsys):
        exit_code = main(
            [
                "ai-ask",
                "mock:mock-agent-1",
                "What matters in agent memory?",
                "--storage",
                "memory",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "SKYNET ai-conversation" in out
        assert "participant : mock:mock-agent-1" in out
        assert "Q: What matters in agent memory?" in out
        assert "A:" in out
        assert "PASSED" in out
        assert "provenance" in out

    def test_ask_with_follow_ups_counts_turns(self, capsys):
        exit_code = main(
            [
                "ai-ask",
                "mock:mock-agent-1",
                "main question",
                "--follow-up",
                "follow one",
                "--storage",
                "memory",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "turns       : 2" in out

    def test_unknown_participant_fails_with_action_error(self, capsys):
        exit_code = main(
            ["ai-ask", "mock:nobody", "hi", "--storage", "memory"]
        )
        assert exit_code == 1
        out = capsys.readouterr().out
        assert "FAILED" in out


class TestAIDemoCommand:
    def test_demo_compares_three_mock_participants(self, capsys):
        exit_code = main(
            [
                "ai-demo",
                "How should long-term memory in AI agents be built?",
                "--storage",
                "memory",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "SKYNET ai-comparison" in out
        assert "3 asked" in out
        assert "3 responded" in out
        assert "agreement   :" in out
        assert "note        : agreement is recorded, not treated as truth" in out

    def test_demo_json_mode(self, capsys):
        import json

        exit_code = main(
            ["ai-demo", "q", "--storage", "memory", "--json"]
        )
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "completed"
