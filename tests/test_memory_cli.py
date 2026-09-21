"""CLI memory-command tests (temp sqlite backend; fully offline)."""

from __future__ import annotations

import json

from agency.cli import main


class TestRememberCommand:
    def test_remember_stores_memory(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setenv("SKYNET_MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("SKYNET_MEMORY_SQLITE_PATH", str(tmp_path / "m.db"))
        exit_code = main(
            [
                "remember",
                "Retrieval quality beats storage volume for agent memory.",
                "--type",
                "semantic",
                "--importance",
                "0.9",
                "--tag",
                "memory",
                "--tag",
                "design",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "memory stored:" in out
        assert "semantic" in out


class TestRecallCommand:
    def test_recall_finds_stored_memory(self, capsys, tmp_path, monkeypatch):
        db = str(tmp_path / "m.db")
        monkeypatch.setenv("SKYNET_MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("SKYNET_MEMORY_SQLITE_PATH", db)
        main(["remember", "Wikipedia is a solid starting source for concepts"])
        exit_code = main(["recall", "wikipedia source concepts"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "recalled 1 memory" in out
        assert "Wikipedia" in out

    def test_recall_json_mode(self, capsys, tmp_path, monkeypatch):
        db = str(tmp_path / "m.db")
        monkeypatch.setenv("SKYNET_MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("SKYNET_MEMORY_SQLITE_PATH", db)
        main(["remember", "provenance chain fact for json recall"])
        capsys.readouterr()  # discard the remember output
        exit_code = main(["recall", "provenance", "--json"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["type"] == "semantic"
        assert "provenance" in payload[0]

    def test_recall_empty_store_fails_cleanly(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setenv("SKYNET_MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("SKYNET_MEMORY_SQLITE_PATH", str(tmp_path / "empty.db"))
        exit_code = main(["recall", "anything"])
        assert exit_code == 1
        assert "no memories found" in capsys.readouterr().out


class TestMemoryStatsCommand:
    def test_stats_counts(self, capsys, tmp_path, monkeypatch):
        db = str(tmp_path / "m.db")
        monkeypatch.setenv("SKYNET_MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("SKYNET_MEMORY_SQLITE_PATH", db)
        main(["remember", "fact one"])
        main(["remember", "fact two", "--type", "episodic"])
        exit_code = main(["memory-stats"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "2 record(s)" in out
        assert "semantic" in out and "episodic" in out
