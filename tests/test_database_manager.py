"""Tests for workflow database management."""

from __future__ import annotations

import sqlite3

from src.database_manager import DatabaseManager


def test_initializes_expected_tables(tmp_path):
    db_path = tmp_path / "state.db"
    DatabaseManager(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert "task_completions" in tables
    assert "task_cache" in tables


def test_records_task_completion(tmp_path):
    db = DatabaseManager(tmp_path / "state.db")
    db.record_task_completion(
        task_name="paper_signal_cycle",
        run_type="signal",
        mode="paper",
        status="success",
        metadata={"equity": 12345.67},
    )

    with sqlite3.connect(tmp_path / "state.db") as conn:
        row = conn.execute(
            "SELECT task_name, run_type, mode, status, metadata_json FROM task_completions"
        ).fetchone()

    assert row is not None
    assert row[0] == "paper_signal_cycle"
    assert row[1] == "signal"
    assert row[2] == "paper"
    assert row[3] == "success"
    assert '"equity": 12345.67' in row[4]


def test_cache_set_and_get(tmp_path):
    db = DatabaseManager(tmp_path / "state.db")

    assert db.get_cache("missing") is None

    db.set_cache("evaluation:last_metrics", {"pass": True, "sharpe_ratio": 1.23})

    cached = db.get_cache("evaluation:last_metrics")
    assert cached == {"pass": True, "sharpe_ratio": 1.23}

    db.set_cache("evaluation:last_metrics", {"pass": False, "sharpe_ratio": -0.5})
    updated = db.get_cache("evaluation:last_metrics")
    assert updated == {"pass": False, "sharpe_ratio": -0.5}
