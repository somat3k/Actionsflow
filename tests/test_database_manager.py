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


# ── Redis-integration tests ────────────────────────────────────────────────────

def test_redis_backend_is_embedded(tmp_path):
    """DatabaseManager should use embedded fakeredis by default."""
    db = DatabaseManager(tmp_path / "state.db")
    assert db.redis.is_available
    assert db.redis.is_embedded


def test_cache_write_through_to_redis(tmp_path):
    """Values written via set_cache must be readable from Redis directly."""
    db = DatabaseManager(tmp_path / "state.db")
    import json

    db.set_cache("wt-key", {"score": 99})
    raw = db.redis.get("wt-key")
    assert raw is not None
    assert json.loads(raw) == {"score": 99}


def test_cache_redis_miss_falls_back_to_sqlite(tmp_path):
    """After flushing Redis, get_cache should re-read from SQLite."""
    db = DatabaseManager(tmp_path / "state.db")

    db.set_cache("fallback-key", {"x": 42})
    # Evict from Redis
    db.redis.flush()
    assert db.redis.get("fallback-key") is None

    # Should still return value from SQLite and re-warm Redis
    result = db.get_cache("fallback-key")
    assert result == {"x": 42}

    # Redis should now be warm again
    assert db.redis.get("fallback-key") is not None


def test_cache_with_explicit_ttl(tmp_path):
    """TTL passed to set_cache should be stored in SQLite and applied in Redis."""
    db = DatabaseManager(tmp_path / "state.db")
    db.set_cache("ttl-entry", {"v": 1}, ttl_seconds=300)

    # Redis should have a TTL applied
    remaining = db.redis.ttl("ttl-entry")
    assert remaining is not None and remaining > 0

    # SQLite should store the TTL
    with sqlite3.connect(tmp_path / "state.db") as conn:
        row = conn.execute(
            "SELECT ttl_seconds FROM task_cache WHERE cache_key = ?", ("ttl-entry",)
        ).fetchone()
    assert row is not None and row[0] == 300


def test_delete_cache(tmp_path):
    """delete_cache should remove the entry from both Redis and SQLite."""
    db = DatabaseManager(tmp_path / "state.db")
    db.set_cache("del-key", {"bye": True})
    assert db.get_cache("del-key") is not None

    db.delete_cache("del-key")

    # SQLite gone
    with sqlite3.connect(tmp_path / "state.db") as conn:
        row = conn.execute(
            "SELECT 1 FROM task_cache WHERE cache_key = ?", ("del-key",)
        ).fetchone()
    assert row is None

    # Redis gone
    assert db.redis.get("del-key") is None
    # get_cache should return None
    assert db.get_cache("del-key") is None


def test_cache_keys_returns_matching_keys(tmp_path):
    """cache_keys should enumerate keys stored in Redis."""
    db = DatabaseManager(tmp_path / "state.db")
    db.redis.flush()
    db.set_cache("training:btc", {"ts": 1})
    db.set_cache("training:eth", {"ts": 2})
    db.set_cache("evaluation:metrics", {"pass": True})

    training_keys = sorted(db.cache_keys("training:*"))
    assert training_keys == ["training:btc", "training:eth"]


def test_close_releases_redis(tmp_path):
    """close() should shut down the Redis controller."""
    db = DatabaseManager(tmp_path / "state.db")
    assert db.redis.is_available
    db.close()
    assert not db.redis.is_available


def test_cache_disabled_redis_not_used(tmp_path):
    """When cache_enabled=False, Redis must not be initialised and SQLite still works."""
    db = DatabaseManager(tmp_path / "state.db", cache_enabled=False)
    assert not db.redis.is_available

    # set_cache/get_cache must still work via SQLite alone
    db.set_cache("disabled-key", {"ok": True})
    assert db.get_cache("disabled-key") == {"ok": True}


def test_custom_namespace_is_forwarded(tmp_path):
    """namespace passed to DatabaseManager must be used by the Redis controller."""
    db = DatabaseManager(tmp_path / "state.db", namespace="myns")
    assert db.redis.namespace == "myns"
    db.set_cache("ns-key", {"v": 1})
    # Verify the key is stored under the custom namespace in Redis
    assert db.redis.exists("ns-key")
