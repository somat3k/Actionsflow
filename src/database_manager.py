"""
SQLite-backed workflow database management with Redis write-through cache.

Architecture
------------
* **Redis layer** (via :class:`~src.redis_controller.RedisController`):
  in-memory, low-latency reads/writes with optional TTL.  Uses an embedded
  ``fakeredis`` instance by default so no external Redis daemon is needed.
  Connect to an external server by setting the ``REDIS_URL`` env variable.
* **SQLite layer**: durable persistence for all cache entries, task-completion
  history, and dataset metadata.

Cache read path: Redis → SQLite warm-up on miss → return value
Cache write path: SQLite (always) + Redis (when available)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.redis_controller import RedisController

log = logging.getLogger(__name__)


class DatabaseManager:
    """Manage durable workflow task state backed by SQLite with Redis caching.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    redis_url:
        Optional Redis URL (e.g. ``"redis://localhost:6379/0"``).  When
        ``None`` the value of the ``REDIS_URL`` environment variable is used.
        If neither is set an embedded ``fakeredis`` instance is used.
    cache_ttl:
        Default TTL in seconds for Redis cache entries (default 3600 s).
        Pass ``None`` to disable TTL.
    """

    def __init__(
        self,
        db_path: Path,
        redis_url: Optional[str] = None,
        cache_ttl: Optional[int] = 3600,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._redis = RedisController(
            namespace="qt",
            url=redis_url,
            default_ttl=cache_ttl,
        )
        log.debug(
            "DatabaseManager: Redis backend=%s embedded=%s",
            "available" if self._redis.is_available else "disabled",
            self._redis.is_embedded,
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_completions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ttl_seconds INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    rows INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            # Migrate existing task_cache table to add ttl_seconds column if absent
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(task_cache)").fetchall()
            }
            if "ttl_seconds" not in cols:
                conn.execute("ALTER TABLE task_cache ADD COLUMN ttl_seconds INTEGER")
            conn.commit()

    def record_task_completion(
        self,
        task_name: str,
        run_type: str,
        mode: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a completed workflow task event with optional JSON metadata."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_completions
                (task_name, run_type, mode, status, metadata_json, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_name,
                    run_type,
                    mode,
                    status,
                    json.dumps(metadata or {}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def set_cache(self, key: str, value: Any, ttl_seconds: Optional[int] = -1) -> None:
        """Upsert a JSON-serializable cache value for *key*.

        The value is written to both SQLite (durable) and Redis (fast reads).

        Parameters
        ----------
        ttl_seconds:
            ``-1`` (default) means use the RedisController's ``default_ttl``.
            ``None`` stores with no expiry.  A positive integer sets an
            explicit TTL in Redis (SQLite entries never expire).
        """
        serialized = json.dumps(value)
        now = datetime.now(timezone.utc).isoformat()
        ttl_to_store = self._redis.default_ttl if ttl_seconds == -1 else ttl_seconds
        # ── SQLite (durable) ──────────────────────────────────────────────────
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_cache (cache_key, cache_value_json, updated_at, ttl_seconds)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET
                    cache_value_json=excluded.cache_value_json,
                    updated_at=excluded.updated_at,
                    ttl_seconds=excluded.ttl_seconds
                """,
                (key, serialized, now, ttl_to_store),
            )
            conn.commit()
        # ── Redis (fast layer) ────────────────────────────────────────────────
        self._redis.set(key, serialized, ttl_seconds=ttl_seconds)

    def get_cache(self, key: str) -> Optional[Any]:
        """Return cached value for *key*, or ``None`` when absent.

        Read order: Redis → SQLite (warm Redis on miss).
        """
        # ── 1. Try Redis first ────────────────────────────────────────────────
        raw = self._redis.get(key)
        if raw is not None:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass

        # ── 2. Fall back to SQLite and warm Redis ─────────────────────────────
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cache_value_json, ttl_seconds FROM task_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        value_json, stored_ttl = row[0], row[1]
        # Warm Redis so the next read is fast.
        # Use stored_ttl when available; -1 tells RedisController to use its default_ttl.
        redis_ttl = stored_ttl if stored_ttl is not None else -1
        self._redis.set(key, value_json, ttl_seconds=redis_ttl)
        try:
            return json.loads(value_json)
        except (json.JSONDecodeError, ValueError):
            return None

    def delete_cache(self, key: str) -> None:
        """Remove a cache entry from both Redis and SQLite."""
        self._redis.delete(key)
        with self._connect() as conn:
            conn.execute("DELETE FROM task_cache WHERE cache_key = ?", (key,))
            conn.commit()

    def cache_keys(self, pattern: str = "*") -> List[str]:
        """Return cache keys matching *pattern* (glob, e.g. ``"training:*"``)."""
        # Prefer Redis for fast enumeration; fall back to SQLite
        if self._redis.is_available:
            return self._redis.keys(pattern)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT cache_key FROM task_cache WHERE cache_key LIKE ?",
                (pattern.replace("*", "%").replace("?", "_"),),
            ).fetchall()
        return [r[0] for r in rows]

    def record_dataset(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int],
        end_ms: Optional[int],
        rows: int,
        path: str,
    ) -> None:
        """Record a stored dataset artifact."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets
                (symbol, interval, start_ms, end_ms, rows, path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.upper(),
                    interval,
                    start_ms,
                    end_ms,
                    rows,
                    path,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_latest_dataset(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """Return the most recent dataset metadata for a symbol/interval."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT symbol, interval, start_ms, end_ms, rows, path, created_at
                FROM datasets
                WHERE symbol = ? AND interval = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (symbol.upper(), interval),
            ).fetchone()
        if row is None:
            return None
        return {
            "symbol": row[0],
            "interval": row[1],
            "start_ms": row[2],
            "end_ms": row[3],
            "rows": row[4],
            "path": row[5],
            "created_at": row[6],
        }

    # ── Convenience access ────────────────────────────────────────────────────

    @property
    def redis(self) -> RedisController:
        """Return the underlying :class:`~src.redis_controller.RedisController`."""
        return self._redis

    def close(self) -> None:
        """Release the Redis connection (embedded instances are a no-op)."""
        self._redis.close()
