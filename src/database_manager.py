"""
SQLite-backed workflow database management.
Provides lightweight task completion tracking and cached key/value storage.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class DatabaseManager:
    """Manage durable workflow task state in a local SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                    updated_at TEXT NOT NULL
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

    def set_cache(self, key: str, value: Any) -> None:
        """Upsert a JSON-serializable cache value for a key."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_cache (cache_key, cache_value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET
                    cache_value_json=excluded.cache_value_json,
                    updated_at=excluded.updated_at
                """,
                (
                    key,
                    json.dumps(value),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_cache(self, key: str) -> Optional[Any]:
        """Return cached value for a key, or None when the key is not present."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cache_value_json FROM task_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

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
