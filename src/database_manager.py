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
            conn.commit()

    def record_task_completion(
        self,
        task_name: str,
        run_type: str,
        mode: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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
                    json.dumps(metadata or {}, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def set_cache(self, key: str, value: Dict[str, Any]) -> None:
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
                    json.dumps(value, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cache_value_json FROM task_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
