"""
QuantGuard-MAS
LLM Usage and Performance Tracker (SQLite)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.config import Config


class UsageTracker:

    def __init__(self, database_path=None):
        self.database_path = Path(database_path or Config.SQLITE_DATABASE_PATH)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.database_path)

    def _initialize_database(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    key_number INTEGER,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0,
                    finish_reason TEXT,
                    request_id TEXT,
                    attempts INTEGER DEFAULT 1,
                    status TEXT NOT NULL,
                    error_type TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp
                ON llm_usage(timestamp)
                """
            )

    def record(
        self,
        provider,
        model,
        key_number,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        latency_ms=0,
        finish_reason=None,
        request_id=None,
        attempts=1,
        status="success",
        error_type=None,
    ):
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_usage (
                    timestamp, provider, model, key_number,
                    input_tokens, output_tokens, total_tokens,
                    latency_ms, finish_reason, request_id,
                    attempts, status, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    provider,
                    model,
                    key_number,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    latency_ms,
                    finish_reason,
                    request_id,
                    attempts,
                    status,
                    error_type,
                ),
            )

    def get_summary(self):
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    provider,
                    model,
                    COUNT(*) AS requests,
                    SUM(input_tokens) AS input_tokens,
                    SUM(output_tokens) AS output_tokens,
                    SUM(total_tokens) AS total_tokens,
                    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_requests,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_requests
                FROM llm_usage
                GROUP BY provider, model
                ORDER BY requests DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_recent_logs(self, limit=25):
        """Return the most recent per-request telemetry rows for dashboard/API audit."""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM llm_usage ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]
