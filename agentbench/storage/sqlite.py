"""SQLite storage backend for persisting evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from agentbench.core.models import EvalRun


class SQLiteStorage:
    """
    Async SQLite storage for AgentBench evaluation results.

    Creates a local .agentbench/runs.db database in the current directory
    (or a user-specified path) to persist run history.
    """

    DEFAULT_DB_PATH = Path(".agentbench") / "runs.db"

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH

    async def setup(self) -> None:
        """Create the database directory and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    suite_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    overall_score REAL,
                    total_cases INTEGER,
                    passed_cases INTEGER,
                    failed_cases INTEGER,
                    duration_ms REAL,
                    total_cost_usd REAL,
                    metrics_json TEXT,
                    run_data_json TEXT NOT NULL
                )
            """)
            await db.commit()

    async def save_run(self, run: EvalRun) -> None:
        """Persist an EvalRun to the database."""
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                (run_id, suite_name, status, started_at, completed_at,
                 overall_score, total_cases, passed_cases, failed_cases,
                 duration_ms, total_cost_usd, metrics_json, run_data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.suite_name,
                    run.status.value,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.overall_score,
                    run.total_cases,
                    run.passed_cases,
                    run.failed_cases,
                    run.duration_ms,
                    run.total_cost_usd,
                    json.dumps(run.metrics),
                    run.model_dump_json(),
                ),
            )
            await db.commit()

    async def get_run(self, run_id: str) -> EvalRun | None:
        """Retrieve a single run by ID."""
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT run_data_json FROM eval_runs WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return EvalRun.model_validate_json(row[0])
        return None

    async def list_runs(self, suite_name: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List recent runs with summary information."""
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:
            if suite_name:
                query = """
                    SELECT run_id, suite_name, status, started_at, overall_score,
                           total_cases, passed_cases, failed_cases, duration_ms, total_cost_usd
                    FROM eval_runs WHERE suite_name = ?
                    ORDER BY started_at DESC LIMIT ?
                """
                params = (suite_name, limit)
            else:
                query = """
                    SELECT run_id, suite_name, status, started_at, overall_score,
                           total_cases, passed_cases, failed_cases, duration_ms, total_cost_usd
                    FROM eval_runs
                    ORDER BY started_at DESC LIMIT ?
                """
                params = (limit,)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

    async def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID. Returns True if deleted, False if not found."""
        await self.setup()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM eval_runs WHERE run_id = ?", (run_id,))
            await db.commit()
            return cursor.rowcount > 0
