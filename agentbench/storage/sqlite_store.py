"""
SQLite-backed storage for evaluation runs.

Uses SQLAlchemy async with aiosqlite for local persistence.
"""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from agentbench.core.models import EvalRun


class Base(DeclarativeBase):
    pass


class EvalRunRecord(Base):
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True)
    name = Column(String, default="")
    dataset_name = Column(String, default="")
    agent_type = Column(String, default="")
    status = Column(String, default="pending")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    total_cases = Column(Integer, default=0)
    passed_cases = Column(Integer, default=0)
    pass_rate = Column(Float, default=0.0)
    total_cost_usd = Column(Float, default=0.0)
    total_latency_ms = Column(Float, default=0.0)
    results_json = Column(Text, default="[]")
    metadata_json = Column(Text, default="{}")


class SQLiteStore:
    """Async SQLite store for EvalRun persistence."""

    def __init__(self, db_path: str = "./agentbench.db") -> None:
        db_url = f"sqlite+aiosqlite:///{db_path}"
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close all database connections held by this store."""
        await self.engine.dispose()

    async def __aenter__(self) -> SQLiteStore:
        await self.init()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def save_run(self, run: EvalRun) -> None:
        """Persist an EvalRun to the database."""
        async with self.async_session() as session:
            record = EvalRunRecord(
                id=run.id,
                name=run.name,
                dataset_name=run.dataset_name,
                agent_type=run.agent_type,
                status=run.status.value,
                started_at=run.started_at,
                completed_at=run.completed_at,
                total_cases=run.total_cases,
                passed_cases=run.passed_cases,
                pass_rate=run.pass_rate,
                total_cost_usd=run.total_cost_usd,
                total_latency_ms=run.total_latency_ms,
                results_json=json.dumps([r.model_dump() for r in run.results]),
                metadata_json=json.dumps(run.metadata),
            )
            session.add(record)
            await session.commit()

    async def get_run(self, run_id: str) -> EvalRunRecord | None:
        """Retrieve a single run by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(EvalRunRecord).where(EvalRunRecord.id == run_id)
            )
            return result.scalar_one_or_none()

    async def list_runs(self, limit: int = 50) -> list[EvalRunRecord]:
        """List recent evaluation runs."""
        async with self.async_session() as session:
            result = await session.execute(
                select(EvalRunRecord)
                .order_by(EvalRunRecord.started_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
