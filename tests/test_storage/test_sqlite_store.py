"""Tests for SQLiteStore."""
import json

from agentbench.core.models import AgentResult, EvalResult, EvalRun, EvaluatorScore, RunStatus
from agentbench.storage.sqlite_store import SQLiteStore


def _run(passed: bool, name: str = "") -> EvalRun:
    return EvalRun(
        name=name,
        dataset_name="suite",
        agent_type="echo",
        status=RunStatus.COMPLETED,
        results=[
            EvalResult(
                case_id="c1",
                passed=passed,
                scores=[EvaluatorScore(evaluator="contains", score=1.0 if passed else 0.0,
                                       passed=passed)],
                agent_result=AgentResult(response="hi", latency_ms=12.0),
            )
        ],
    )


async def test_save_and_get_run(tmp_path):
    async with SQLiteStore(str(tmp_path / "runs.db")) as store:
        run = _run(passed=True)
        await store.save_run(run)

        record = await store.get_run(run.id)
        assert record is not None
        assert record.dataset_name == "suite"
        assert record.status == "completed"
        assert record.total_cases == 1
        assert record.pass_rate == 1.0
        assert record.total_latency_ms == 12.0
        assert json.loads(record.results_json)[0]["case_id"] == "c1"


async def test_get_missing_run_returns_none(tmp_path):
    async with SQLiteStore(str(tmp_path / "runs.db")) as store:
        assert await store.get_run("missing") is None


async def test_list_runs_respects_limit(tmp_path):
    async with SQLiteStore(str(tmp_path / "runs.db")) as store:
        for i in range(3):
            await store.save_run(_run(passed=bool(i % 2), name=f"run-{i}"))

        assert len(await store.list_runs()) == 3
        assert len(await store.list_runs(limit=2)) == 2
