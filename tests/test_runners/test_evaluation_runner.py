"""Tests for EvaluationRunner — the end-to-end evaluation pipeline."""
from agentbench.adapters.base import AgentAdapter
from agentbench.adapters.echo import EchoAdapter
from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import (
    AgentResult,
    EvalCase,
    EvalDataset,
    EvaluatorScore,
    ExpectedOutput,
    RunStatus,
)
from agentbench.evaluators.contains import ContainsEvaluator


class FixedAdapter(AgentAdapter):
    adapter_type = "fixed"

    def __init__(self, response: str) -> None:
        super().__init__()
        self.response = response
        self.setup_called = False
        self.teardown_called = False

    async def setup(self) -> None:
        self.setup_called = True

    async def teardown(self) -> None:
        self.teardown_called = True

    async def run(self, input, context):
        return AgentResult(response=self.response, latency_ms=5.0, estimated_cost_usd=0.001)


class FailingAdapter(AgentAdapter):
    adapter_type = "failing"

    async def run(self, input, context):
        raise RuntimeError("agent exploded")


class BrokenEvaluator(BaseEvaluator):
    name = "broken"

    async def evaluate(self, result, case) -> EvaluatorScore:
        raise ValueError("evaluator bug")


def _dataset(*cases, **kwargs):
    return EvalDataset(name="test-suite", cases=list(cases), **kwargs)


def _case(case_id, contains):
    return EvalCase(id=case_id, input=f"question {case_id}",
                    expected=ExpectedOutput(contains=contains))


async def test_run_from_yaml_config_uses_echo_adapter():
    from agentbench.runners.evaluation_runner import EvaluationRunner

    dataset = _dataset(
        _case("a", ["question"]),
        agent={"type": "echo", "latency_ms": 0},
        evaluators=["contains"],
    )
    run = await EvaluationRunner(dataset).run()

    assert run.status == RunStatus.COMPLETED
    assert run.completed_at is not None
    assert run.agent_type == "echo"
    assert run.dataset_name == "test-suite"
    assert run.total_cases == 1
    assert run.pass_rate == 1.0


async def test_mixed_pass_and_fail_and_aggregates():
    from agentbench.runners.evaluation_runner import EvaluationRunner

    adapter = FixedAdapter("Your refund is on the way.")
    dataset = _dataset(_case("pass", ["refund"]), _case("fail", ["invoice"]))
    run = await EvaluationRunner(dataset, adapter=adapter,
                                 evaluators=[ContainsEvaluator()]).run()

    results = {r.case_id: r for r in run.results}
    assert results["pass"].passed
    assert not results["fail"].passed
    assert run.pass_rate == 0.5
    assert run.total_latency_ms == 10.0
    assert abs(run.total_cost_usd - 0.002) < 1e-9
    assert adapter.setup_called and adapter.teardown_called


async def test_agent_exception_marks_case_failed():
    from agentbench.runners.evaluation_runner import EvaluationRunner

    run = await EvaluationRunner(_dataset(_case("x", ["a"])), adapter=FailingAdapter(),
                                 evaluators=[ContainsEvaluator()]).run()

    assert run.status == RunStatus.COMPLETED
    assert not run.results[0].passed
    assert "agent exploded" in run.results[0].error


async def test_evaluator_exception_does_not_crash_run():
    from agentbench.runners.evaluation_runner import EvaluationRunner

    adapter = FixedAdapter("refund")
    run = await EvaluationRunner(_dataset(_case("x", ["refund"])), adapter=adapter,
                                 evaluators=[BrokenEvaluator(), ContainsEvaluator()]).run()

    result = run.results[0]
    assert result.passed
    assert [s.evaluator for s in result.scores] == ["contains"]


async def test_progress_callback_called_for_every_case():
    from agentbench.runners.evaluation_runner import EvaluationRunner

    calls = []
    dataset = _dataset(*[_case(str(i), ["Echo"]) for i in range(4)])
    runner = EvaluationRunner(
        dataset,
        adapter=EchoAdapter({"latency_ms": 0}),
        evaluators=[ContainsEvaluator()],
        max_concurrency=2,
        progress_callback=lambda result, done, total: calls.append((done, total)),
    )
    run = await runner.run()

    assert run.total_cases == 4
    assert sorted(done for done, _ in calls) == [1, 2, 3, 4]
    assert all(total == 4 for _, total in calls)
