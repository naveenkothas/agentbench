"""Integration tests: engine + adapters + evaluators end-to-end."""

import pytest

from agentbench.adapters.builtin import CustomAdapter, EchoAdapter
from agentbench.core.models import (
    AgentConfig,
    AgentResult,
    Assertion,
    EvalCase,
    EvalSuiteConfig,
    ToolCall,
    ToolCallAssertion,
)
from agentbench.runners.engine import EvaluationEngine


def make_echo_suite(cases: list[EvalCase], response: str = "test response") -> EvalSuiteConfig:
    return EvalSuiteConfig(
        name="Test Suite",
        agent=AgentConfig(type="echo", extra={"response": response}),
        cases=cases,
    )


class TestEvaluationEngine:
    @pytest.fixture
    def engine(self):
        return EvaluationEngine(max_concurrency=2)

    @pytest.mark.asyncio
    async def test_runs_simple_suite(self, engine):
        cases = [
            EvalCase(
                id="c1",
                input="hello",
                expected=Assertion(contains=["test"]),
            )
        ]
        suite = make_echo_suite(cases, response="test response here")
        run = await engine.run(suite)
        assert run.total_cases == 1
        assert run.passed_cases == 1

    @pytest.mark.asyncio
    async def test_detects_failures(self, engine):
        cases = [
            EvalCase(
                id="fail-001",
                input="hello",
                expected=Assertion(contains=["MISSING_WORD"]),
            )
        ]
        suite = make_echo_suite(cases, response="something else entirely")
        run = await engine.run(suite)
        assert run.failed_cases == 1
        assert run.passed_cases == 0

    @pytest.mark.asyncio
    async def test_overall_score_is_averaged(self, engine):
        cases = [
            EvalCase(id="c1", input="a", expected=Assertion(contains=["test"])),
            EvalCase(id="c2", input="b", expected=Assertion(contains=["MISSING"])),
        ]
        suite = make_echo_suite(cases, response="test response")
        run = await engine.run(suite)
        assert run.total_cases == 2
        assert run.passed_cases == 1
        assert run.failed_cases == 1

    @pytest.mark.asyncio
    async def test_runs_multiple_cases_concurrently(self, engine):
        cases = [
            EvalCase(
                id=f"c{i}",
                input=f"input {i}",
                expected=Assertion(contains=["response"]),
            )
            for i in range(10)
        ]
        suite = make_echo_suite(cases, response="good response")
        run = await engine.run(suite)
        assert run.total_cases == 10
        assert run.passed_cases == 10

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, engine):
        calls = []

        def cb(result, done, total):
            calls.append((done, total))

        engine.progress_callback = cb
        cases = [
            EvalCase(id="c1", input="x", expected=Assertion()),
            EvalCase(id="c2", input="y", expected=Assertion()),
        ]
        suite = make_echo_suite(cases)
        await engine.run(suite)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_custom_adapter_integration(self, engine):
        async def my_agent(input: str, context: dict) -> AgentResult:
            return AgentResult(
                response=f"processed: {input}",
                tool_calls=[ToolCall(name="search", arguments={"q": input})],
            )

        config = AgentConfig(type="custom")
        adapter = CustomAdapter(config)
        adapter.set_callable(my_agent)

        cases = [
            EvalCase(
                id="custom-001",
                input="query text",
                expected=Assertion(
                    contains=["processed"],
                    tool_call=ToolCallAssertion(name="search"),
                ),
            )
        ]
        suite = EvalSuiteConfig(
            name="Custom Agent Test",
            agent=AgentConfig(type="custom"),
            cases=cases,
        )

        run = await engine.run(suite, adapter=adapter)
        assert run.passed_cases == 1

    @pytest.mark.asyncio
    async def test_agent_error_is_captured(self, engine):
        async def failing_agent(input: str, context: dict) -> AgentResult:
            raise RuntimeError("Agent is down!")

        config = AgentConfig(type="custom")
        adapter = CustomAdapter(config)
        adapter.set_callable(failing_agent)

        cases = [EvalCase(id="err-001", input="test", expected=Assertion(contains=["response"]))]
        suite = EvalSuiteConfig(
            name="Error Test",
            agent=AgentConfig(type="custom"),
            cases=cases,
        )
        run = await engine.run(suite, adapter=adapter)
        # Should record the failure without crashing
        assert run.total_cases == 1
        assert run.failed_cases == 1

    @pytest.mark.asyncio
    async def test_metrics_are_computed(self, engine):
        cases = [
            EvalCase(id="c1", input="a", expected=Assertion(contains=["test"])),
            EvalCase(id="c2", input="b", expected=Assertion(contains=["test"])),
        ]
        suite = make_echo_suite(cases, response="test response")
        run = await engine.run(suite)
        assert "contains" in run.metrics
        assert run.metrics["contains"] == 1.0

    @pytest.mark.asyncio
    async def test_explicit_evaluators_list(self, engine):
        cases = [
            EvalCase(
                id="c1",
                input="hello",
                expected=Assertion(contains=["test"], max_latency_ms=5000.0),
            )
        ]
        suite = EvalSuiteConfig(
            name="Explicit Evaluators",
            agent=AgentConfig(type="echo", extra={"response": "test response"}),
            cases=cases,
            evaluators=["contains", "latency"],
        )
        run = await engine.run(suite)
        evaluator_names = {s.evaluator_name for r in run.case_results for s in r.evaluator_scores}
        assert "contains" in evaluator_names
        assert "latency" in evaluator_names
