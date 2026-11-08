"""Tests for LatencyEvaluator."""
import pytest

from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput
from agentbench.evaluators.latency import LatencyEvaluator


@pytest.fixture
def evaluator():
    return LatencyEvaluator()


@pytest.mark.asyncio
async def test_within_threshold(evaluator):
    result = AgentResult(response="OK", latency_ms=500.0)
    case = EvalCase(id="l1", input="Hello",
                    expected=ExpectedOutput(max_latency_ms=1000.0))
    score = await evaluator.evaluate(result, case)
    assert score.passed
    assert score.score == 1.0


@pytest.mark.asyncio
async def test_exceeds_threshold(evaluator):
    result = AgentResult(response="OK", latency_ms=2000.0)
    case = EvalCase(id="l1", input="Hello",
                    expected=ExpectedOutput(max_latency_ms=1000.0))
    score = await evaluator.evaluate(result, case)
    assert not score.passed


@pytest.mark.asyncio
async def test_no_threshold_always_passes(evaluator):
    result = AgentResult(response="OK", latency_ms=99999.0)
    case = EvalCase(id="l1", input="Hello", expected=ExpectedOutput())
    score = await evaluator.evaluate(result, case)
    assert score.passed
