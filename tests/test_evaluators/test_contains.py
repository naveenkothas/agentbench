"""Tests for ContainsEvaluator."""
import pytest
from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput
from agentbench.evaluators.contains import ContainsEvaluator


@pytest.fixture
def evaluator():
    return ContainsEvaluator()


@pytest.mark.asyncio
async def test_contains_all_present(evaluator):
    result = AgentResult(response="Your refund will be processed in 3 days.")
    case = EvalCase(id="c1", input="Refund?",
                    expected=ExpectedOutput(contains=["refund", "processed"]))
    score = await evaluator.evaluate(result, case)
    assert score.passed
    assert score.score == 1.0


@pytest.mark.asyncio
async def test_contains_partial_match(evaluator):
    result = AgentResult(response="Your refund is pending.")
    case = EvalCase(id="c1", input="Refund?",
                    expected=ExpectedOutput(contains=["refund", "processed", "days"]))
    score = await evaluator.evaluate(result, case)
    assert not score.passed
    assert 0 < score.score < 1.0


@pytest.mark.asyncio
async def test_not_contains_forbidden_present(evaluator):
    result = AgentResult(response="Error: We cannot process your request.")
    case = EvalCase(id="c1", input="Help?",
                    expected=ExpectedOutput(not_contains=["error", "cannot"]))
    score = await evaluator.evaluate(result, case)
    assert not score.passed


@pytest.mark.asyncio
async def test_not_contains_passes_when_absent(evaluator):
    result = AgentResult(response="Sure, I will help you with that.")
    case = EvalCase(id="c1", input="Help?",
                    expected=ExpectedOutput(not_contains=["error", "cannot"]))
    score = await evaluator.evaluate(result, case)
    assert score.passed
