"""Tests for ExactMatchEvaluator."""
import pytest
from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput
from agentbench.evaluators.exact_match import ExactMatchEvaluator


@pytest.fixture
def evaluator():
    return ExactMatchEvaluator()


@pytest.mark.asyncio
async def test_exact_match_passes(evaluator):
    result = AgentResult(response="Paris")
    case = EvalCase(id="q1", input="Capital of France?",
                    expected=ExpectedOutput(response="Paris"))
    score = await evaluator.evaluate(result, case)
    assert score.passed
    assert score.score == 1.0


@pytest.mark.asyncio
async def test_exact_match_fails(evaluator):
    result = AgentResult(response="London")
    case = EvalCase(id="q1", input="Capital of France?",
                    expected=ExpectedOutput(response="Paris"))
    score = await evaluator.evaluate(result, case)
    assert not score.passed
    assert score.score == 0.0


@pytest.mark.asyncio
async def test_exact_match_case_insensitive(evaluator):
    result = AgentResult(response="paris")
    case = EvalCase(id="q1", input="Capital?",
                    expected=ExpectedOutput(response="Paris"))
    score = await evaluator.evaluate(result, case)
    assert score.passed


@pytest.mark.asyncio
async def test_no_expected_response_skips(evaluator):
    result = AgentResult(response="Anything")
    case = EvalCase(id="q1", input="Query", expected=ExpectedOutput())
    score = await evaluator.evaluate(result, case)
    assert score.passed
