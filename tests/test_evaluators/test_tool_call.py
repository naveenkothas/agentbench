"""Tests for ToolCallEvaluator."""
import pytest
from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput, ToolCall
from agentbench.evaluators.tool_call import ToolCallEvaluator


@pytest.fixture
def evaluator():
    return ToolCallEvaluator()


@pytest.mark.asyncio
async def test_correct_tool_call(evaluator):
    result = AgentResult(
        response="Checking order...",
        tool_calls=[ToolCall(name="get_order_status", arguments={"order_id": "123"})],
    )
    case = EvalCase(
        id="t1", input="Where is my order 123?",
        expected=ExpectedOutput(tool_call={"name": "get_order_status",
                                           "arguments": {"order_id": "123"}}),
    )
    score = await evaluator.evaluate(result, case)
    assert score.passed
    assert score.score == 1.0


@pytest.mark.asyncio
async def test_wrong_tool_called(evaluator):
    result = AgentResult(
        response="Checking weather...",
        tool_calls=[ToolCall(name="get_weather", arguments={})],
    )
    case = EvalCase(
        id="t1", input="What is my balance?",
        expected=ExpectedOutput(tool_call={"name": "get_account_balance"}),
    )
    score = await evaluator.evaluate(result, case)
    assert not score.passed


@pytest.mark.asyncio
async def test_no_tool_called_when_expected(evaluator):
    result = AgentResult(response="I do not know.")
    case = EvalCase(
        id="t1", input="Order status?",
        expected=ExpectedOutput(tool_call={"name": "get_order_status"}),
    )
    score = await evaluator.evaluate(result, case)
    assert not score.passed


@pytest.mark.asyncio
async def test_no_tool_expectation_passes(evaluator):
    result = AgentResult(response="Hello!")
    case = EvalCase(id="t1", input="Hi", expected=ExpectedOutput())
    score = await evaluator.evaluate(result, case)
    assert score.passed
