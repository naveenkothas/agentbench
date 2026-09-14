"""Shared pytest fixtures for AgentBench tests."""
import pytest
from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput


@pytest.fixture
def basic_result():
    return AgentResult(response="The refund will be processed within 5 business days.")


@pytest.fixture
def basic_case():
    return EvalCase(
        id="test-001",
        input="How do I get a refund?",
        expected=ExpectedOutput(
            contains=["refund"],
            not_contains=["error"],
        ),
    )


@pytest.fixture
def tool_result():
    from agentbench.core.models import ToolCall
    return AgentResult(
        response="Let me check your order status.",
        tool_calls=[
            ToolCall(name="get_order_status", arguments={"order_id": "12345"})
        ],
        latency_ms=150.0,
        estimated_cost_usd=0.0001,
    )
