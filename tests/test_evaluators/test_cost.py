"""Tests for CostEvaluator."""
import pytest

from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput
from agentbench.evaluators.cost import CostEvaluator


def _case(max_cost=None):
    return EvalCase(id="c1", input="q", expected=ExpectedOutput(max_cost_usd=max_cost))


async def test_within_budget_passes():
    score = await CostEvaluator().evaluate(AgentResult(response="ok", estimated_cost_usd=0.001),
                                           _case(0.01))
    assert score.passed
    assert score.score == 1.0


async def test_over_budget_fails_with_partial_score():
    score = await CostEvaluator().evaluate(AgentResult(response="ok", estimated_cost_usd=0.015),
                                           _case(0.01))
    assert not score.passed
    assert score.score == pytest.approx(0.5)


async def test_far_over_budget_scores_zero():
    score = await CostEvaluator().evaluate(AgentResult(response="ok", estimated_cost_usd=1.0),
                                           _case(0.01))
    assert not score.passed
    assert score.score == 0.0


async def test_no_threshold_passes():
    score = await CostEvaluator().evaluate(AgentResult(response="ok", estimated_cost_usd=5.0),
                                           _case())
    assert score.passed


async def test_threshold_from_config():
    evaluator = CostEvaluator(config={"max_cost_usd": 0.001})
    score = await evaluator.evaluate(AgentResult(response="ok", estimated_cost_usd=0.002), _case())
    assert not score.passed
