"""Tests for HallucinationEvaluator."""
import pytest
from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput
from agentbench.evaluators.hallucination import HallucinationEvaluator, _simple_overlap_score


def test_overlap_score_high():
    response = "The return policy allows returns within 30 days with a receipt."
    context = "Our return policy allows returns within 30 days of purchase with a valid receipt."
    score = _simple_overlap_score(response, context)
    assert score > 0.5


def test_overlap_score_low():
    response = "We offer free unlimited returns forever with no questions asked."
    context = "Returns are accepted within 14 days only with original packaging."
    score = _simple_overlap_score(response, context)
    assert score < 0.8


@pytest.mark.asyncio
async def test_no_context_skips():
    evaluator = HallucinationEvaluator()
    result = AgentResult(response="Anything")
    case = EvalCase(id="h1", input="Tell me about returns", expected=ExpectedOutput())
    score = await evaluator.evaluate(result, case)
    assert score.passed


@pytest.mark.asyncio
async def test_grounded_response_passes():
    evaluator = HallucinationEvaluator(config={"threshold": 0.2})
    result = AgentResult(response="The return policy allows returns within 30 days.")
    case = EvalCase(
        id="h1",
        input="What is the return policy?",
        expected=ExpectedOutput(),
        context={"documents": ["Our return policy allows returns within 30 days of purchase."]},
    )
    score = await evaluator.evaluate(result, case)
    assert score.passed
