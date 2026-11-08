"""Tests for JSONSchemaEvaluator."""
from agentbench.core.models import AgentResult, EvalCase, ExpectedOutput
from agentbench.evaluators.json_schema import JSONSchemaEvaluator

SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "total": {"type": "number"}},
    "required": ["order_id", "total"],
}


def _case(schema=None):
    return EvalCase(id="j1", input="q", expected=ExpectedOutput(json_schema=schema))


async def test_valid_json_matching_schema_passes():
    result = AgentResult(response='{"order_id": "A1", "total": 9.5}')
    score = await JSONSchemaEvaluator().evaluate(result, _case(SCHEMA))
    assert score.passed
    assert "conforms to schema" in score.reason


async def test_schema_violation_fails():
    result = AgentResult(response='{"order_id": 123}')
    score = await JSONSchemaEvaluator().evaluate(result, _case(SCHEMA))
    assert not score.passed
    assert score.score == 0.0


async def test_invalid_json_fails():
    score = await JSONSchemaEvaluator().evaluate(AgentResult(response="not json"), _case(SCHEMA))
    assert not score.passed
    assert "not valid JSON" in score.reason


async def test_valid_json_without_schema_passes():
    score = await JSONSchemaEvaluator().evaluate(AgentResult(response="[1, 2]"), _case())
    assert score.passed


async def test_schema_from_config():
    evaluator = JSONSchemaEvaluator(config={"schema": SCHEMA})
    score = await evaluator.evaluate(AgentResult(response="{}"), _case())
    assert not score.passed
