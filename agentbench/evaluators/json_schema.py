"""JSON Schema evaluator — validates structured agent output."""
from __future__ import annotations

import json

from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


class JSONSchemaEvaluator(BaseEvaluator):
    """
    Validates that the agent response is valid JSON conforming to a schema.

    Requires ``jsonschema`` library if schema validation is needed.
    Falls back to JSON parse-only check if jsonschema is not installed.
    """

    name = "json_schema"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        schema = case.expected.json_schema or self.config.get("schema")

        # Try to parse as JSON
        try:
            parsed = json.loads(result.response)
        except json.JSONDecodeError as e:
            return EvaluatorScore(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason=f"Response is not valid JSON: {e}",
            )

        if schema is None:
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason="Response is valid JSON (no schema to validate against).",
            )

        try:
            import jsonschema
            jsonschema.validate(instance=parsed, schema=schema)
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason="Response is valid JSON and conforms to schema.",
            )
        except ImportError:
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason="Response is valid JSON (jsonschema not installed — schema not validated).",
            )
        except Exception as e:
            return EvaluatorScore(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason=f"JSON schema validation failed: {e}",
            )
