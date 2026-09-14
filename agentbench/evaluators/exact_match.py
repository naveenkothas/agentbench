"""Exact match evaluator — deterministic string equality check."""
from __future__ import annotations

from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


class ExactMatchEvaluator(BaseEvaluator):
    """
    Passes when the agent response exactly matches the expected response.

    Case-sensitivity is configurable via ``case_sensitive`` config key.
    Useful for deterministic factual questions (capitals, dates, etc.).
    """

    name = "exact_match"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        expected = case.expected.response
        if expected is None:
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason="No expected response defined — skipping exact match.",
            )

        case_sensitive = self.config.get("case_sensitive", False)
        actual = result.response
        if not case_sensitive:
            actual = actual.strip().lower()
            expected = expected.strip().lower()
        else:
            actual = actual.strip()
            expected = expected.strip()

        passed = actual == expected
        return EvaluatorScore(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="Exact match" if passed else f"Expected: {expected!r} | Got: {result.response.strip()!r}",
        )
