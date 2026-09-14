"""Contains / not-contains evaluator."""
from __future__ import annotations

from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


class ContainsEvaluator(BaseEvaluator):
    """
    Checks that the response contains all required strings
    and none of the forbidden strings.

    Configured via EvalCase.expected.contains and .not_contains.
    """

    name = "contains"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        required = case.expected.contains
        forbidden = case.expected.not_contains

        if not required and not forbidden:
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason="No contains/not_contains constraints defined.",
            )

        response_lower = result.response.lower()
        failures: list[str] = []
        hits = 0
        total = len(required) + len(forbidden)

        for phrase in required:
            if phrase.lower() in response_lower:
                hits += 1
            else:
                failures.append(f"Missing required: {phrase!r}")

        for phrase in forbidden:
            if phrase.lower() not in response_lower:
                hits += 1
            else:
                failures.append(f"Found forbidden: {phrase!r}")

        score = hits / total if total > 0 else 1.0
        passed = len(failures) == 0

        return EvaluatorScore(
            evaluator=self.name,
            score=score,
            passed=passed,
            reason="; ".join(failures) if failures else "All contains/not_contains checks passed.",
        )
