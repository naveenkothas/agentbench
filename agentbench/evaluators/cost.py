"""Cost evaluator — checks estimated API cost."""
from __future__ import annotations

from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


class CostEvaluator(BaseEvaluator):
    """
    Checks that agent cost stays within budget.

    Threshold can be set via:
    - ``case.expected.max_cost_usd``
    - evaluator config: ``{"max_cost_usd": 0.01}``
    """

    name = "cost"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        max_cost = (
            case.expected.max_cost_usd
            or self.config.get("max_cost_usd")
        )

        if max_cost is None:
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason=f"Cost: ${result.estimated_cost_usd:.6f} (no threshold set)",
                details={"cost_usd": result.estimated_cost_usd},
            )

        passed = result.estimated_cost_usd <= max_cost
        if passed:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (result.estimated_cost_usd - max_cost) / max_cost)

        return EvaluatorScore(
            evaluator=self.name,
            score=score,
            passed=passed,
            reason=(
                f"Cost ${result.estimated_cost_usd:.6f} ≤ ${max_cost:.6f}"
                if passed
                else f"Cost ${result.estimated_cost_usd:.6f} exceeds ${max_cost:.6f}"
            ),
            details={"cost_usd": result.estimated_cost_usd, "max_cost_usd": max_cost},
        )
