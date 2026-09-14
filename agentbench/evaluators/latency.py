"""Latency evaluator — checks agent response time."""
from __future__ import annotations

from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


class LatencyEvaluator(BaseEvaluator):
    """
    Checks agent latency against a maximum threshold.

    The threshold can be set via:
    - ``case.expected.max_latency_ms``
    - evaluator config: ``{"max_latency_ms": 2000}``
    """

    name = "latency"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        max_ms = (
            case.expected.max_latency_ms
            or self.config.get("max_latency_ms")
        )

        if max_ms is None:
            # No threshold — just record latency
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason=f"Latency: {result.latency_ms:.1f}ms (no threshold set)",
                details={"latency_ms": result.latency_ms},
            )

        passed = result.latency_ms <= max_ms
        # Score: 1.0 if under threshold, degrades linearly above threshold
        if passed:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (result.latency_ms - max_ms) / max_ms)

        return EvaluatorScore(
            evaluator=self.name,
            score=score,
            passed=passed,
            reason=(
                f"Latency {result.latency_ms:.1f}ms ≤ {max_ms}ms"
                if passed
                else f"Latency {result.latency_ms:.1f}ms > {max_ms}ms threshold"
            ),
            details={"latency_ms": result.latency_ms, "max_latency_ms": max_ms},
        )
