"""
Base class for all AgentBench evaluators.

To create a custom evaluator, subclass BaseEvaluator and implement `evaluate`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    Subclass this to implement a custom evaluation strategy.

    Example::

        class MyEvaluator(BaseEvaluator):
            name = "my_evaluator"

            async def evaluate(
                self, result: AgentResult, case: EvalCase
            ) -> EvaluatorScore:
                passed = "expected phrase" in result.response
                return EvaluatorScore(
                    evaluator=self.name,
                    score=1.0 if passed else 0.0,
                    passed=passed,
                    reason="Contains expected phrase" if passed else "Missing phrase",
                )
    """

    #: Unique name used to reference this evaluator in YAML configs
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        """
        Evaluate a single agent result against its expected output.

        Args:
            result: The raw output produced by the agent.
            case: The evaluation case including expected output and context.

        Returns:
            EvaluatorScore with a score in [0, 1], pass/fail, and reasoning.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
