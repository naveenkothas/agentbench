"""Base evaluator interface for AgentBench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


class BaseEvaluator(ABC):
    """
    Abstract base class for all AgentBench evaluators.

    To create a custom evaluator, subclass this and implement the
    `evaluate` method. Register it with the EvaluatorRegistry.

    Example::

        class MyEvaluator(BaseEvaluator):
            name = "my_evaluator"

            async def evaluate(
                self,
                result: AgentResult,
                case: EvalCase,
                context: dict[str, Any],
            ) -> EvaluatorScore:
                passed = "expected phrase" in result.response
                return EvaluatorScore(
                    evaluator_name=self.name,
                    score=1.0 if passed else 0.0,
                    passed=passed,
                    reason="Response contains expected phrase" if passed else "Missing expected phrase",
                )
    """

    name: str = "base"
    description: str = ""
    pass_threshold: float = 0.5

    @abstractmethod
    async def evaluate(
        self,
        result: AgentResult,
        case: EvalCase,
        context: dict[str, Any],
    ) -> EvaluatorScore:
        """
        Evaluate an agent result against a test case.

        Args:
            result: The AgentResult produced by the agent.
            case: The EvalCase containing the input, expected output, and assertions.
            context: Additional context, e.g., run-level configuration.

        Returns:
            An EvaluatorScore with the score (0.0–1.0), pass/fail, and optional reason.
        """
        ...

    def _make_score(
        self,
        score: float,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> EvaluatorScore:
        """Helper to build an EvaluatorScore."""
        threshold = threshold if threshold is not None else self.pass_threshold
        return EvaluatorScore(
            evaluator_name=self.name,
            score=score,
            passed=score >= threshold,
            reason=reason,
            details=details or {},
        )


class EvaluatorRegistry:
    """Registry for all available evaluators."""

    _registry: dict[str, type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
        """Register an evaluator class by its name."""
        cls._registry[evaluator_cls.name] = evaluator_cls
        return evaluator_cls

    @classmethod
    def get(cls, name: str) -> type[BaseEvaluator]:
        """Retrieve a registered evaluator class by name."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Evaluator '{name}' not found. Available evaluators: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered evaluator names."""
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseEvaluator:
        """Instantiate a registered evaluator by name."""
        evaluator_cls = cls.get(name)
        return evaluator_cls(**kwargs)


def register_evaluator(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
    """Decorator to register an evaluator with the global registry."""
    return EvaluatorRegistry.register(cls)
