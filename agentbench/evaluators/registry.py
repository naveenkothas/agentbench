"""Registry for evaluators."""
from __future__ import annotations

from typing import Any, Type

from agentbench.core.base_evaluator import BaseEvaluator

_REGISTRY: dict[str, Type[BaseEvaluator]] = {}


class EvaluatorRegistry:
    @classmethod
    def register(cls, evaluator_class: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
        _REGISTRY[evaluator_class.name] = evaluator_class
        return evaluator_class

    @classmethod
    def get(cls, name: str) -> Type[BaseEvaluator]:
        if name not in _REGISTRY:
            available = ", ".join(_REGISTRY.keys())
            raise ValueError(
                f"Unknown evaluator: {name!r}. Available: {available}"
            )
        return _REGISTRY[name]

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseEvaluator:
        evaluator_cls = cls.get(name)
        return evaluator_cls(config=config)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(_REGISTRY.keys())


def _register_builtins() -> None:
    from agentbench.evaluators.exact_match import ExactMatchEvaluator
    from agentbench.evaluators.contains import ContainsEvaluator
    from agentbench.evaluators.tool_call import ToolCallEvaluator
    from agentbench.evaluators.latency import LatencyEvaluator
    from agentbench.evaluators.cost import CostEvaluator
    from agentbench.evaluators.json_schema import JSONSchemaEvaluator
    from agentbench.evaluators.hallucination import HallucinationEvaluator
    from agentbench.evaluators.llm_judge import LLMJudgeEvaluator

    for cls in [
        ExactMatchEvaluator, ContainsEvaluator, ToolCallEvaluator,
        LatencyEvaluator, CostEvaluator, JSONSchemaEvaluator,
        HallucinationEvaluator, LLMJudgeEvaluator,
    ]:
        EvaluatorRegistry.register(cls)


_register_builtins()
