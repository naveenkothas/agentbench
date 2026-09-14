"""
Registry for agent adapters.

Adapters register themselves here so they can be referenced by name
in YAML evaluation files.
"""
from __future__ import annotations

from typing import Any, Type

from agentbench.adapters.base import AgentAdapter

_REGISTRY: dict[str, Type[AgentAdapter]] = {}


class AdapterRegistry:
    """Central registry for all agent adapters."""

    @classmethod
    def register(cls, adapter_class: Type[AgentAdapter]) -> Type[AgentAdapter]:
        """Register an adapter class. Can be used as a decorator."""
        _REGISTRY[adapter_class.adapter_type] = adapter_class
        return adapter_class

    @classmethod
    def get(cls, adapter_type: str) -> Type[AgentAdapter]:
        if adapter_type not in _REGISTRY:
            available = ", ".join(_REGISTRY.keys())
            raise ValueError(
                f"Unknown adapter type: {adapter_type!r}. "
                f"Available adapters: {available}"
            )
        return _REGISTRY[adapter_type]

    @classmethod
    def create(cls, adapter_type: str, config: dict[str, Any] | None = None) -> AgentAdapter:
        """Instantiate an adapter by type name."""
        adapter_cls = cls.get(adapter_type)
        return adapter_cls(config=config)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(_REGISTRY.keys())


# Register built-in adapters
def _register_builtins() -> None:
    from agentbench.adapters.echo import EchoAdapter
    from agentbench.adapters.http import HTTPAdapter
    from agentbench.adapters.openai_adapter import OpenAIAdapter

    AdapterRegistry.register(EchoAdapter)
    AdapterRegistry.register(HTTPAdapter)
    AdapterRegistry.register(OpenAIAdapter)


_register_builtins()
