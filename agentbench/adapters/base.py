"""
Abstract base class for agent adapters.

Every adapter converts AgentBench's generic interface into calls against
a specific agent framework (OpenAI, LangGraph, custom Python, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentbench.core.models import AgentResult


class AgentAdapter(ABC):
    """
    Abstract adapter that wraps an agent implementation.

    Subclass this to connect any agent to AgentBench.

    Example::

        class MyAgentAdapter(AgentAdapter):
            adapter_type = "my_agent"

            async def run(self, input: str, context: dict) -> AgentResult:
                response = await my_agent.chat(input)
                return AgentResult(response=response)
    """

    #: Unique identifier used in YAML ``agent.type``
    adapter_type: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        """
        Execute the agent on the given input and return an AgentResult.

        Args:
            input: The user query or task.
            context: Additional context (documents, history, metadata).

        Returns:
            AgentResult containing the response, tool calls, tokens, cost, etc.
        """
        ...

    async def setup(self) -> None:
        """Optional async setup called once before a run begins."""

    async def teardown(self) -> None:
        """Optional async teardown called once after a run completes."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.adapter_type!r})"
