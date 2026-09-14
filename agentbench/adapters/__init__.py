"""Agent adapters — bridge between AgentBench and specific agent frameworks."""
from agentbench.adapters.base import AgentAdapter
from agentbench.adapters.echo import EchoAdapter
from agentbench.adapters.registry import AdapterRegistry

__all__ = ["AgentAdapter", "EchoAdapter", "AdapterRegistry"]
