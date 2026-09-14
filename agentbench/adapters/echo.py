"""
EchoAdapter — a deterministic adapter for testing AgentBench itself.

Returns predictable responses without calling any external API.
Ideal for CI pipelines, demos, and unit tests.
"""
from __future__ import annotations

import time
from typing import Any

from agentbench.adapters.base import AgentAdapter
from agentbench.core.models import AgentResult


class EchoAdapter(AgentAdapter):
    """
    Simple adapter that echoes the input back as the response.

    Supports configurable response templates and simulated latency.
    Requires no API keys — works entirely offline.

    Config options::

        type: echo
        response_template: "Echo: {input}"
        latency_ms: 50
    """

    adapter_type = "echo"

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        start = time.monotonic()

        template = self.config.get("response_template", "Echo: {input}")
        response = template.format(input=input, **context)

        # Simulate configurable latency
        simulated_latency = self.config.get("latency_ms", 10) / 1000
        if simulated_latency > 0:
            import asyncio
            await asyncio.sleep(simulated_latency)

        elapsed_ms = (time.monotonic() - start) * 1000

        return AgentResult(
            response=response,
            latency_ms=elapsed_ms,
            prompt_tokens=len(input.split()),
            completion_tokens=len(response.split()),
            total_tokens=len(input.split()) + len(response.split()),
            estimated_cost_usd=0.0,
            metadata={"adapter": "echo"},
        )
