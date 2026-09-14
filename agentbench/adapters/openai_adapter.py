"""
OpenAI adapter — evaluates agents built with the OpenAI Python SDK.
Requires: pip install agentbench[openai]
"""
from __future__ import annotations

import time
from typing import Any

from agentbench.adapters.base import AgentAdapter
from agentbench.core.models import AgentResult, ToolCall

# Cost per 1K tokens (approximate, update as pricing changes)
OPENAI_COSTS = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.000150, "completion": 0.000600},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}


class OpenAIAdapter(AgentAdapter):
    """
    Adapter for OpenAI chat completion agents with optional tool use.

    Config options::

        type: openai
        model: gpt-4o-mini
        system_prompt: "You are a helpful assistant."
        tools: []          # OpenAI tool definitions
        temperature: 0
        max_tokens: 1000
    """

    adapter_type = "openai"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._client: Any = None

    async def setup(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install agentbench[openai]")
        self._client = AsyncOpenAI()

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        if self._client is None:
            await self.setup()

        model = self.config.get("model", "gpt-4o-mini")
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        tools = self.config.get("tools", [])
        temperature = self.config.get("temperature", 0)
        max_tokens = self.config.get("max_tokens", 1000)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input},
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        start = time.monotonic()
        response = await self._client.chat.completions.create(**kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000

        message = response.choices[0].message
        response_text = message.content or ""

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            import json
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    )
                )

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else 0

        costs = OPENAI_COSTS.get(model, {"prompt": 0, "completion": 0})
        cost = (
            prompt_tokens / 1000 * costs["prompt"]
            + completion_tokens / 1000 * costs["completion"]
        )

        return AgentResult(
            response=response_text,
            tool_calls=tool_calls,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            metadata={"model": model, "adapter": "openai"},
        )
