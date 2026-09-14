"""Built-in agent adapters for AgentBench."""

from __future__ import annotations

import time
from typing import Any, Callable, Awaitable

from agentbench.adapters.base import BaseAgentAdapter, register_adapter
from agentbench.core.models import AgentConfig, AgentResult, ToolCall


@register_adapter
class EchoAdapter(BaseAgentAdapter):
    """
    A deterministic echo adapter for testing without an LLM.
    Returns a configurable response — perfect for CI/CD and local testing.
    """

    adapter_type = "echo"

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        start = time.monotonic()
        response = self.config.extra.get("response", f"Echo: {input}")
        tool_calls_raw = self.config.extra.get("tool_calls", [])

        tool_calls = [
            ToolCall(name=tc.get("name", "unknown"), arguments=tc.get("arguments", {}))
            for tc in tool_calls_raw
        ]

        return AgentResult(
            response=response,
            tool_calls=tool_calls,
            latency_ms=(time.monotonic() - start) * 1000,
            input_tokens=len(input.split()),
            output_tokens=len(response.split()),
            total_tokens=len(input.split()) + len(response.split()),
            estimated_cost_usd=0.0,
            metadata={"adapter": "echo"},
        )


@register_adapter
class CustomAdapter(BaseAgentAdapter):
    """
    Adapter for user-defined Python callables.

    Expects the config.extra["callable"] key to contain an async function
    with signature: async (input: str, context: dict) -> AgentResult
    """

    adapter_type = "custom"
    _fn: Callable[[str, dict[str, Any]], Awaitable[AgentResult]] | None = None

    def set_callable(
        self, fn: Callable[[str, dict[str, Any]], Awaitable[AgentResult]]
    ) -> None:
        """Attach an async callable to this adapter."""
        self._fn = fn

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        if self._fn is None:
            raise RuntimeError(
                "CustomAdapter requires a callable. Call adapter.set_callable(fn) "
                "or use the Python SDK to attach your agent function."
            )
        return await self._fn(input, context)


@register_adapter
class OpenAIAdapter(BaseAgentAdapter):
    """
    Adapter for OpenAI Chat Completions API.

    Requires: pip install agentbench[openai]
    Environment variable: OPENAI_API_KEY
    """

    adapter_type = "openai"

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI adapter requires 'openai'. Install with: pip install agentbench[openai]"
            )

        start = time.monotonic()
        client = openai.AsyncOpenAI()
        model = self.config.model or "gpt-4o-mini"

        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": input})

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.config.temperature or 0.0,
            max_tokens=self.config.max_tokens or 1024,
        )

        latency_ms = (time.monotonic() - start) * 1000
        message = response.choices[0].message
        content = message.content or ""

        # Parse tool calls if present
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

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Rough cost estimate (varies by model)
        cost_per_1k = {"gpt-4o": 0.005, "gpt-4o-mini": 0.00015, "gpt-4": 0.03}
        rate = cost_per_1k.get(model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model, 0.002)
        estimated_cost = ((input_tokens + output_tokens) / 1000) * rate

        return AgentResult(
            response=content,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=estimated_cost,
            metadata={"model": model, "adapter": "openai"},
        )


@register_adapter
class AnthropicAdapter(BaseAgentAdapter):
    """
    Adapter for Anthropic Claude API.

    Requires: pip install agentbench[anthropic]
    Environment variable: ANTHROPIC_API_KEY
    """

    adapter_type = "anthropic"

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic adapter requires 'anthropic'. "
                "Install with: pip install agentbench[anthropic]"
            )

        start = time.monotonic()
        client = anthropic.AsyncAnthropic()
        model = self.config.model or "claude-3-haiku-20240307"

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.config.max_tokens or 1024,
            "messages": [{"role": "user", "content": input}],
        }
        if self.config.system_prompt:
            kwargs["system"] = self.config.system_prompt
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        response = await client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Approximate cost
        cost_rates = {
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
        }
        rate = next((v for k, v in cost_rates.items() if k in model), 0.003)
        estimated_cost = ((input_tokens + output_tokens) / 1000) * rate

        return AgentResult(
            response=content,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=estimated_cost,
            metadata={"model": model, "adapter": "anthropic"},
        )


@register_adapter
class AzureOpenAIAdapter(BaseAgentAdapter):
    """
    Adapter for Azure OpenAI deployments.

    Requires: pip install agentbench[openai]
    Environment variables: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
    """

    adapter_type = "azure_openai"

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Azure OpenAI adapter requires 'openai'. "
                "Install with: pip install agentbench[openai]"
            )

        import os
        start = time.monotonic()

        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=self.config.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=self.config.extra.get("api_version", "2024-02-01"),
        )

        deployment = self.config.extra.get(
            "deployment", os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        )

        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": input})

        response = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=self.config.temperature or 0.0,
            max_tokens=self.config.max_tokens or 1024,
        )

        latency_ms = (time.monotonic() - start) * 1000
        content = response.choices[0].message.content or ""
        usage = response.usage

        return AgentResult(
            response=content,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            metadata={"deployment": deployment, "adapter": "azure_openai"},
        )
