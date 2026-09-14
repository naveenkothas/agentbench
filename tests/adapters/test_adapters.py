"""Tests for built-in agent adapters."""

import pytest

from agentbench.adapters.builtin import CustomAdapter, EchoAdapter
from agentbench.core.models import AgentConfig, AgentResult


def make_config(adapter_type: str = "echo", **kwargs) -> AgentConfig:
    return AgentConfig(type=adapter_type, extra=kwargs)


class TestEchoAdapter:
    @pytest.mark.asyncio
    async def test_returns_configured_response(self):
        config = make_config("echo", response="Hello from echo!")
        adapter = EchoAdapter(config)
        result = await adapter.run("test input", {})
        assert result.response == "Hello from echo!"

    @pytest.mark.asyncio
    async def test_default_echo_format(self):
        config = make_config("echo")
        adapter = EchoAdapter(config)
        result = await adapter.run("test message", {})
        assert "test message" in result.response

    @pytest.mark.asyncio
    async def test_returns_configured_tool_calls(self):
        config = make_config(
            "echo",
            response="Checking...",
            tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
        )
        adapter = EchoAdapter(config)
        result = await adapter.run("test", {})
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments["query"] == "test"

    @pytest.mark.asyncio
    async def test_tracks_latency(self):
        config = make_config("echo", response="fast")
        adapter = EchoAdapter(config)
        result = await adapter.run("test", {})
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_counts_tokens(self):
        config = make_config("echo", response="one two three")
        adapter = EchoAdapter(config)
        result = await adapter.run("hello world", {})
        assert result.input_tokens > 0
        assert result.output_tokens > 0


class TestCustomAdapter:
    @pytest.mark.asyncio
    async def test_calls_user_function(self):
        config = make_config("custom")
        adapter = CustomAdapter(config)

        async def my_agent(input: str, context: dict) -> AgentResult:
            return AgentResult(response=f"processed: {input}")

        adapter.set_callable(my_agent)
        result = await adapter.run("hello", {})
        assert result.response == "processed: hello"

    @pytest.mark.asyncio
    async def test_raises_without_callable(self):
        config = make_config("custom")
        adapter = CustomAdapter(config)
        with pytest.raises(RuntimeError, match="requires a callable"):
            await adapter.run("test", {})

    @pytest.mark.asyncio
    async def test_passes_context(self):
        config = make_config("custom")
        adapter = CustomAdapter(config)
        received_context = {}

        async def capture_context(input: str, context: dict) -> AgentResult:
            received_context.update(context)
            return AgentResult(response="ok")

        adapter.set_callable(capture_context)
        await adapter.run("test", {"user_id": "abc123"})
        assert received_context.get("user_id") == "abc123"
