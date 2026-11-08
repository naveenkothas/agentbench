"""Tests for EchoAdapter."""
import pytest

from agentbench.adapters.echo import EchoAdapter


@pytest.mark.asyncio
async def test_echo_basic():
    adapter = EchoAdapter()
    result = await adapter.run("hello world", {})
    assert "hello world" in result.response
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_echo_custom_template():
    adapter = EchoAdapter(config={"response_template": "Answer: {input}"})
    result = await adapter.run("test query", {})
    assert result.response == "Answer: test query"


@pytest.mark.asyncio
async def test_echo_tokens_counted():
    adapter = EchoAdapter()
    result = await adapter.run("hello world test", {})
    assert result.total_tokens > 0
