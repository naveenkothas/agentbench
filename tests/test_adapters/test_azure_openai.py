"""
Tests for AzureOpenAIAdapter.

These run the real OpenAI SDK against a mocked HTTP transport, so they check
the actual requests Azure would receive: URL, api-version and auth headers.
"""
import json
import sys
import types

import httpx
import pytest

openai = pytest.importorskip("openai")

from agentbench.adapters.azure_openai_adapter import (  # noqa: E402
    DEFAULT_API_VERSION,
    ENTRA_SCOPE,
    AzureOpenAIAdapter,
    resolve_azure_settings,
)
from agentbench.adapters.registry import AdapterRegistry  # noqa: E402

ENDPOINT = "https://contoso.openai.azure.com"
AZURE_ENV = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "OPENAI_API_VERSION",
)


def completion(content="Hello from Azure", tool_calls=None, model="gpt-4o-mini-2024-07-18",
               finish_reason="stop"):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        }],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 2000, "total_tokens": 3000},
    }


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in AZURE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def azure_http(monkeypatch):
    """Route the real AsyncAzureOpenAI client through a mock transport."""
    state = {"requests": [], "response": completion(), "client_kwargs": None}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        return httpx.Response(200, json=state["response"])

    real_client = openai.AsyncAzureOpenAI

    def client_factory(**kwargs):
        state["client_kwargs"] = kwargs
        return real_client(**kwargs, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai, "AsyncAzureOpenAI", client_factory)
    return state


def make_adapter(**overrides):
    config = {"endpoint": ENDPOINT, "deployment": "support-bot", "api_key": "secret"}
    config.update(overrides)
    return AzureOpenAIAdapter(config)


async def test_sends_request_to_deployment_with_api_key(azure_http):
    adapter = make_adapter(system_prompt="Be brief.", temperature=0.2, max_tokens=50)
    result = await adapter.run("Where is my order?", {})
    await adapter.teardown()

    request = azure_http["requests"][0]
    assert request.url.host == "contoso.openai.azure.com"
    assert request.url.path == "/openai/deployments/support-bot/chat/completions"
    assert request.url.params["api-version"] == DEFAULT_API_VERSION
    assert request.headers["api-key"] == "secret"

    body = json.loads(request.content)
    assert body["model"] == "support-bot"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Where is my order?"},
    ]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 50
    assert "tools" not in body

    assert result.response == "Hello from Azure"
    assert result.total_tokens == 3000
    assert result.metadata["adapter"] == "azure_openai"
    assert result.metadata["model"] == "support-bot"
    assert result.metadata["finish_reason"] == "stop"


async def test_parses_tool_calls_and_sends_tools(azure_http):
    azure_http["response"] = completion(content=None, finish_reason="tool_calls", tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_order_status", "arguments": '{"order_id": "123"}'},
    }])
    tools = [{"type": "function", "function": {"name": "get_order_status", "parameters": {}}}]
    adapter = make_adapter(tools=tools)
    result = await adapter.run("Status of 123?", {})
    await adapter.teardown()

    assert json.loads(azure_http["requests"][0].content)["tools"] == tools
    assert result.response == ""
    assert result.tool_calls[0].name == "get_order_status"
    assert result.tool_calls[0].arguments == {"order_id": "123"}


async def test_cost_uses_model_reported_by_azure(azure_http):
    adapter = make_adapter()
    result = await adapter.run("hi", {})
    await adapter.teardown()
    # gpt-4o-mini-2024-07-18 prices as gpt-4o-mini: 1K prompt + 2K completion tokens
    assert result.estimated_cost_usd == pytest.approx(0.000150 + 2 * 0.000600)


async def test_cost_prefers_configured_model(azure_http):
    adapter = make_adapter(model="gpt-4o")
    result = await adapter.run("hi", {})
    await adapter.teardown()
    assert result.estimated_cost_usd == pytest.approx(0.005 + 2 * 0.015)


async def test_reads_settings_from_environment(azure_http, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    adapter = AzureOpenAIAdapter()
    await adapter.run("hi", {})
    await adapter.teardown()

    request = azure_http["requests"][0]
    assert request.url.path == "/openai/deployments/env-deployment/chat/completions"
    assert request.url.params["api-version"] == "2025-01-01-preview"
    assert request.headers["api-key"] == "env-key"


async def test_entra_id_auth_sends_bearer_token(azure_http, monkeypatch):
    closed = []

    class FakeCredential:
        async def close(self):
            closed.append(True)

    def get_bearer_token_provider(credential, scope):
        assert isinstance(credential, FakeCredential)
        assert scope == ENTRA_SCOPE

        async def provider():
            return "entra-token"
        return provider

    aio = types.ModuleType("azure.identity.aio")
    aio.DefaultAzureCredential = FakeCredential
    aio.get_bearer_token_provider = get_bearer_token_provider
    monkeypatch.setitem(sys.modules, "azure.identity.aio", aio)

    adapter = make_adapter(api_key=None, auth="entra")
    await adapter.run("hi", {})
    await adapter.teardown()

    request = azure_http["requests"][0]
    assert request.headers["authorization"] == "Bearer entra-token"
    assert "api-key" not in request.headers
    assert "api_key" not in azure_http["client_kwargs"]
    assert closed == [True]


async def test_teardown_closes_client(azure_http):
    adapter = make_adapter()
    await adapter.run("hi", {})
    client = adapter._client
    await adapter.teardown()
    assert adapter._client is None
    assert client.is_closed()


def test_missing_settings_are_all_reported():
    with pytest.raises(ValueError) as error:
        resolve_azure_settings({})
    message = str(error.value)
    assert "AZURE_OPENAI_ENDPOINT" in message
    assert "AZURE_OPENAI_DEPLOYMENT" in message
    assert "AZURE_OPENAI_API_KEY" in message


def test_entra_auth_does_not_need_api_key():
    settings = resolve_azure_settings({"endpoint": ENDPOINT, "deployment": "d", "auth": "entra"})
    assert settings["auth"] == "entra"


def test_unknown_auth_rejected():
    with pytest.raises(ValueError, match="Unsupported Azure OpenAI auth"):
        resolve_azure_settings({"endpoint": ENDPOINT, "deployment": "d", "auth": "password"})


def test_config_overrides_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "from-env")
    settings = resolve_azure_settings({"endpoint": ENDPOINT, "deployment": "from-yaml",
                                       "api_key": "k"})
    assert settings["deployment"] == "from-yaml"


def test_registered_as_azure_openai():
    adapter = AdapterRegistry.create("azure_openai", {"deployment": "d"})
    assert isinstance(adapter, AzureOpenAIAdapter)


async def test_runs_from_yaml_through_runner(azure_http, tmp_path):
    from agentbench.datasets.loader import DatasetLoader
    from agentbench.runners.evaluation_runner import EvaluationRunner

    path = tmp_path / "azure.yaml"
    path.write_text(f"""
name: Azure suite
agent:
  type: azure_openai
  endpoint: {ENDPOINT}
  deployment: support-bot
  api_key: secret
cases:
  - id: greet
    input: Say hello
    expected:
      contains: [Azure]
""")
    run = await EvaluationRunner(DatasetLoader.from_yaml(path)).run()

    assert run.agent_type == "azure_openai"
    assert run.pass_rate == 1.0
