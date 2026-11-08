"""Tests for LLMJudgeEvaluator provider selection, including Azure OpenAI."""
import json

import httpx
import pytest

openai = pytest.importorskip("openai")

from agentbench.core.config import get_settings  # noqa: E402
from agentbench.core.models import AgentResult, EvalCase  # noqa: E402
from agentbench.evaluators.llm_judge import LLMJudgeEvaluator  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # ignore any developer .env file
    for name in ("AGENTBENCH_JUDGE_PROVIDER", "AZURE_OPENAI_ENDPOINT",
                 "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_KEY",
                 "AZURE_OPENAI_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def azure_judge_http(monkeypatch):
    requests = []
    verdict = {"score": 0.9, "passed": True, "reason": "Accurate."}

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(verdict)}}],
        })

    real_client = openai.AsyncAzureOpenAI
    monkeypatch.setattr(openai, "AsyncAzureOpenAI", lambda **kw: real_client(
        **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    return requests


def _azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://contoso.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "judge-gpt4o")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")


async def _judge(evaluator):
    return await evaluator.evaluate(AgentResult(response="Paris"),
                                    EvalCase(id="q", input="Capital of France?"))


async def test_azure_openai_judge(azure_judge_http, monkeypatch):
    _azure_env(monkeypatch)
    score = await _judge(LLMJudgeEvaluator({"provider": "azure_openai"}))

    assert score.passed
    assert score.score == 0.9
    assert score.details["provider"] == "azure_openai"
    request = azure_judge_http[0]
    assert request.url.path == "/openai/deployments/judge-gpt4o/chat/completions"
    assert request.headers["api-key"] == "secret"
    assert json.loads(request.content)["response_format"] == {"type": "json_object"}


async def test_provider_defaults_to_environment_setting(azure_judge_http, monkeypatch):
    _azure_env(monkeypatch)
    monkeypatch.setenv("AGENTBENCH_JUDGE_PROVIDER", "azure_openai")
    score = await _judge(LLMJudgeEvaluator())

    assert score.details["provider"] == "azure_openai"
    assert len(azure_judge_http) == 1


async def test_azure_judge_reports_missing_configuration():
    score = await _judge(LLMJudgeEvaluator({"provider": "azure_openai"}))
    assert not score.passed
    assert "AZURE_OPENAI_ENDPOINT" in score.reason


async def test_unknown_provider_fails_cleanly():
    score = await _judge(LLMJudgeEvaluator({"provider": "nope"}))
    assert not score.passed
    assert "Unsupported judge provider" in score.reason
