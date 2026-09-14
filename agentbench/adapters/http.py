"""
HTTPAdapter — calls any agent exposed as an HTTP endpoint.

Useful for evaluating deployed agents or microservices.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from agentbench.adapters.base import AgentAdapter
from agentbench.core.models import AgentResult


class HTTPAdapter(AgentAdapter):
    """
    Adapter that sends requests to an HTTP agent endpoint.

    Config options::

        type: http
        url: "http://localhost:8000/agent/run"
        method: POST
        headers:
          Authorization: "Bearer ${MY_TOKEN}"
        response_field: "response"
        timeout: 30
    """

    adapter_type = "http"

    async def run(self, input: str, context: dict[str, Any]) -> AgentResult:
        url = self.config.get("url", "")
        if not url:
            raise ValueError("HTTPAdapter requires 'url' in config")

        method = self.config.get("method", "POST").upper()
        headers = self.config.get("headers", {})
        response_field = self.config.get("response_field", "response")
        timeout = self.config.get("timeout", 30)

        payload = {"input": input, "context": context}

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout) as client:
            http_resp = await client.request(method, url, json=payload, headers=headers)
            http_resp.raise_for_status()
        elapsed_ms = (time.monotonic() - start) * 1000

        data = http_resp.json()
        response_text = data.get(response_field, str(data))

        return AgentResult(
            response=response_text,
            latency_ms=elapsed_ms,
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            estimated_cost_usd=data.get("cost_usd", 0.0),
            metadata={"adapter": "http", "url": url, "status_code": http_resp.status_code},
            raw=data,
        )
