"""
Azure OpenAI adapter — evaluates agents running on Azure OpenAI deployments.
Requires: pip install agentbench[azure]
"""
from __future__ import annotations

import os
from typing import Any

from agentbench.adapters.openai_adapter import OpenAIAdapter

DEFAULT_API_VERSION = "2024-10-21"
ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"


def resolve_azure_settings(config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve Azure OpenAI connection settings from adapter config, falling back
    to the standard ``AZURE_OPENAI_*`` environment variables.

    Raises ``ValueError`` naming every missing setting.
    """
    auth = str(config.get("auth", "api_key")).lower()
    if auth not in ("api_key", "entra"):
        raise ValueError(f"Unsupported Azure OpenAI auth {auth!r}. Use 'api_key' or 'entra'.")

    settings = {
        "endpoint": config.get("endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "deployment": config.get("deployment") or os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        "api_version": (
            config.get("api_version")
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or os.environ.get("OPENAI_API_VERSION")
            or DEFAULT_API_VERSION
        ),
        "api_key": config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "auth": auth,
    }

    missing = []
    if not settings["endpoint"]:
        missing.append("endpoint (or AZURE_OPENAI_ENDPOINT)")
    if not settings["deployment"]:
        missing.append("deployment (or AZURE_OPENAI_DEPLOYMENT)")
    if auth == "api_key" and not settings["api_key"]:
        missing.append("api_key (or AZURE_OPENAI_API_KEY), or set auth: entra")
    if missing:
        raise ValueError("Azure OpenAI is not configured. Missing: " + "; ".join(missing))

    return settings


def create_azure_client(settings: dict[str, Any]) -> tuple[Any, Any]:
    """
    Build an ``AsyncAzureOpenAI`` client.

    Returns ``(client, credential)``; ``credential`` is the Entra ID credential
    to close on teardown, or ``None`` for API-key auth.
    """
    try:
        from openai import AsyncAzureOpenAI
    except ImportError:
        raise ImportError("Install the Azure extra: pip install agentbench[azure]") from None

    kwargs: dict[str, Any] = {
        "azure_endpoint": settings["endpoint"],
        "api_version": settings["api_version"],
    }
    credential = None

    if settings["auth"] == "entra":
        try:
            from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
        except ImportError:
            raise ImportError(
                "Entra ID auth needs azure-identity: pip install agentbench[azure]"
            ) from None
        credential = DefaultAzureCredential()
        kwargs["azure_ad_token_provider"] = get_bearer_token_provider(credential, ENTRA_SCOPE)
    else:
        kwargs["api_key"] = settings["api_key"]

    return AsyncAzureOpenAI(**kwargs), credential


class AzureOpenAIAdapter(OpenAIAdapter):
    """
    Adapter for chat completion agents hosted on Azure OpenAI, with optional tool use.

    Config options::

        type: azure_openai
        endpoint: https://my-resource.openai.azure.com   # or AZURE_OPENAI_ENDPOINT
        deployment: my-gpt-4o-mini                         # or AZURE_OPENAI_DEPLOYMENT
        api_version: "2024-10-21"                          # or AZURE_OPENAI_API_VERSION
        auth: api_key       # api_key (AZURE_OPENAI_API_KEY) or entra (Microsoft Entra ID)
        model: gpt-4o-mini  # optional: underlying model, used only for cost estimates
        system_prompt: "You are a helpful assistant."
        tools: []           # OpenAI tool definitions
        temperature: 0
        max_tokens: 1000

    Put secrets in environment variables rather than YAML files.
    """

    adapter_type = "azure_openai"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._credential: Any = None
        self._deployment: str = ""

    def _create_client(self) -> Any:
        settings = resolve_azure_settings(self.config)
        self._deployment = settings["deployment"]
        client, self._credential = create_azure_client(settings)
        return client

    def _model_param(self) -> str:
        # Azure routes requests by deployment name, sent as `model`.
        return self._deployment

    def _pricing_model(self, response: Any) -> str | None:
        # Deployment names are user-chosen, so price by the configured model,
        # else by the model Azure reports it served.
        return self.config.get("model") or getattr(response, "model", None)

    async def teardown(self) -> None:
        await super().teardown()
        if self._credential is not None:
            await self._credential.close()
            self._credential = None
