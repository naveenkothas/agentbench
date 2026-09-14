"""
Global configuration loaded from environment variables and .env files.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = Field("sqlite+aiosqlite:///./agentbench.db", alias="DATABASE_URL")

    # LLM providers (optional — not required for local evaluations)
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    azure_openai_api_key: str = Field("", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field("", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field("", alias="AZURE_OPENAI_DEPLOYMENT")

    # LLM Judge settings
    judge_model: str = Field("gpt-4o-mini", alias="AGENTBENCH_JUDGE_MODEL")
    judge_provider: str = Field("openai", alias="AGENTBENCH_JUDGE_PROVIDER")

    # Server
    api_host: str = Field("0.0.0.0", alias="AGENTBENCH_HOST")
    api_port: int = Field(8080, alias="AGENTBENCH_PORT")

    # Evaluation
    max_concurrency: int = Field(5, alias="AGENTBENCH_MAX_CONCURRENCY")
    default_timeout_sec: float = Field(30.0, alias="AGENTBENCH_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
