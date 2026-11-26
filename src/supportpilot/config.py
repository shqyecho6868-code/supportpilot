from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    llm_provider: Literal["anthropic", "rule_based"] = "anthropic"

    database_url: str = "postgresql://supportpilot:supportpilot@localhost:5432/supportpilot"
    redis_url: str = "redis://localhost:6379/0"

    memory_max_messages: int = 20
    memory_ttl_seconds: int = 86_400

    # Upper bound on model <-> tool round trips within a single user turn.
    max_agent_steps: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
