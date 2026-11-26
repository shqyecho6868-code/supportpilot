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

    # Small, fast model for query rewriting and contextual-retrieval chunk context.
    fast_model: str = "claude-haiku-5-5"

    # Knowledge-base retrieval. Every stage can be switched off independently.
    retrieval_vector: bool = True
    retrieval_bm25: bool = True
    # "auto" uses ParadeDB pg_search when the extension is available, else Postgres full-text search.
    retrieval_bm25_backend: Literal["auto", "paradedb", "postgres_fts"] = "auto"
    retrieval_rerank: bool = True
    retrieval_contextual: bool = True
    retrieval_query_rewrite: bool = True
    retrieval_top_k: int = 5
    retrieval_candidates: int = 20  # per-retriever candidates before fusion / reranking
    retrieval_rrf_k: int = 60
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"


@lru_cache
def get_settings() -> Settings:
    return Settings()
