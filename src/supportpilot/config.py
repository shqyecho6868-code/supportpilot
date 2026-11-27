from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    # Production default. Local dev (.env) and CI override it with Haiku to save cost.
    anthropic_model: str = "claude-sonnet-5-5"
    llm_provider: Literal["anthropic", "rule_based"] = "anthropic"

    database_url: str = "postgresql://supportpilot:supportpilot@localhost:5432/supportpilot"
    redis_url: str = "redis://localhost:6379/0"

    memory_max_messages: int = 20
    memory_ttl_seconds: int = 86_400

    # Upper bound on model <-> tool round trips within a single user turn.
    max_agent_steps: int = 8
    # "single": one agent holds every tool (default since P5). "routed": a router sends each turn to
    # a specialist agent. Compared in docs/eval-report.md: the single agent completed more tasks,
    # mixed-intent and multi-turn ones above all, at a lower median latency and the same cost.
    agent_architecture: Literal["routed", "single"] = "single"

    # Small, fast model for query rewriting, agentic-RAG decisions, long-term memory extraction
    # and contextual-retrieval chunk context.
    fast_model: str = "claude-haiku-5-5"

    # Reliability. Transient failures are retried with exponential backoff; if the main model
    # still fails (or does not exist) the call is repeated on fallback_model.
    fallback_model: str = "claude-haiku-5-5"
    # Policy declines (stop_reason "refusal") on Sonnet 5.5 / Opus 5.5 are re-run by the API on its
    # default fallback model (server-side fallback, `fallbacks: "default"`). Haiku has none.
    server_side_fallback: bool = True
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 0.5
    llm_retry_max_delay: float = 8.0
    # Upper bound on tokens (input + output, including cached input) per conversation. Past it the
    # assistant ends politely and hands the conversation to a human. 0 disables the limit.
    conversation_token_limit: int = 200_000

    # Human-in-the-loop: a pending confirmation expires after this many seconds.
    confirmation_ttl_seconds: int = 900

    # Agentic RAG: decide whether to retrieve, grade the passages, rewrite and search again
    # (at most agentic_rag_max_rewrites extra rounds). Off by default: see docs/eval-report.md.
    agentic_rag: bool = False
    agentic_rag_max_rewrites: int = 2

    # Long-term memory (mem0, stored in pgvector): customer preferences and past issues across
    # sessions. Uses fast_model to extract memories after each turn.
    long_term_memory: bool = True
    long_term_memory_top_k: int = 5
    # mem0 extraction (docs/eval-report.md, P5): cache mem0's fixed extraction prompt (default: -61%
    # cost per turn, recall unchanged); "rules" sends only turns that can carry a durable fact
    # (longterm.worth_remembering: a further -28%, but missed 2 of 22 facts, so off by default);
    # sync awaits the extraction before the reply (only to measure the background path against).
    long_term_memory_gate: Literal["rules", "all"] = "all"
    long_term_memory_cache: bool = True
    long_term_memory_sync: bool = False
    mem0_collection: str = "mem0_memories"
    mem0_history_db: str = ".mem0/history.db"

    # Prompt-injection guard on customer messages and tool results.
    injection_guard: bool = True

    # Langfuse tracing: on when both keys are set.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None
    langfuse_enabled: bool = True
    app_environment: str = "development"

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
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
