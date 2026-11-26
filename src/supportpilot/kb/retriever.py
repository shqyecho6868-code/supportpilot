"""Hybrid retrieval over the policy knowledge base.

    query -> [vector (pgvector)] --+
             [BM25 (pg_search)]  --+-> RRF fusion -> [cross-encoder rerank] -> top k

Each stage is switched by ``RetrievalConfig``; with only one retriever enabled,
fusion is a no-op. ``contextual`` selects the contextualized text/embedding
columns written at ingest time instead of the raw chunk columns.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal

from psycopg_pool import AsyncConnectionPool

from supportpilot.config import Settings
from supportpilot.kb.models import aembed_query, arerank_scores

log = logging.getLogger(__name__)

Bm25Backend = Literal["auto", "paradedb", "postgres_fts"]


@dataclass(frozen=True)
class RetrievalConfig:
    vector: bool = True
    bm25: bool = True
    rerank: bool = True
    contextual: bool = True
    bm25_backend: Bm25Backend = "auto"
    top_k: int = 5
    candidates: int = 20
    rrf_k: int = 60
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    @classmethod
    def from_settings(cls, s: Settings) -> "RetrievalConfig":
        return cls(
            vector=s.retrieval_vector,
            bm25=s.retrieval_bm25,
            rerank=s.retrieval_rerank,
            contextual=s.retrieval_contextual,
            bm25_backend=s.retrieval_bm25_backend,
            top_k=s.retrieval_top_k,
            candidates=s.retrieval_candidates,
            rrf_k=s.retrieval_rrf_k,
            embedding_model=s.embedding_model,
            reranker_model=s.reranker_model,
        )

    def with_(self, **changes: Any) -> "RetrievalConfig":
        return replace(self, **changes)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    content: str
    context: str
    score: float

    @property
    def source(self) -> str:
        return f"{self.doc_title} § {self.section}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "text": self.content,
            "score": round(self.score, 4),
        }


_COLUMNS = "id, doc_id, doc_title, section, content, context, contextualized"


class Retriever:
    def __init__(self, pool: AsyncConnectionPool, config: RetrievalConfig | None = None):
        self._pool = pool
        self.config = config or RetrievalConfig()
        self._paradedb: bool | None = None

    async def _has_paradedb(self) -> bool:
        if self._paradedb is None:
            async with self._pool.connection() as conn:
                cur = await conn.execute("SELECT to_regclass('kb_chunks_bm25') IS NOT NULL AS ok")
                self._paradedb = bool((await cur.fetchone())["ok"])
        return self._paradedb

    async def bm25_backend(self, cfg: RetrievalConfig) -> str:
        if cfg.bm25_backend == "auto":
            return "paradedb" if await self._has_paradedb() else "postgres_fts"
        return cfg.bm25_backend

    async def _vector_search(self, query: str, cfg: RetrievalConfig) -> list[dict[str, Any]]:
        column = "embedding_ctx" if cfg.contextual else "embedding"
        vec = await aembed_query(cfg.embedding_model, query)
        literal = "[" + ",".join(f"{v:.7f}" for v in vec) + "]"
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLUMNS}, 1 - ({column} <=> %s::vector) AS score "
                f"FROM kb_chunks ORDER BY {column} <=> %s::vector, id LIMIT %s",
                (literal, literal, cfg.candidates),
            )
            return await cur.fetchall()

    async def _bm25_search(self, query: str, cfg: RetrievalConfig) -> list[dict[str, Any]]:
        column = "contextualized" if cfg.contextual else "content"
        backend = await self.bm25_backend(cfg)
        async with self._pool.connection() as conn:
            if backend == "paradedb":
                # `|||` tokenizes the query with the field's tokenizer and matches any term (OR),
                # so free text is safe here: there is no query syntax to inject.
                cur = await conn.execute(
                    f"SELECT {_COLUMNS}, pdb.score(id) AS score FROM kb_chunks "
                    f"WHERE {column} ||| %s ORDER BY score DESC, id LIMIT %s",
                    (query, cfg.candidates),
                )
            else:
                cur = await conn.execute(
                    f"SELECT {_COLUMNS}, ts_rank_cd(to_tsvector('english', {column}), q) AS score "
                    f"FROM kb_chunks, "
                    f"to_tsquery('english', array_to_string(tsvector_to_array(to_tsvector('english', %s)),"
                    f" ' | ')) AS q "
                    f"WHERE to_tsvector('english', {column}) @@ q ORDER BY score DESC, id LIMIT %s",
                    (query, cfg.candidates),
                )
            return await cur.fetchall()

    async def search(self, query: str, config: RetrievalConfig | None = None) -> list[Hit]:
        cfg = config or self.config
        if not (cfg.vector or cfg.bm25):
            raise ValueError("at least one of vector / bm25 retrieval must be enabled")
        query = query.strip()
        if not query:
            return []

        ranked_lists: list[list[dict[str, Any]]] = []
        if cfg.vector:
            ranked_lists.append(await self._vector_search(query, cfg))
        if cfg.bm25:
            ranked_lists.append(await self._bm25_search(query, cfg))

        # Reciprocal rank fusion. With a single list this keeps its order.
        rows: dict[str, dict[str, Any]] = {}
        fused: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, row in enumerate(ranked, start=1):
                rows.setdefault(row["id"], row)
                fused[row["id"]] = fused.get(row["id"], 0.0) + 1.0 / (cfg.rrf_k + rank)
        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))[: cfg.candidates]
        scores = {cid: fused[cid] for cid in ordered}

        if cfg.rerank and ordered:
            text_col = "contextualized" if cfg.contextual else "content"
            rerank = await arerank_scores(cfg.reranker_model, query, [rows[c][text_col] for c in ordered])
            scores = dict(zip(ordered, rerank, strict=True))
            ordered = sorted(ordered, key=lambda cid: (-scores[cid], cid))

        return [
            Hit(
                chunk_id=cid,
                doc_id=rows[cid]["doc_id"],
                doc_title=rows[cid]["doc_title"],
                section=rows[cid]["section"],
                content=rows[cid]["content"],
                context=rows[cid]["context"],
                score=scores[cid],
            )
            for cid in ordered[: cfg.top_k]
        ]
