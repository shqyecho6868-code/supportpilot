"""Build the knowledge-base tables: chunk, contextualize, embed, index."""

import asyncio
import logging

import psycopg
from psycopg import sql

from supportpilot.db import SQL_DIR
from supportpilot.kb.contextualize import Contextualizer
from supportpilot.kb.documents import chunk_document, load_documents
from supportpilot.kb.models import embed_passages

log = logging.getLogger(__name__)


def _vector(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.7f}" for v in values) + "]"


async def ensure_schema(conn: psycopg.AsyncConnection) -> bool:
    """Create the KB tables. Returns True if the ParadeDB BM25 index is available."""
    await conn.execute((SQL_DIR / "03_kb_schema.sql").read_text())
    try:
        await conn.execute((SQL_DIR / "04_kb_bm25.sql").read_text())
        return True
    except psycopg.Error as exc:
        log.warning("pg_search unavailable, BM25 falls back to Postgres full-text search: %s", exc)
        return False


async def ingest(conninfo: str, contextualizer: Contextualizer, embedding_model: str) -> dict[str, int]:
    docs = load_documents()
    rows = []
    for doc in docs:
        for chunk in chunk_document(doc):
            context = await contextualizer.context_for(doc, chunk)
            contextualized = f"{context}\n\n{chunk.content}" if context else chunk.content
            rows.append((chunk, context, contextualized))
    contextualizer.save()

    embeddings = await asyncio.to_thread(embed_passages, embedding_model, [r[0].content for r in rows])
    embeddings_ctx = await asyncio.to_thread(embed_passages, embedding_model, [r[2] for r in rows])

    async with await psycopg.AsyncConnection.connect(conninfo, autocommit=True) as conn:
        bm25 = await ensure_schema(conn)
        async with conn.transaction():
            await conn.execute(sql.SQL("DELETE FROM kb_chunks"))
            for (chunk, context, contextualized), emb, emb_ctx in zip(
                rows, embeddings, embeddings_ctx, strict=True
            ):
                await conn.execute(
                    "INSERT INTO kb_chunks (id, doc_id, doc_title, section, chunk_index, content, context, "
                    "contextualized, embedding, embedding_ctx) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::vector)",
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.doc_title,
                        chunk.section,
                        chunk.chunk_index,
                        chunk.content,
                        context,
                        contextualized,
                        _vector(emb),
                        _vector(emb_ctx),
                    ),
                )
    return {"documents": len(docs), "chunks": len(rows), "paradedb_bm25": int(bm25)}


async def kb_is_ready(conninfo: str) -> bool:
    async with await psycopg.AsyncConnection.connect(conninfo, autocommit=True) as conn:
        cur = await conn.execute("SELECT to_regclass('kb_chunks') IS NOT NULL")
        (exists,) = await cur.fetchone()
        if not exists:
            return False
        cur = await conn.execute("SELECT count(*) FROM kb_chunks")
        (count,) = await cur.fetchone()
        return count > 0
