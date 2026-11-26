-- Knowledge base (policy documents). Created by `scripts/ingest_kb.py`, not by the
-- Postgres entrypoint, because the BM25 index in 04_kb_bm25.sql is optional.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
    id                TEXT PRIMARY KEY,            -- "<doc_id>#<section-slug>[-n]"
    doc_id            TEXT NOT NULL,
    doc_title         TEXT NOT NULL,
    section           TEXT NOT NULL,
    chunk_index       INTEGER NOT NULL,            -- position within the document
    content           TEXT NOT NULL,               -- the raw chunk (starts with its section heading)
    context           TEXT NOT NULL DEFAULT '',    -- contextual-retrieval blurb generated at ingest
    contextualized    TEXT NOT NULL,               -- context + content, used when contextual retrieval is on
    embedding         vector(384) NOT NULL,        -- embedding of content
    embedding_ctx     vector(384) NOT NULL         -- embedding of contextualized
);

-- Postgres full-text fallback for when pg_search is unavailable.
CREATE INDEX IF NOT EXISTS kb_chunks_fts ON kb_chunks USING gin (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS kb_chunks_fts_ctx ON kb_chunks USING gin (to_tsvector('english', contextualized));
