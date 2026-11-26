-- ParadeDB pg_search BM25 index over both text columns, with English stemming.
-- Optional: if this fails, retrieval falls back to Postgres full-text search.

CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE INDEX IF NOT EXISTS kb_chunks_bm25 ON kb_chunks
USING bm25 (
    id,
    (content::pdb.simple('stemmer=english')),
    (contextualized::pdb.simple('stemmer=english'))
);
