# Policy retrieval

The knowledge agent answers store-policy questions from eight Markdown documents in
`kb/policies/` (returns, refunds, shipping, exchanges, warranty, payment, Larkspur Plus
membership, privacy). The documents are written to overlap on purpose: several windows of
14, 15, 30, 45 and 60 days, member vs. non-member rules, and electronics, furniture and
final-sale exceptions. That overlap is what lets the evaluation tell good retrieval from bad.

## Pipeline

```
customer message
  -> query rewrite (follow-ups only; Haiku, thinking off)      RETRIEVAL_QUERY_REWRITE
  -> MCP tool search_policy(query)
       vector: bge-small-en-v1.5 (fastembed, local) + pgvector  RETRIEVAL_VECTOR
       BM25:   ParadeDB pg_search, English stemming             RETRIEVAL_BM25 / RETRIEVAL_BM25_BACKEND
       -> reciprocal rank fusion (k = 60)
       -> bge-reranker-base cross-encoder (fastembed, local)    RETRIEVAL_RERANK
       -> top 5 passages, each with "source" = "<Document> § <Section>"
  -> answer from the passages only, citing [Document § Section]
```

Every stage is a setting (environment variable or `.env`), read into `RetrievalConfig`:

| Setting | Default | Effect |
|---|---|---|
| `RETRIEVAL_VECTOR` | `true` | dense retrieval over `embedding` / `embedding_ctx` |
| `RETRIEVAL_BM25` | `true` | lexical retrieval; fused with the vector list by RRF |
| `RETRIEVAL_BM25_BACKEND` | `auto` | `paradedb` (pg_search), `postgres_fts` (built-in full-text search), or `auto` |
| `RETRIEVAL_RERANK` | `true` | cross-encoder rerank of the fused candidates |
| `RETRIEVAL_CONTEXTUAL` | `true` | search the contextualized text and embedding instead of the raw chunk |
| `RETRIEVAL_QUERY_REWRITE` | `true` | rewrite follow-up questions into standalone queries |
| `RETRIEVAL_TOP_K` / `RETRIEVAL_CANDIDATES` / `RETRIEVAL_RRF_K` | 5 / 20 / 60 | result count, candidates per retriever, RRF constant |

At least one of vector and BM25 must be on. With only one of them, fusion keeps its order.

## Chunking

One chunk per `##` section (sections longer than 220 words would be split on paragraphs;
none currently are), 55 chunks in total. The section is the unit of citation. Chunk ids are
stable: `returns#electronics`. A chunk keeps its own section heading, as a plain Markdown
splitter would, but not the document title.

## Contextual Retrieval

At ingest time Claude Haiku writes one or two sentences that place each chunk in its
document ("This chunk is from the Exchanges Policy and covers the limits on exchanges...").
The table stores both versions of every chunk, so the ablation can switch the feature without
re-ingesting:

| column | used when contextual retrieval is |
|---|---|
| `content`, `embedding` | off |
| `contextualized` (= context + content), `embedding_ctx` | on |

The whole document goes into a system block with `cache_control`, and only the chunk changes
between calls, so every chunk after the first in a document reads the document from the prompt
cache. On the first full ingest: 36,963 input tokens read from cache, 6,268 written to cache,
6,328 uncached. Generated contexts are stored in `kb/contexts.json`, keyed by chunk id and a
hash of (document, chunk); re-ingesting unchanged documents makes no API calls, which is also
how the offline test suite indexes the KB.

## BM25 backend

ParadeDB `pg_search` 0.27 works in the compose image, so BM25 runs on a `bm25` index over
`content` and `contextualized` with the `pdb.simple('stemmer=english')` tokenizer, queried with
the match-any operator `|||` (the query string is tokenized, so free text cannot inject query
syntax). No fallback was needed. If `CREATE EXTENSION pg_search` fails, ingestion logs a warning
and `auto` switches to Postgres full-text search (`to_tsvector('english')` with an OR query and
`ts_rank_cd`); `RETRIEVAL_BM25_BACKEND=postgres_fts` forces it, and the test suite covers it.

## Citations

The answer prompt gives each passage a `cite` attribute and requires `[Document § Section]`
after every fact. The graph checks citations against the retrieved passages; if the model cited
none, the top two sources are appended as a `Sources:` line so no policy answer goes out without
them. The eval reports `answer.citation_rate`: how often the model cited by itself.

## Sampling parameters

The installed `anthropic` SDK (2.14, API version 2026-10-21) no longer accepts `temperature`,
`top_p` or `top_k`, so "temperature 0" cannot be set. Query rewriting, chunk contexts, the eval
judge and Ragas run with thinking disabled and `effort: low` instead, the most stable setting
available. LLM-graded metrics still move by a few points between runs.
