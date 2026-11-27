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
       -> bge-reranker-v2-m3 cross-encoder (fastembed, local)   RETRIEVAL_RERANK / RERANKER_MODEL
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
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | the cross-encoder; CI's eval gate uses `jinaai/jina-reranker-v1-tiny-en` (see eval-report.md, P4) |
| `RETRIEVAL_CONTEXTUAL` | `true` | search the contextualized text and embedding instead of the raw chunk |
| `RETRIEVAL_QUERY_REWRITE` | `true` | rewrite follow-up questions into standalone queries |
| `RETRIEVAL_TOP_K` / `RETRIEVAL_CANDIDATES` / `RETRIEVAL_RRF_K` | 5 / 20 / 60 | result count, candidates per retriever, RRF constant |

At least one of vector and BM25 must be on. With only one of them, fusion keeps its order.

## Agentic RAG (optional, `AGENTIC_RAG=true`)

The default knowledge agent is single-shot: rewrite (follow-ups only), search once, answer. With
agentic RAG the agent controls retrieval itself, with Haiku making each decision through a small
structured-output call (thinking off, low effort):

```
decide   -- retrieval_decision: does this message need a policy search? (a first message always
            searches; a follow-up such as "thanks" or one already answered in the conversation may
            skip it). Also writes the standalone query.
search   -- search_policy(query), same hybrid pipeline as above
grade    -- retrieval_grade: are the passages enough for a complete answer, including the
            conditions (membership, product category, fees, days)? If not: what is missing, and a
            new query aimed at it
rewrite  -- search again with that query (at most AGENTIC_RAG_MAX_REWRITES = 2 extra searches;
            the last search is not graded, and a repeated query stops the loop)
answer   -- from the union of all passages found (at most 10), citing [Document § Section]
```

Every decision, query and passage list is recorded in the turn's state and Langfuse trace. It
costs 2 to 4 extra Haiku calls per policy question. It is off by default: in the evaluation it
did not improve the answers enough to pay for the extra calls and latency
(see [eval-report.md](eval-report.md#agentic-rag)).

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
`top_p` or `top_k` (they return 400 on every model under that version, and Haiku 5.5 / Sonnet 5.5
reject non-default values even on older versions), so "temperature 0" cannot be set. Query
rewriting, chunk contexts, the eval judge and Ragas run with thinking off and `effort: low`
instead, the most stable setting available (Sonnet 5.5 turns thinking off with `between_tools`;
Opus 5.5 cannot turn it off). LLM-graded metrics still move by a few points between runs.
