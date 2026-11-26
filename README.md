# SupportPilot

An e-commerce customer-support agent built on Claude, LangGraph and the Model Context Protocol.

> Status: work in progress. Done so far: the order-support core (tools, orchestration,
> human-in-the-loop confirmation, short-term memory), a policy knowledge base with hybrid
> retrieval, an evaluation suite, and the production layer: optional agentic RAG, long-term
> memory (mem0), retries and model fallback, a conversation token limit, a prompt-injection
> guard, Langfuse tracing, CI with an eval gate, and a one-command Docker stack.

## Features (so far)

- **Order tools over MCP** -- `get_order`, `list_my_orders`, `update_address`, `create_ticket`.
  The customer's identity is injected by the orchestrator through MCP request `_meta`; no tool
  takes a `user_id` argument, so the model cannot query another customer's data.
- **Idempotent writes** -- `update_address` only works on `pending` orders and takes an
  idempotency key; repeating a call with the same key applies the change once.
- **LangGraph orchestration** -- a router sends each turn to `order_agent`, `knowledge_agent`,
  `chitchat` (small talk and out-of-scope requests: a polite reply, no ticket) or `handoff`
  (opens a ticket for a human, only when the customer asks for one or the request needs one).
- **Policy knowledge base** -- eight policy documents in `kb/policies`, searched through the MCP
  tool `search_policy`: bge-small vectors (pgvector) + BM25 (ParadeDB pg_search) fused with
  RRF, reranked by a local bge-reranker, over chunks enriched with Contextual Retrieval. Follow-up
  questions are rewritten into standalone queries. Answers cite `[Document § Section]`.
  See [docs/retrieval.md](docs/retrieval.md).
- **Evaluation** -- a 48-case golden set, retrieval / answer / agent metrics, a retrieval
  ablation and a baseline regression check. See [docs/eval-report.md](docs/eval-report.md).
- **Human-in-the-loop** -- address changes pause with `interrupt()` until the customer approves
  or rejects via `POST /confirm`. State is checkpointed to Postgres, so a paused turn survives
  a server restart.
- **Short-term memory** -- recent turns per session live in Redis with a sliding TTL. A turn that
  is waiting for confirmation is recorded immediately (marked "not applied"), so it is not lost if
  the customer moves on; confirmations expire after 15 minutes (`/confirm` then returns 410).
- **Long-term memory** -- [mem0](https://github.com/mem0ai/mem0) keeps durable facts per customer
  across sessions (usual shipping address, past issues and tickets) in pgvector. Recalled before
  each turn as unverified context, extracted by Haiku in the background after it.
  `LONG_TERM_MEMORY=false` turns it off.
- **Agentic RAG (optional)** -- `AGENTIC_RAG=true`: the knowledge agent decides whether a search is
  needed, grades the passages, and rewrites the query for what is missing (at most 2 extra
  searches). Off by default because it did not improve the evaluation; see
  [docs/eval-report.md](docs/eval-report.md).
- **Reliability** -- timeouts, retries with exponential backoff and jitter on 429 / 5xx / timeouts,
  and fallback to Haiku when the main model fails or does not exist. A per-conversation token
  limit ends the conversation politely and opens a ticket for a human.
- **Prompt-injection guard** -- customer messages, tool results and recalled memories are scanned;
  flagged messages put the agents on notice, flagged tool data is marked untrusted or dropped.
- **Observability** -- one [Langfuse](https://langfuse.com) trace per turn: routing decision, every
  Claude call (tokens, cost, latency), tool calls, retrieval results, guard decisions. Evaluation
  runs are traced too, tagged `eval`.
- **Streaming API** -- `POST /chat` streams Server-Sent Events.

## Architecture

```
client --SSE--> FastAPI /chat ----> LangGraph (Postgres checkpointer)
                                     router -> order_agent <-> confirm (interrupt)
                                            -> knowledge_agent (rewrite -> search_policy -> cited answer)
                                            -> chitchat
                                            -> handoff
                                     agents --> ToolGateway --MCP--> order tools, search_policy --> Postgres
                                                                     (orders | kb_chunks: pgvector + pg_search)
                       /confirm ---> resume paused thread
                       Redis: conversation history, session ownership, confirm locks, token counts
                       mem0 (pgvector): long-term customer memory      Langfuse: traces
```

## Quick start

Everything (ParadeDB, Redis and the API) with one command:

```bash
cp .env.example .env              # add ANTHROPIC_API_KEY (and optionally the Langfuse keys)
docker compose up -d --build --wait
curl http://localhost:8000/health
```

The first start downloads the local embedding and reranker models (about 1.5 GB, cached in the
`models` volume) and indexes the policy documents, so it takes a few minutes. Logs:
`docker compose logs -f api`.

For development, run only the databases in Docker and the API on the host:

```bash
docker compose up -d --wait postgres redis
uv sync
uv run python scripts/ingest_kb.py   # index the policy documents (chunk contexts are cached in kb/contexts.json)
uv run supportpilot               # http://localhost:8000
```

```bash
curl -N http://localhost:8000/chat -H 'X-User-Id: cust_001' -H 'Content-Type: application/json' \
  -d '{"message": "Change the address of ORD-1001 to 500 Market St, San Francisco, CA 94105, US"}'

curl http://localhost:8000/confirm -H 'X-User-Id: cust_001' -H 'Content-Type: application/json' \
  -d '{"thread_id": "<thread_id from the confirmation_required event>", "decision": "approve"}'
```

Or use the interactive client: `uv run python scripts/chat.py --user cust_001`.

### Langfuse

Create a project at [cloud.langfuse.com](https://cloud.langfuse.com) (Settings -> API keys) and set
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env`. Each `/chat` turn is
one trace in the session of its `session_id`; eval traces carry the tags `eval`, `eval:<command>`
and `run:<run id>`.

## API

| Method | Path       | Description |
|--------|------------|-------------|
| POST   | `/chat`    | `{message, session_id?}` -> SSE: `meta`, `guard`, `route`, `sources`, `token`, `action_result`, `confirmation_required` or `final`, `usage`, `error`, `done` |
| POST   | `/confirm` | `{thread_id, decision: approve \| reject}` -> resumes a paused turn; 410 `confirmation_expired` after `CONFIRMATION_TTL_SECONDS` |
| GET    | `/health`  | Postgres and Redis status |

Authentication is mocked: the `X-User-Id` header carries the customer id.

## Mock data

| Customer | Orders |
|----------|--------|
| `cust_001` Alice Johnson | ORD-1001 pending, ORD-1002 shipped, ORD-1003 delivered |
| `cust_002` Bob Smith     | ORD-2001 pending, ORD-2002 cancelled, ORD-2003 delivered |
| `cust_003` Carol Nguyen  | ORD-3001 shipped, ORD-3002 pending |
| `cust_004` David Garcia  | ORD-4001 delivered, ORD-4002 cancelled, ORD-4003 pending |
| `cust_005` Erin Park     | ORD-5001 delivered (its gift message carries a prompt-injection payload, for the guard tests) |

## Tests

```bash
uv run pytest                     # offline: uses a deterministic rule-based LLM stand-in
uv run pytest -m live             # the same scenarios against real Claude (needs ANTHROPIC_API_KEY)
```

## Evaluation

```bash
uv run supportpilot-eval ablation   # retrieval ablation -> evals/results/ablation.json
uv run supportpilot-eval run        # full eval with the configured pipeline
uv run supportpilot-eval baseline   # full eval, then write evals/baseline.json
uv run supportpilot-eval compare    # full eval vs. baseline; a drop > 5 points re-runs once and
                                    # exits 1 only if the same metric drops in both runs
uv run supportpilot-eval ablation --judge-model claude-sonnet-5-5   # one-off deep evaluation
uv run supportpilot-eval retrieval-sweep  # rerankers x top_k, retrieval metrics only (no grader)
uv run supportpilot-eval run --agentic on # full eval with agentic RAG switched on
uv run supportpilot-eval compare --subset ci --baseline evals/baseline-ci.json   # the PR gate
```

Every run prints its token usage and cost at the end.

## CI

`.github/workflows/ci.yml`: on every push and pull request, ruff and the offline test suite run
against ParadeDB and Redis service containers. On pull requests, the eval gate then runs the
17-case CI subset (`evals/subsets/ci.txt`) with Haiku as agent and grader (about $0.03, a re-run on
a first-run regression doubles it) and fails on a reproducible regression of more than 5 points
against `evals/baseline-ci.json`. Repository secrets: `ANTHROPIC_API_KEY`, and optionally
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.

Routine runs (and CI) are graded by Haiku against `evals/baseline.json`; the one-off deep
evaluation is graded by Sonnet and recorded in `evals/baseline-deep.json`.

## Roadmap

- [x] Policy knowledge base with hybrid retrieval (BM25 + vectors) and evaluation
- [x] Agentic RAG, long-term memory, reliability, guard, observability, CI, Docker stack
- [ ] Cloud deployment, frontend
