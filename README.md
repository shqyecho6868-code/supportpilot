# SupportPilot

An e-commerce customer-support agent built on Claude, LangGraph and the Model Context Protocol.

> Status: work in progress. Done so far: the order-support core (tools, orchestration,
> human-in-the-loop confirmation, short-term memory), a policy knowledge base with hybrid
> retrieval, and an evaluation suite. Observability and CI come next.

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
- **Short-term memory** -- recent turns per session live in Redis with a sliding TTL.
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
                       Redis: conversation history, session ownership, confirm locks
```

## Quick start

```bash
cp .env.example .env              # add ANTHROPIC_API_KEY
docker compose up -d --wait       # ParadeDB (Postgres + pg_search + pgvector) and Redis
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

## API

| Method | Path       | Description |
|--------|------------|-------------|
| POST   | `/chat`    | `{message, session_id?}` -> SSE: `meta`, `route`, `sources`, `token`, `action_result`, `confirmation_required` or `final`, `error`, `done` |
| POST   | `/confirm` | `{thread_id, decision: approve \| reject}` -> resumes a paused turn |
| GET    | `/health`  | Postgres and Redis status |

Authentication is mocked: the `X-User-Id` header carries the customer id.

## Mock data

| Customer | Orders |
|----------|--------|
| `cust_001` Alice Johnson | ORD-1001 pending, ORD-1002 shipped, ORD-1003 delivered |
| `cust_002` Bob Smith     | ORD-2001 pending, ORD-2002 cancelled, ORD-2003 delivered |
| `cust_003` Carol Nguyen  | ORD-3001 shipped, ORD-3002 pending |
| `cust_004` David Garcia  | ORD-4001 delivered, ORD-4002 cancelled, ORD-4003 pending |

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
uv run supportpilot-eval compare    # full eval vs. baseline; exits 1 if any metric drops > 5 points
```

## Roadmap

- [x] Policy knowledge base with hybrid retrieval (BM25 + vectors) and evaluation
- [ ] Agentic RAG, long-term memory, reliability, observability, CI
