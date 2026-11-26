# SupportPilot

An e-commerce customer-support agent built on Claude, LangGraph and the Model Context Protocol.

> Status: work in progress. This stage covers the order-support core: tools, orchestration,
> human-in-the-loop confirmation and short-term memory. Retrieval, evaluation and
> observability come next.

## Features (so far)

- **Order tools over MCP** -- `get_order`, `list_my_orders`, `update_address`, `create_ticket`.
  The customer's identity is injected by the orchestrator through MCP request `_meta`; no tool
  takes a `user_id` argument, so the model cannot query another customer's data.
- **Idempotent writes** -- `update_address` only works on `pending` orders and takes an
  idempotency key; repeating a call with the same key applies the change once.
- **LangGraph orchestration** -- a router sends each turn to `order_agent`, `knowledge_agent`
  (placeholder) or `handoff` (opens a ticket for a human).
- **Human-in-the-loop** -- address changes pause with `interrupt()` until the customer approves
  or rejects via `POST /confirm`. State is checkpointed to Postgres, so a paused turn survives
  a server restart.
- **Short-term memory** -- recent turns per session live in Redis with a sliding TTL.
- **Streaming API** -- `POST /chat` streams Server-Sent Events.

## Architecture

```
client --SSE--> FastAPI /chat ----> LangGraph (Postgres checkpointer)
                                     router -> order_agent <-> confirm (interrupt)
                                            -> knowledge_agent
                                            -> handoff
                                     order_agent --> ToolGateway --MCP--> order tools --> Postgres
                       /confirm ---> resume paused thread
                       Redis: conversation history, session ownership, confirm locks
```

## Quick start

```bash
cp .env.example .env              # add ANTHROPIC_API_KEY
docker compose up -d --wait       # ParadeDB (Postgres + pg_search + pgvector) and Redis
uv sync
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
| POST   | `/chat`    | `{message, session_id?}` -> SSE: `meta`, `route`, `token`, `action_result`, `confirmation_required` or `final`, `error`, `done` |
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

## Roadmap

- [ ] Policy knowledge base with hybrid retrieval (BM25 + vectors) and evaluation
- [ ] Agentic RAG, long-term memory, reliability, observability, CI
