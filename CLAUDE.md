# SupportPilot -- project conventions

## Language
- Everything in the repo is English: code, comments, docstrings, commit messages, docs, prompts, mock data.
- Talk to the user in Chinese (keep technical terms such as Bean, Docker, API in English).

## Stack
Python 3.12 + uv, FastAPI (async, SSE), LangGraph with the Postgres checkpointer, official MCP Python SDK
(v2: `mcp.server.mcpserver.MCPServer`, `mcp.Client`), ParadeDB (Postgres + pg_search + pgvector), Redis,
pytest. The only model provider is Claude via the official `anthropic` SDK; the main model is Sonnet
(`claude-sonnet-5-5`), and the key comes from `ANTHROPIC_API_KEY`.

## Commands
```bash
docker compose up -d --wait          # start Postgres + Redis
uv sync                              # install deps
uv run supportpilot                  # run the API on :8000
uv run pytest                        # full offline test suite (needs the compose stack)
uv run pytest -m live                # tests that call real Claude
uv run ruff check . && uv run ruff format .
uv run python scripts/reset_db.py    # rebuild the dev database from db/init/*.sql
uv run python scripts/chat.py --user cust_001   # interactive client
```

## Layout
- `src/supportpilot/mcp_server/` -- MCP tools. Identity comes from request `_meta`, never from tool args.
- `src/supportpilot/tools.py` -- ToolGateway: the only place tools are executed; injects identity and
  idempotency keys, and lists the tools that need human confirmation.
- `src/supportpilot/graph/` -- LangGraph state, prompts and nodes.
- `src/supportpilot/llm/` -- `AnthropicLLM` (real) and `RuleBasedLLM` (deterministic test double).
- `src/supportpilot/api/` -- FastAPI app.
- `db/init/` -- schema and seed SQL. `tests/` -- pytest.

## Rules
- Never add a `user_id` parameter to a tool. Scope every query by the injected user id.
- Any tool that mutates customer data goes in `CONFIRMATION_REQUIRED` and takes an orchestrator-owned
  idempotency key.
- Code before `interrupt()` in a node re-runs on resume: keep it free of side effects.
- Tests must pass offline with `LLM_PROVIDER=rule_based`; tests that hit the API are marked `live`.
- Never commit `.env`.
