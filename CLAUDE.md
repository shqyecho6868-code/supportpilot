# SupportPilot -- project conventions

## Language
- Everything in the repo is English: code, comments, docstrings, commit messages, docs, prompts, mock data.
- Talk to the user in Chinese (keep technical terms such as Bean, Docker, API in English).

## Stack
Python 3.12 + uv, FastAPI (async, SSE), LangGraph with the Postgres checkpointer, official MCP Python SDK
(v2: `mcp.server.mcpserver.MCPServer`, `mcp.Client`), ParadeDB (Postgres + pg_search + pgvector), Redis,
pytest. The only model provider is Claude via the official `anthropic` SDK; the main model is Haiku
(`claude-haiku-5-5`); routine evaluation is graded by Haiku and the one-off deep evaluation by Sonnet
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
uv run python scripts/ingest_kb.py   # (re)index kb/policies; new chunks call Haiku, others reuse kb/contexts.json
uv run supportpilot-eval compare     # full eval vs. evals/baseline.json, non-zero exit on a >5pp regression
```

## Layout
- `src/supportpilot/mcp_server/` -- MCP tools. Identity comes from request `_meta`, never from tool args.
- `src/supportpilot/tools.py` -- ToolGateway: the only place tools are executed; injects identity and
  idempotency keys, and lists the tools that need human confirmation.
- `src/supportpilot/graph/` -- LangGraph state, prompts and nodes.
- `src/supportpilot/llm/` -- `AnthropicLLM` (real) and `RuleBasedLLM` (deterministic test double).
- `src/supportpilot/api/` -- FastAPI app.
- `src/supportpilot/kb/` -- policy chunking, Contextual Retrieval, fastembed models, hybrid `Retriever`.
- `src/supportpilot/evals/` -- eval runner, metrics, judges (`supportpilot-eval`); cases in `evals/golden.jsonl`.
- `kb/policies/` -- policy documents (source of truth); `kb/contexts.json` -- cached chunk contexts.
- `db/init/` -- schema and seed SQL. `tests/` -- pytest.

## Rules
- Never add a `user_id` parameter to a tool. Scope every query by the injected user id.
- Any tool that mutates customer data goes in `CONFIRMATION_REQUIRED` and takes an orchestrator-owned
  idempotency key.
- Code before `interrupt()` in a node re-runs on resume: keep it free of side effects.
- Tests must pass offline with `LLM_PROVIDER=rule_based`; tests that hit the API are marked `live`.
- Never commit `.env`.
- After editing `kb/policies/*.md`, re-run `scripts/ingest_kb.py` and commit the updated `kb/contexts.json`
  (the offline tests index the KB from it), then check `evals/golden.jsonl` source labels still hold.
