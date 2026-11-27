# SupportPilot -- project conventions

## Language (highest priority)
- ALWAYS reply to the user in Simplified Chinese (简体中文), in every message, no matter how much English code, logs or docs are in context. Keep technical terms (API, Docker, MCP) in English.
- Everything written into the repo is English: code, comments, docstrings, commit messages, docs, prompts, mock data.

## Stack
Python 3.12 + uv, FastAPI (async, SSE), LangGraph with the Postgres checkpointer, official MCP Python SDK
(v2: `mcp.server.mcpserver.MCPServer`, `mcp.Client`), ParadeDB (Postgres + pg_search + pgvector), Redis,
pytest. The only model provider is Claude via the official `anthropic` SDK.
The production default is Sonnet (`claude-sonnet-5-5`) with automatic fallback to Haiku (`claude-haiku-5-5`). Local dev and CI set the main model to Haiku via `.env` to save cost. Routine evals are graded by Haiku; the one-off deep evaluation by Sonnet. The key comes from `ANTHROPIC_API_KEY`.

## Commands
```bash
docker compose up -d --build --wait  # full stack: Postgres + Redis + API on :8000
docker compose up -d --wait postgres redis   # databases only (API on the host)
uv sync                              # install deps
uv run supportpilot                  # run the API on :8000
uv run pytest                        # full offline test suite (needs the compose stack)
uv run pytest -m live                # tests that call real Claude
uv run ruff check . && uv run ruff format .
uv run python scripts/reset_db.py    # rebuild the dev database from db/init/*.sql
uv run python scripts/chat.py --user cust_001   # interactive client
uv run python scripts/ingest_kb.py   # (re)index kb/policies; new chunks call Haiku, others reuse kb/contexts.json
uv run supportpilot-eval compare     # full eval vs. evals/baseline.json, exit 1 on a reproducible regression
uv run supportpilot-eval compare --subset ci --baseline evals/baseline-ci.json   # the PR gate (~$0.02)
uv run supportpilot-eval retrieval-sweep    # rerankers x top_k, retrieval metrics only (no grader)
uv run supportpilot-eval run --architecture routed   # the router + specialists design (P5 comparison)
uv run python scripts/memory_bench.py --variant cached   # mem0 cost / latency / recall (~$0.1-0.3 per variant)
uv run python scripts/turn_stats.py --send  # per-turn tokens / cost / p50-p95 latency via the API, read from Langfuse
uv run --with pillow python scripts/record_demo.py   # re-record docs/demo.gif (then reset_db.py)
```
CI's eval gate reranks with `jinaai/jina-reranker-v1-tiny-en` (`RERANKER_MODEL`) against
`evals/baseline-ci.json`; regenerate that baseline with the same env var. Reported numbers use v2-m3.

## Layout
- `src/supportpilot/mcp_server/` -- MCP tools. Identity comes from request `_meta`, never from tool args.
- `src/supportpilot/tools.py` -- ToolGateway: the only place tools are executed; injects identity and
  idempotency keys, and lists the tools that need human confirmation.
- `src/supportpilot/graph/` -- LangGraph state, prompts and nodes. Default `AGENT_ARCHITECTURE=single`
  (one agent, every tool); `routed` is the router + specialists design it was compared with.
- `src/supportpilot/llm/` -- `AnthropicLLM` (real: retries, timeout, fallback model) and `RuleBasedLLM`
  (deterministic test double); `usage.py` meters tokens and cost per turn.
- `src/supportpilot/guard.py` -- prompt-injection scan of customer messages, tool results, memories.
- `src/supportpilot/longterm.py` -- mem0 long-term memory; `memory.py` -- short-term (Redis).
- `src/supportpilot/observability.py` -- Langfuse tracing (no-op without keys).
- `src/supportpilot/api/` -- FastAPI app.
- `src/supportpilot/kb/` -- policy chunking, Contextual Retrieval, fastembed models, hybrid `Retriever`.
- `src/supportpilot/evals/` -- eval runner, metrics, judges (`supportpilot-eval`); cases in `evals/golden.jsonl`
  (every case has one of seven `type`s; `turns` are run live), CI subset in `evals/subsets/ci.txt`,
  memory recall scenarios in `evals/memory.jsonl`.
- `.github/workflows/ci.yml` -- tests on push/PR; eval gate on PRs (secret `ANTHROPIC_API_KEY`).
- `kb/policies/` -- policy documents (source of truth); `kb/contexts.json` -- cached chunk contexts.
- `db/init/` -- schema and seed SQL. `tests/` -- pytest.

## Rules
- Never add a `user_id` parameter to a tool. Scope every query by the injected user id.
- Any tool that mutates customer data goes in `CONFIRMATION_REQUIRED` and takes an orchestrator-owned
  idempotency key.
- Code before `interrupt()` in a node re-runs on resume: keep it free of side effects.
- Tests must pass offline with `LLM_PROVIDER=rule_based`; tests that hit the API are marked `live`.
- Never commit `.env`.
- Every Claude call goes through `AnthropicLLM` (or records usage via `llm.usage.record`) so it is
  metered, traced and covered by retries/fallback.
- Before any eval run that calls the API, state the expected call count and cost; Ragas and the
  judge run only on the full configuration, ablations use retrieval metrics only.
- An intended behaviour change that moves eval metrics needs a new baseline: `promote` a saved run
  for `evals/baseline.json`, and re-run `baseline --subset ci --baseline evals/baseline-ci.json`.
- After editing `kb/policies/*.md`, re-run `scripts/ingest_kb.py` and commit the updated `kb/contexts.json`
  (the offline tests index the KB from it), then check `evals/golden.jsonl` source labels still hold.
- If an API-calling run is estimated to cost more than $1, stop and ask the user first.
