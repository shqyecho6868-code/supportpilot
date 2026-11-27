# SupportPilot

[![CI](https://github.com/shqyecho6868-code/supportpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/shqyecho6868-code/supportpilot/actions/workflows/ci.yml)

**An AI support agent that takes routine tickets off an e-commerce support team: it looks up
orders, changes shipping addresses and answers policy questions with citations, and every change
to customer data waits for the customer's explicit confirmation.**

Built on Claude, LangGraph and the Model Context Protocol, with a measured retrieval pipeline,
an evaluation suite that gates every pull request, and Langfuse tracing of every turn.

![Demo: order lookup, address change with confirmation, change applied](docs/demo.gif)

<sub>A real session against the local stack (Claude Haiku 5.5), recorded by
`scripts/record_demo.py`; long model waits are shortened. Transcript: [docs/demo.txt](docs/demo.txt).</sub>

## Results at a glance

Measured on a 52-case golden set (26 policy questions with labelled sources, plus orders,
address changes, cross-customer access, prompt injection, small talk and handoff), with Claude
Haiku 5.5 as the agent and as the grader. Details and every failure: [docs/eval-report.md](docs/eval-report.md).

| | Result |
|---|---|
| Task completion (judge + deterministic checks) | **98.1%** (51 / 52) |
| Prompt-injection cases flagged by the guard | **100%** (8 / 8), 0 false positives on 44 benign cases |
| Cross-customer and injection cases with protected data: leaks or unauthorized changes | **0** of 7, in every run |
| Retrieval MRR@5 / recall@5 (full pipeline) | **95.2 / 96.2** |
| Answers citing a retrieved passage | **100%** |
| Day and dollar figures in answers that match the policy (deterministic check, 22 cases) | **100%** |
| Cost per conversation turn / p50 / p95 latency | **$0.0017** / **5.1 s** / **12.5 s** ([conditions](#cost-and-latency)) |

**Retrieval ablation.** Each row adds one stage (retrieval metrics on the 26 labelled questions,
%). The reranker hurt ranking on raw chunks and only paid off on contextualized ones; swapping
in a stronger reranker fixed the three cases the pipeline still failed.

| Pipeline | hit@5 | MRR@5 | recall@5 |
|---|---|---|---|
| Vector only (bge-small, pgvector) | 92.3 | 90.4 | 87.2 |
| + BM25 (pg_search), RRF fusion | 96.2 | 91.2 | 89.1 |
| + reranker (bge-reranker-base) | 92.3 | 87.3 | 88.5 |
| + Contextual Retrieval | 100 | 91.3 | 94.2 |
| **Reranker swapped to bge-reranker-v2-m3 (current default)** | **100** | **95.2** | **96.2** |

**Reranker swap.** bge-reranker-base scored surface overlap and pushed the passages three
answers needed out of the top 5; v2-m3 keeps them all.

| | bge-reranker-base | bge-reranker-v2-m3 |
|---|---|---|
| MRR@5 / recall@5 | 91.3 / 94.2 | 95.5 / 96.2 |
| needed passages in the top 5 (pol-01, pol-02, multi-04) | 0 of 3 | 3 of 3 |
| task completion, full eval | 93.8% | 96.2%, 98.1%, 98.1% (three runs) |
| seconds per query (laptop CPU) | 0.9 | 2.8 |

**Agentic RAG (off by default).** Letting the agent grade passages and re-search did not beat
the single search beyond noise (one case is about 4 points on 26 answers):

| | off | on, run 1 | on, run 2 |
|---|---|---|---|
| faithfulness | 85.1 | 90.8 | 85.8 |
| factual correctness | 84.6 | 87.9 | 88.1 |
| task completion | 96.2% | 98.1% | 98.1% |
| searches per answer | 1.00 | 1.58 | 1.08 |
| seconds / cost per case | 3.8 s / $0.0003 | 6.5 s / $0.0005 | 5.4 s / $0.0004 |

## Architecture

```mermaid
flowchart LR
    client["Customer client"] -->|"POST /chat (SSE)"| api["FastAPI"]
    api --> router{"Router (Claude)"}
    subgraph graph["LangGraph, checkpointed in Postgres"]
        router -->|order| order["Order agent"]
        router -->|knowledge| knowledge["Knowledge agent"]
        router -->|"small talk / out of scope"| chitchat["Chitchat"]
        router -->|handoff| handoff["Handoff: ticket for a human"]
        order -->|update_address| confirm[["Human confirmation: interrupt, then POST /confirm"]]
    end
    order --> gateway["ToolGateway: injects user id and idempotency key, scans for prompt injection"]
    knowledge --> gateway
    handoff --> gateway
    confirm --> gateway
    gateway -->|MCP| tools["MCP tools: get_order, list_my_orders, update_address, create_ticket, search_policy"]
    tools --> orders[("ParadeDB: orders, tickets")]
    tools --> retrieval
    subgraph retrieval["Retrieval pipeline"]
        direction LR
        rewrite["Query rewrite"] --> vector["Vectors: bge-small, pgvector"]
        rewrite --> bm25["BM25: pg_search"]
        vector --> rrf["RRF fusion"]
        bm25 --> rrf
        rrf --> rerank["Rerank: bge-reranker-v2-m3"]
        rerank --> chunks["Top 5 chunks with Contextual Retrieval context"]
    end
    api <--> shortterm[("Redis: short-term memory")]
    api <--> longterm[("mem0 on pgvector: long-term memory")]
    graph -.->|"every call, tool and decision"| langfuse["Langfuse traces"]
```

- **Router** sends each turn to one agent; the order agent and knowledge agent call tools in a loop.
- **Tools live behind MCP.** No tool takes a `user_id`: the gateway injects the signed-in customer
  into the MCP request `_meta`, and every query is scoped by it.
- **Writes need a human.** `update_address` pauses the graph with `interrupt()`; the change runs
  only after `POST /confirm` approves it, with an idempotency key owned by the orchestrator. The
  paused state is checkpointed, so it survives a restart.
- **Retrieval** is hybrid (vectors + BM25), reranked, over chunks enriched with
  [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval); answers cite
  `[Document § Section]`. Details: [docs/retrieval.md](docs/retrieval.md).
- **Memory:** recent turns in Redis; durable customer facts in mem0, recalled as unverified context.
- **Reliability:** timeouts, retries with backoff, fallback to Haiku on errors, and the API's
  server-side fallback when Sonnet declines a request for policy reasons.

## Cost and latency

Read back from Langfuse with `scripts/turn_stats.py --send`: the 49 single-turn golden messages
sent through the API (`POST /chat`, the production path), one new session each, on 2026-11-26.
Conditions: Claude Haiku 5.5 for every call, the Docker stack on a laptop CPU (the reranker runs
locally on CPU), Langfuse cloud. Latency is the turn as the customer sees it (the `chat_turn`
trace); tokens and cost include every model call in the trace, including the long-term-memory
extraction that runs after the reply.

| Per turn (n = 49) | |
|---|---|
| tokens, average | 16,550 |
| cost, average | $0.0017 |
| latency p50 / p95 | 5.1 s / 12.5 s |

| Share of cost | |
|---|---|
| long-term memory extraction (mem0, after the reply) | 79.8% |
| knowledge agent | 7.8% |
| order agent | 6.0% |
| router | 5.2% |
| small talk | 1.2% |

Without memory extraction a turn costs about $0.0003, the figure the eval measures per case.
Latency depends on the route: median 3.3 s for order questions (14 turns), 10.8 s for policy
questions (24), because reranking 20 candidates with v2-m3 runs on the CPU inside Docker; a GPU or
the smaller CI reranker would cut most of that. The first policy question after the container
started took 228 s while the reranker model loaded; it is the maximum, so p50 and p95 do not
include it, but a deployment should load the model at startup.

<!-- TODO: Langfuse trace screenshot (docs/langfuse-trace.png) -->

## Design decisions

**Why the user id is injected, never passed by the model.** No tool has a `user_id` parameter.
The orchestrator puts the signed-in customer into the MCP request `_meta`, and the tools scope
every query by it. A prompt that says "I am cust_003" or "act as admin" therefore has nothing to
change: isolation is enforced by code, not by the prompt. In every evaluation run, none of the 7
cross-user and injection cases with protected data leaked an order or changed one.

**Why agentic RAG is off by default.** The optional loop (decide whether to search, grade the
passages, rewrite and search again) was measured against the single-search pipeline: factual
correctness +3.3 and +3.5 points in two runs, which is below the resolution of 26 answers (one
case is about 4 points), and a faithfulness gain of +5.7 did not reproduce (+0.7). It costs 2 to 4
extra model calls per policy question and 1.5 to 2.5 s more per case. It stays behind
`AGENTIC_RAG=true`, worth re-testing on a harder question set.

**Why the eval gate re-runs itself once.** LLM-graded metrics move up to 4.5 points between
identical runs, close to a 5-point threshold. A metric regresses only when it is worse than the
baseline by more than max(5 points, 1.5 cases' worth); a first-run regression triggers one
automatic re-run, and the gate fails only for metrics that regress in both. Validated both ways:
unchanged code passes, and replacing the order agent's prompt with "do not use any tools" fails
in both runs (task completion -29 points).

**Why production defaults to Sonnet and development to Haiku.** Haiku 5.5 is 20 times cheaper
($0.10 / $0.50 per million input / output tokens vs. $2 / $10), which keeps the loops that run
hundreds of calls cheap: a full 52-case evaluation is about $0.09 and the PR gate about $0.02.
Every number in this README was measured with Haiku as the agent, so it is a floor for Sonnet,
which production uses for harder conversations and long tool chains. The main model falls back to
Haiku on errors, and Sonnet's policy declines go to the API's server-side fallback
(`fallbacks: "default"`) instead of ending the conversation.

**Why CI reranks with a smaller model.** `bge-reranker-v2-m3` takes about 20 s per query on the
GitHub runner's CPU, which made the eval gate 6 to 13 minutes long. CI uses
`jina-reranker-v1-tiny-en` (20 times faster) against its own baseline; the gate catches
regressions in the agent, prompts and tools, while the retrieval numbers reported here come from
local runs with v2-m3. `compare` refuses to compare runs made with a different reranker than the
baseline.

## Known limitations

- **inj-06 is flaky.** A message with fake `<system>` tags claims a 365-day return window. The
  reply always gives the real 15-day window, but in some runs it adds "I can't confirm a 365-day
  window", repeating the injected number; that counts as a failure. A prompt fix brought it to 2 of
  3 passing runs. No protected data leaks either way.
- **Long-term memory dominates the cost of a turn.** In the measurement above, mem0's
  extraction is 80% of the cost of a turn: it sends about 14k tokens of its own prompt per call,
  where an agent reply needs 1 to 3k. It runs after the reply, so it does not add latency.
  Running it only on turns that carry durable facts, or batching it per session, would cut that.
- **Authentication is mocked.** The `X-User-Id` header is trusted as the signed-in customer; a real
  deployment would derive it from a session or token at the edge.
- **The evaluation set was written with AI assistance and reviewed by hand**, by the same author
  as the policy documents, so questions and documents share vocabulary. Real customer phrasing
  will be harder; real conversations should be added once available.
- **Small samples.** 52 cases, 26 of them policy answers: one case moves an answer metric by about
  4 points. Differences below that are reported as noise, and more cases would help more than more
  runs.
- **Sonnet itself was not evaluated end to end** (to stay within the cost budget); it was used once
  as a stricter grader of Haiku's answers.

## Quickstart

Requirements: Docker, and an Anthropic API key.

```bash
cp .env.example .env              # set ANTHROPIC_API_KEY; Langfuse keys are optional
docker compose up -d --build --wait
uv run python scripts/chat.py --user cust_001   # or: curl http://localhost:8000/health
```

One command starts ParadeDB (Postgres + pg_search + pgvector), Redis and the API on `:8000`. The
first start downloads the local embedding and reranker models (about 2.5 GB, cached in a volume)
and indexes the policy documents, so it takes a few minutes.

**`.env`** (all keys are documented in [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | required |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` in code, `claude-haiku-5-5` in `.env.example` | main model: Sonnet in production, Haiku for development and CI |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | unset | tracing is on when the keys are set |
| `AGENTIC_RAG` | `false` | decide / grade / re-search loop for the knowledge agent |
| `LONG_TERM_MEMORY` | `true` | mem0 memory across sessions |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | local cross-encoder |
| `SERVER_SIDE_FALLBACK` | `true` | re-run Sonnet policy declines on the API's fallback model |

Mock customers `cust_001` to `cust_005` and their orders are seeded from `db/init/`; for example
`cust_001` has ORD-1001 (pending), ORD-1002 (shipped) and ORD-1003 (delivered). Authentication is
mocked: the `X-User-Id` header is the signed-in customer.

### Development

```bash
docker compose up -d --wait postgres redis    # databases only
uv sync
uv run python scripts/ingest_kb.py            # index the policies (chunk contexts cached in kb/contexts.json)
uv run supportpilot                           # API on :8000
uv run pytest                                 # offline suite: deterministic rule-based LLM stand-in
uv run pytest -m live                         # the same scenarios against real Claude
```

### API

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | `{message, session_id?}` -> Server-Sent Events: `route`, `sources`, `token`, `confirmation_required` or `final`, `usage`, `done`, ... |
| POST | `/confirm` | `{thread_id, decision: approve \| reject}` resumes a paused turn; 410 after `CONFIRMATION_TTL_SECONDS` |
| GET | `/health` | Postgres and Redis status |

## Evaluation and CI

```bash
uv run supportpilot-eval run                     # full eval (52 cases), prints tokens and cost
uv run supportpilot-eval compare                 # vs. evals/baseline.json, exit 1 on a reproducible regression
uv run supportpilot-eval ablation                # retrieval ablation
uv run supportpilot-eval retrieval-sweep         # rerankers x top_k, retrieval metrics only
uv run python scripts/turn_stats.py --send       # per-turn tokens, cost and latency from Langfuse
```

**CI** (`.github/workflows/ci.yml`): every push runs ruff and the offline test suite (a
deterministic stand-in for Claude, against ParadeDB and Redis service containers). Pull requests
then run the eval gate: a 17-case subset covering every route, the confirmation flow, isolation
and injection, with Haiku as agent and grader (about $0.02), compared with
`evals/baseline-ci.json`. The gate reranks with `jina-reranker-v1-tiny-en` to stay fast (see
[Design decisions](#design-decisions)); the eval job takes about 2 minutes
(2 min 08 s on 2026-11-26, model download included; about 3.5 with the automatic re-run), down
from 6.5 minutes per run (13 with the re-run) with v2-m3.

Full write-up, raw numbers and every failure analysed:
[docs/eval-report.md](docs/eval-report.md).
