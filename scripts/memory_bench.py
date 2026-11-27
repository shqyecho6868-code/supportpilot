"""Long-term memory (mem0) cost, latency and cross-session recall, per extraction strategy.

    uv run python scripts/memory_bench.py --variant background   # -> evals/results/memory-background.json
    uv run python scripts/memory_bench.py --report               # table of every saved variant

Runs in-process through the production path (FastAPI ``POST /chat`` and ``POST /confirm`` ->
router, agents, MCP tools, retrieval, guard, mem0) against the eval database, with Claude Haiku
for every call. The workload is every golden case (each live message, one session per case,
confirmations approved) plus the memory scenarios in evals/memory.jsonl, in which a customer
mentions a durable fact. Measured per turn:

* latency: from the request until the reply (``final``) or the confirmation prompt, plus the
  ``/confirm`` call that applies a change, as the customer waits for it;
* cost: every model call of the run, extraction included (background calls are drained first);
* recall: after all extractions finished, a new-session memory search (what the next session's
  first turn recalls) for each scenario must return its fact.

Variants:
    sync               extraction on every turn, awaited before the reply (the naive placement)
    background         extraction on every turn, after the reply (the P3/P4 production setting)
    gated              background, only turns that can carry a durable fact (rule gate)
    cached             background, every turn, mem0's fixed prompt cached
    gated_cached       gated + cached
"""

import argparse
import asyncio
import datetime as dt
import json
import statistics
import sys
import time
import uuid
from typing import Any

import httpx
import psycopg

from supportpilot.config import get_settings
from supportpilot.db import reset_database
from supportpilot.evals.cases import EVALS_DIR, RESULTS_DIR, load_cases
from supportpilot.evals.runner import prepare_database
from supportpilot.llm.usage import PROCESS_METER
from supportpilot.longterm import EXTRACTION_CALL

VARIANTS: dict[str, dict[str, Any]] = {
    "sync": {"long_term_memory_gate": "all", "long_term_memory_cache": False, "long_term_memory_sync": True},
    "background": {
        "long_term_memory_gate": "all",
        "long_term_memory_cache": False,
        "long_term_memory_sync": False,
    },
    "gated": {
        "long_term_memory_gate": "rules",
        "long_term_memory_cache": False,
        "long_term_memory_sync": False,
    },
    "cached": {
        "long_term_memory_gate": "all",
        "long_term_memory_cache": True,
        "long_term_memory_sync": False,
    },
    "gated_cached": {
        "long_term_memory_gate": "rules",
        "long_term_memory_cache": True,
        "long_term_memory_sync": False,
    },
}
MODEL = "claude-haiku-5-5"
REDIS_DB = 13
BENCH_DB = "supportpilot_memory_bench"  # separate from the eval database


def load_scenarios() -> list[dict[str, Any]]:
    path = EVALS_DIR / "memory.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _customer(scenario_id: str) -> str:
    return f"cust_{scenario_id.replace('-', '_')}"


async def seed_scenario_customers(url: str, scenarios: list[dict[str, Any]]) -> None:
    """One fresh customer per memory scenario, so each recall sees only that scenario's memories."""
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        for s in scenarios:
            cid = _customer(s["id"])
            await conn.execute(
                "INSERT INTO customers (id, name, email) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (cid, f"Bench {s['id']}", f"{cid}@example.com"),
            )
            if s.get("order"):
                await conn.execute(
                    "INSERT INTO orders (id, customer_id, status, items, total_cents, shipping_address, "
                    "placed_at) VALUES (%s, %s, 'pending', %s, 1999, %s, now()) ON CONFLICT DO NOTHING",
                    (
                        "ORD-9022",
                        cid,
                        json.dumps([{"sku": "SKU-MUG-02", "name": "Ceramic Coffee Mug", "qty": 1}]),
                        json.dumps(
                            {
                                "street": "1 Main St",
                                "city": "Albany",
                                "state": "NY",
                                "postal_code": "12207",
                                "country": "US",
                            }
                        ),
                    ),
                )


async def turn(client: httpx.AsyncClient, user_id: str, session_id: str, message: str) -> dict[str, Any]:
    """One customer turn; approves a confirmation like the customer would. Returns latency and reply."""
    started = time.monotonic()
    result: dict[str, Any] = {"message": message}
    async with client.stream(
        "POST", "/chat", json={"message": message, "session_id": session_id}, headers={"X-User-Id": user_id}
    ) as resp:
        event = None
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event in ("final", "confirmation_required", "error"):
                data = json.loads(line.split(":", 1)[1])
                result.setdefault("seconds", time.monotonic() - started)
                result[event] = data
    if conf := result.get("confirmation_required"):
        confirmed = time.monotonic()
        r = await client.post(
            "/confirm",
            json={"thread_id": conf["thread_id"], "decision": "approve"},
            headers={"X-User-Id": user_id},
        )
        result["seconds"] += time.monotonic() - confirmed
        result["final"] = {"reply": r.json().get("reply")}
    result["seconds"] = round(result.get("seconds", time.monotonic() - started), 3)
    return result


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def run(variant: str, limit: int | None = None) -> dict[str, Any]:
    from supportpilot.api.app import create_app

    base = get_settings()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    url = await prepare_database(base.database_url, base.fast_model, base.embedding_model, name=BENCH_DB)
    settings = base.model_copy(
        update={
            "database_url": url,
            "redis_url": f"{base.redis_url.rsplit('/', 1)[0]}/{REDIS_DB}",
            "anthropic_model": MODEL,
            "fast_model": MODEL,
            "fallback_model": MODEL,
            "long_term_memory": True,
            "mem0_collection": f"mem0_bench_{variant}_{stamp}",
            "mem0_history_db": f".mem0/bench-{variant}-{stamp}.db",
            "langfuse_enabled": False,
            "conversation_token_limit": 0,
            **VARIANTS[variant],
        }
    )
    scenarios = load_scenarios()
    workload: list[tuple[str, str, list[str]]] = [
        (c.id, c.user_id, [m["content"] for m in c.history if m["role"] == "user"] + list(c.messages))
        for c in load_cases()
    ]
    if limit:
        workload, scenarios = workload[:limit], scenarios[:limit]
    workload += [(s["id"], _customer(s["id"]), s["messages"]) for s in scenarios]

    app = create_app(settings)
    turns: list[dict[str, Any]] = []
    async with app.router.lifespan_context(app):
        ctx = app.state.ctx
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench", timeout=600) as client:
            await reset_database(url)
            await seed_scenario_customers(url, scenarios)
            await turn(client, "cust_001", "warmup", "How long is the return window?")  # loads the reranker
            await ctx.background.drain(300)
            before = len(PROCESS_METER.calls)
            for case_id, user_id, messages in workload:
                await reset_database(url)  # address changes in one case must not leak into the next
                await seed_scenario_customers(url, scenarios)
                session = f"bench-{case_id}-{uuid.uuid4().hex[:6]}"
                for message in messages:
                    t = await turn(client, user_id, session, message)
                    turns.append({"case": case_id, **t})
                    if len(turns) >= 5 and all("error" in x for x in turns[-5:]):
                        sys.exit("ABORTED: 5 turns in a row failed (credit, network?); nothing saved")
                    print(f"  {case_id:9s} {t['seconds']:6.2f}s", flush=True)
            await ctx.background.drain(600)
            calls = PROCESS_METER.calls[before:]

            recall = []
            for s in scenarios:
                memories = await ctx.long_term.recall(_customer(s["id"]), s["probe"])
                text = " ".join(memories).lower()
                hit = all(any(alt in text for alt in group) for group in s["expect"])
                recall.append({"id": s["id"], "hit": hit, "memories": memories})

    seconds = [t["seconds"] for t in turns]
    extraction = [c for c in calls if c.name == EXTRACTION_CALL]
    total_cost = sum(c.cost()["total"] for c in calls)
    extraction_cost = sum(c.cost()["total"] for c in extraction)
    return {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "variant": variant,
        "settings": VARIANTS[variant],
        "model": MODEL,
        "turns": len(turns),
        "sessions": len(workload),
        "cost_usd_per_turn": total_cost / len(turns),
        "extraction_cost_usd_per_turn": extraction_cost / len(turns),
        "extraction_share": extraction_cost / total_cost if total_cost else None,
        "extraction_calls": len(extraction),
        "extraction_cache_read_tokens": sum(c.cache_read_tokens for c in extraction),
        "extraction_input_tokens": sum(
            c.input_tokens + c.cache_write_tokens + c.cache_read_tokens for c in extraction
        ),
        "latency_p50_s": round(statistics.median(seconds), 2),
        "latency_p95_s": round(_pct(seconds, 0.95), 2),
        # Turns that ended in an error (network, API) invalidate the run: rerun it.
        "errored_turns": [t["case"] for t in turns if "error" in t or not t.get("final")],
        "recall_hits": sum(r["hit"] for r in recall),
        "recall_n": len(recall),
        "recall": recall,
        "turn_log": turns,
    }


def report() -> None:
    rows = []
    for variant in VARIANTS:
        path = RESULTS_DIR / f"memory-{variant}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    print(f"{'variant':14s} {'turns':>5s} {'$/turn':>9s} {'mem0 $/turn':>11s} {'mem0 share':>10s} "
          f"{'calls':>5s} {'p50':>6s} {'p95':>6s} {'recall':>7s}")  # fmt: skip
    for r in rows:
        print(
            f"{r['variant']:14s} {r['turns']:5d} {r['cost_usd_per_turn']:9.5f} "
            f"{r['extraction_cost_usd_per_turn']:11.5f} "
            f"{r['extraction_share'] * 100:9.1f}% {r['extraction_calls']:5d} {r['latency_p50_s']:6.2f} "
            f"{r['latency_p95_s']:6.2f} {r['recall_hits']:3d}/{r['recall_n']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int, help="debug: only the first N cases and scenarios, not saved")
    args = parser.parse_args()
    if args.report or not args.variant:
        report()
        return
    if not get_settings().anthropic_api_key:
        sys.exit("ANTHROPIC_API_KEY is required")
    result = asyncio.run(run(args.variant, args.limit))
    if args.limit:
        print(json.dumps({k: v for k, v in result.items() if k not in ("turn_log",)}, indent=2, default=str))
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    if result["errored_turns"]:
        print(
            f"INVALID: {len(result['errored_turns'])} turns failed: {result['errored_turns']}",
            file=sys.stderr,
        )
        sys.exit(1)
    path = RESULTS_DIR / f"memory-{args.variant}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {path}")
    report()


if __name__ == "__main__":
    main()
