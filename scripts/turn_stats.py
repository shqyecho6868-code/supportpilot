"""Per-turn cost and latency, read back from Langfuse.

    uv run python scripts/turn_stats.py --send            # send the golden single-turn messages, then report
    uv run python scripts/turn_stats.py --minutes 30      # report on chat turns of the last 30 minutes

``--send`` posts every golden case without history to the running API (one new session each,
confirmations left pending), so the turns go through the production path: router, agents, MCP
tools, retrieval, guard and background long-term-memory extraction. The report then reads the
``chat_turn`` traces of that window from Langfuse: latency of the turn as the customer sees it
(the root observation, p50 / p95), and tokens and cost of every generation in the trace, which
includes the mem0 extraction that runs after the reply. Writes evals/results/turn-stats.json.
"""

import argparse
import datetime as dt
import json
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx
from langfuse import Langfuse

from supportpilot.config import get_settings
from supportpilot.evals.cases import RESULTS_DIR, load_cases


def send(url: str) -> int:
    cases = [c for c in load_cases() if not c.history]
    with httpx.Client(base_url=url, timeout=180) as client:
        for case in cases:
            body = {"message": case.message, "session_id": f"stats-{uuid.uuid4().hex[:8]}"}
            with client.stream("POST", "/chat", json=body, headers={"X-User-Id": case.user_id}) as resp:
                for _ in resp.iter_lines():
                    pass
            print(f"  sent {case.id}", flush=True)
    return len(cases)


def _all(fetch, **kw) -> list[dict]:
    out, cursor = [], None
    while True:
        page = fetch(limit=100, cursor=cursor, **kw)
        out += [o.dict() for o in page.data]
        cursor = getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return out


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def report(since: dt.datetime, until: dt.datetime, environment: str) -> dict:
    s = get_settings()
    lf = Langfuse(
        public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, base_url=s.langfuse_base_url
    )
    get = lf.api.observations.get_many
    window = dict(environment=environment, from_start_time=since, to_start_time=until)
    turns = _all(get, name="chat_turn", fields="core,basic,metrics,metadata", **window)
    trace_ids = {t["traceId"] for t in turns}
    # Generations may start after the turn ended (background memory extraction): widen the window.
    gens = _all(
        get,
        type="GENERATION",
        fields="core,basic,model,usage",
        environment=environment,
        from_start_time=since,
        to_start_time=until + dt.timedelta(minutes=5),
    )
    gens = [g for g in gens if g["traceId"] in trace_ids]
    tokens, cost = defaultdict(int), defaultdict(float)
    by_name: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
    models = set()
    for g in gens:
        t, c = int(g.get("totalUsage") or 0), float(g.get("totalCost") or 0.0)
        tokens[g["traceId"]] += t
        cost[g["traceId"]] += c
        row = by_name[g["name"]]
        row["calls"] += 1
        row["tokens"] += t
        row["cost"] += c
        models.add(g.get("model"))
    latencies = [t["latency"] for t in turns if t.get("latency") is not None]
    total_cost = sum(cost.values()) or 1e-12
    n = len(turns)
    return {
        "window": [since.isoformat(timespec="seconds"), until.isoformat(timespec="seconds")],
        "environment": environment,
        "models": sorted(m for m in models if m),
        "turns": n,
        "avg_tokens_per_turn": round(sum(tokens.values()) / n) if n else None,
        "avg_cost_per_turn_usd": round(sum(cost.values()) / n, 6) if n else None,
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(_pct(latencies, 0.95), 2) if latencies else None,
        "cost_share": {
            name: {**row, "share": round(row["cost"] / total_cost, 3)}
            for name, row in sorted(by_name.items(), key=lambda kv: -kv[1]["cost"])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="send the golden single-turn messages first")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--minutes", type=float, default=30, help="window to report on without --send")
    parser.add_argument("--environment", default="docker", help="APP_ENVIRONMENT of the API")
    args = parser.parse_args()
    if args.send:
        since = dt.datetime.now(dt.UTC)
        print(f"sent {send(args.url)} turns; waiting for memory extraction and Langfuse ingestion")
        until = dt.datetime.now(dt.UTC)
        time.sleep(45)
    else:
        until = dt.datetime.now(dt.UTC)
        since = until - dt.timedelta(minutes=args.minutes)
    result = report(since, until, args.environment)
    print(json.dumps(result, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    Path(RESULTS_DIR / "turn-stats.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
