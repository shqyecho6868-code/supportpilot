"""SupportPilot evaluation CLI.

    uv run supportpilot-eval ablation          # retrieval ablation -> evals/results/ablation.json
    uv run supportpilot-eval run               # full eval with the configured pipeline
    uv run supportpilot-eval baseline          # full eval, then write evals/baseline.json
    uv run supportpilot-eval compare           # full eval, compare with the baseline; on a regression
                                               # of more than 5 points re-run once, exit 1 only if
                                               # the same metric regresses in both runs
    uv run supportpilot-eval retrieval-sweep   # rerankers x top_k, retrieval metrics only (no grader)
    uv run supportpilot-eval promote evals/results/latest.json   # write a baseline from a saved run

    --subset ci --baseline evals/baseline-ci.json   # the ~15-case PR gate used in CI
    --agentic on|off                                # override AGENTIC_RAG for this run
    --architecture routed                           # router + specialist agents (the P1-P4 design,
                                                    # compared in docs/eval-report.md)

Needs the compose stack and ANTHROPIC_API_KEY. The model under test and, for routine runs, the
judge and Ragas default to Claude Haiku. The one-off deep evaluation is graded by Sonnet:

    uv run supportpilot-eval ablation --judge-model claude-sonnet-5-5
The API no longer accepts sampling parameters (temperature), so LLM calls made by the eval
run with thinking off and low effort for the most stable output available; LLM-graded
metrics still vary a little run to run.
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from supportpilot.config import Settings, get_settings
from supportpilot.db import create_pool
from supportpilot.evals.cases import BASELINE_FILE, CASE_TYPES, RESULTS_DIR, Case, load_cases, load_subset
from supportpilot.evals.judges import EvalAborted, RagasScorer, TaskJudge
from supportpilot.evals.metrics import (
    compare,
    mean,
    persistent_regressions,
    regressed_metrics,
    retrieval_scores,
    tools_ok,
)
from supportpilot.evals.numbers import check_numbers
from supportpilot.evals.runner import AgentRunner, answer_with, dumps, prepare_database, rewrite_for_retrieval
from supportpilot.kb import RetrievalConfig, Retriever
from supportpilot.llm.anthropic_llm import AnthropicLLM
from supportpilot.llm.usage import PROCESS_METER
from supportpilot.observability import configure_tracing

log = logging.getLogger("supportpilot.evals")

DEFAULT_MODEL = "claude-haiku-5-5"  # model under test
# Routine runs (baseline / compare / CI) are graded by Haiku to keep them cheap. The one-off deep
# evaluation passes --judge-model claude-sonnet-5-5 so a stronger model grades the weaker one;
# its results live in evals/baseline-deep.json and are never compared against Haiku-graded runs.
DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"
EVAL_REDIS_DB = 14

# Ablation tiers: each adds one stage to the previous one. "bm25_only" is a diagnostic row.
ABLATION_TIERS: list[tuple[str, str, dict[str, bool]]] = [
    (
        "vector",
        "Vector only (bge-small + pgvector)",
        dict(vector=True, bm25=False, rerank=False, contextual=False),
    ),
    (
        "hybrid",
        "+ BM25 (pg_search), RRF fusion",
        dict(vector=True, bm25=True, rerank=False, contextual=False),
    ),
    ("rerank", "+ bge-reranker-base", dict(vector=True, bm25=True, rerank=True, contextual=False)),
    (
        "contextual",
        "+ Contextual Retrieval (full)",
        dict(vector=True, bm25=True, rerank=True, contextual=True),
    ),
    ("bm25_only", "BM25 only (diagnostic)", dict(vector=False, bm25=True, rerank=False, contextual=False)),
]


async def gather_limited(factories: Iterable[Callable[[], Awaitable[Any]]], limit: int) -> list[Any]:
    sem = asyncio.Semaphore(limit)

    async def run(factory):
        async with sem:
            return await factory()

    return await asyncio.gather(*(run(f) for f in factories))


def _redis_url(settings: Settings) -> str:
    return f"{settings.redis_url.rsplit('/', 1)[0]}/{EVAL_REDIS_DB}"


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


async def retrieval_eval(
    retriever: Retriever, cfg: RetrievalConfig, cases: list[Case], queries: dict[str, str]
):
    per_case = {}
    for case in cases:
        hits = await retriever.search(queries[case.id], cfg)
        per_case[case.id] = {
            "ranked": [h.chunk_id for h in hits],
            **retrieval_scores(case, [h.chunk_id for h in hits]),
        }
    metrics = {
        "retrieval.hit_rate@5": mean([r["hit"] for r in per_case.values()]),
        "retrieval.mrr": mean([r["rr"] for r in per_case.values()]),
        "retrieval.recall@5": mean([r["recall"] for r in per_case.values()]),
    }
    return metrics, per_case


def _llm(args, settings: Settings) -> AnthropicLLM:
    return AnthropicLLM(
        args.model,
        settings.anthropic_api_key,
        fast_model=settings.fast_model,
        fallback_model=settings.fallback_model,
        server_fallback=settings.server_side_fallback,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _agentic(args, settings: Settings) -> bool:
    choice = getattr(args, "agentic", None)
    return settings.agentic_rag if choice in (None, "config") else choice == "on"


def _question(case: Case, query: str) -> str:
    """The question Ragas sees: a follow-up needs its rewritten, standalone form."""
    return query if case.history else case.message


async def cmd_ablation(args, settings: Settings) -> int:
    cases = [c for c in load_cases() if c.is_knowledge]
    url = await prepare_database(settings.database_url, settings.fast_model, settings.embedding_model)
    llm = _llm(args, settings)
    scorer = RagasScorer(args.judge_model, settings.embedding_model, settings.anthropic_api_key)
    pool = create_pool(url, max_size=8)
    await pool.open(wait=True)
    retriever = Retriever(pool)
    base = RetrievalConfig.from_settings(settings)
    queries = {c.id: await rewrite_for_retrieval(llm, c) for c in cases}
    rows, details = [], {}
    backend = await retriever.bm25_backend(base)
    try:
        for key, label, flags in ABLATION_TIERS:
            cfg = base.with_(**flags)
            metrics, per_case = await retrieval_eval(retriever, cfg, cases, queries)
            if not args.retrieval_only:

                async def one(case: Case, cfg=cfg):
                    hits, reply = await answer_with(llm, retriever, cfg, case, queries[case.id])
                    scores = await scorer.score(
                        _question(case, queries[case.id]), reply, [h["text"] for h in hits], case.reference
                    )
                    return case.id, reply, scores

                answered = await gather_limited([lambda c=c: one(c) for c in cases], args.concurrency)
                for cid, reply, scores in answered:
                    per_case[cid].update(reply=reply, **scores)
                metrics["answer.faithfulness"] = mean([s["faithfulness"] for _, _, s in answered])
                metrics["answer.factual_correctness"] = mean(
                    [s["factual_correctness"] for _, _, s in answered]
                )
                metrics["answer.answer_relevancy"] = mean([s["answer_relevancy"] for _, _, s in answered])
            rows.append({"tier": key, "label": label, "config": flags, **metrics})
            details[key] = per_case
            print(
                f"{label:42s} " + "  ".join(f"{k.split('.')[-1]}={_pct(v)}" for k, v in metrics.items()),
                flush=True,
            )
    finally:
        await pool.close()

    out = {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "models": {
            "answer": args.model,
            "query_rewrite": settings.fast_model,
            "ragas_judge": args.judge_model,
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "chunk_context": settings.fast_model,
        },
        "bm25_backend": backend,
        "queries": queries,
        "rows": rows,
        "details": details,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ablation.json").write_text(dumps(out) + "\n")
    print(f"\nwrote {RESULTS_DIR / 'ablation.json'}")
    return 0


async def full_eval(args, settings: Settings) -> dict[str, Any]:
    cases = _select_cases(args)
    timings: dict[str, float] = {}
    clock = time.monotonic()

    def lap(stage: str) -> None:
        nonlocal clock
        timings[stage] = round(time.monotonic() - clock, 1)
        clock = time.monotonic()

    url = await prepare_database(settings.database_url, settings.fast_model, settings.embedding_model)
    lap("prepare_db")
    llm = _llm(args, settings)
    cfg = RetrievalConfig.from_settings(settings)
    agentic = _agentic(args, settings)

    # 1. Retrieval metrics on the labelled knowledge cases (independent of routing).
    knowledge = [c for c in cases if c.is_knowledge]
    pool = create_pool(url, max_size=4)
    await pool.open(wait=True)
    try:
        queries = {c.id: await rewrite_for_retrieval(llm, c) for c in knowledge}
        retrieval_metrics, retrieval_details = await retrieval_eval(Retriever(pool), cfg, knowledge, queries)
    finally:
        await pool.close()
    lap("retrieval")

    # 2. Every case through the real graph (sequential: each case resets the order tables).
    runner = AgentRunner(
        url,
        _redis_url(settings),
        llm,
        cfg,
        query_rewrite=settings.retrieval_query_rewrite,
        agentic_rag=agentic,
        injection_guard=settings.injection_guard,
        architecture=args.architecture,
        trace_tags=tuple(args.trace_tags),
        run_id=args.run_id,
    )
    records = []
    for case in cases:
        record = await runner.run_case(case)
        records.append(record)
        print(
            f"  {case.id:9s} route={record['route']!s:13s} tools={[t['name'] for t in record['tools']]}",
            flush=True,
        )
    lap("agent_cases")

    # 3. LLM grading: task completion for every case, Ragas for knowledge answers.
    judge = TaskJudge(args.judge_model, settings.anthropic_api_key)
    scorer = RagasScorer(args.judge_model, settings.embedding_model, settings.anthropic_api_key)
    by_id = {c.id: c for c in cases}

    async def grade(record):
        case = by_id[record["id"]]
        if record["error"]:
            return {"pass": False, "reason": record["error"]}
        return await judge.grade(
            message=case.message, history=list(case.history), reference=case.reference, record=record
        )

    verdicts = await gather_limited([lambda r=r: grade(r) for r in records], args.concurrency)

    async def ragas(record):
        case = by_id[record["id"]]
        if not record.get("retrieval") or not record["retrieval"]["results"]:
            return None
        contexts = [h["text"] for h in record["retrieval"]["results"]]
        question = _question(case, record["retrieval"]["query"])
        return await scorer.score(question, record["reply"], contexts, case.reference)

    answered = [r for r in records if by_id[r["id"]].is_knowledge]
    if args.skip_ragas:
        ragas_scores = [None] * len(answered)
    else:
        ragas_scores = await gather_limited([lambda r=r: ragas(r) for r in answered], args.concurrency)
    lap("grading")
    print("  timings (s): " + ", ".join(f"{k} {v}" for k, v in timings.items()), flush=True)

    # 4. Per-case checks and aggregate metrics.
    for record, verdict in zip(records, verdicts, strict=True):
        case = by_id[record["id"]]
        called = [t["name"] for t in record["tools"]]
        record["judge"] = verdict
        earlier_routes = (record.get("routes") or [])[: len(case.turn_routes)]
        record["route_ok"] = record["route"] in case.expected_route and all(
            route in accepted for route, accepted in zip(earlier_routes, case.turn_routes, strict=False)
        )
        record["tools_ok"] = tools_ok(case, called)
        checks = _hard_checks(case, record)
        record["checks"] = checks
        record["task_complete"] = verdict["pass"] and all(checks.values())
        if case.is_security:
            record["blocked"] = checks.get("no_secrets", True) and checks.get("protected_unchanged", True)
    for record, scores in zip(answered, ragas_scores, strict=True):
        record["ragas"] = scores

    numeric = [r for r in records if by_id[r["id"]].expect_numbers]
    security = [r for r in records if by_id[r["id"]].is_security]
    no_ticket_expected = [r for r in records if by_id[r["id"]].expect_ticket is False]
    injection = [r for r in records if by_id[r["id"]].category == "prompt_injection"]
    benign = [r for r in records if by_id[r["id"]].category not in ("prompt_injection",)]

    def flagged(r: dict[str, Any]) -> bool:
        flags = r.get("injection_flags") or {}
        return bool(flags.get("user_input") or flags.get("tool_results"))

    # The passages the agent actually answered from (all rounds in agentic mode), vs. the labels.
    agent_retrieval = []
    for r in answered:
        hits = (r.get("retrieval") or {}).get("results") or []
        if not (r.get("retrieval") or {}).get("skipped"):
            agent_retrieval.append(
                retrieval_scores(by_id[r["id"]], [h["chunk_id"] for h in hits], k=len(hits) or 1)
            )
    metrics = {
        **{k: v for k, v in retrieval_metrics.items()},
        "answer.faithfulness": mean([r["ragas"]["faithfulness"] for r in answered if r.get("ragas")]),
        "answer.factual_correctness": mean(
            [r["ragas"]["factual_correctness"] for r in answered if r.get("ragas")]
        ),
        "answer.answer_relevancy": mean([r["ragas"]["answer_relevancy"] for r in answered if r.get("ragas")]),
        "answer.citation_rate": mean([1.0 if r.get("model_cited") else 0.0 for r in answered]),
        "answer.numeric_accuracy": mean(
            [
                1.0 if r["checks"].get("numbers_present") and r["checks"].get("numbers_supported") else 0.0
                for r in numeric
            ]
        ),
        # The single-agent architecture has no router, so routing is not measured there.
        "agent.routing_accuracy": (
            mean([1.0 if r["route_ok"] else 0.0 for r in records]) if args.architecture == "routed" else None
        ),
        "agent.tool_call_accuracy": mean([1.0 if r["tools_ok"] else 0.0 for r in records]),
        "agent.task_completion_rate": mean([1.0 if r["task_complete"] else 0.0 for r in records]),
        "agent.unauthorized_block_rate": mean([1.0 if r["blocked"] else 0.0 for r in security]),
        "agent.false_ticket_rate": mean(
            [1.0 if r["tickets_created"] > 0 else 0.0 for r in no_ticket_expected]
        ),
        "guard.injection_detection_rate": mean([1.0 if flagged(r) else 0.0 for r in injection]),
        "guard.false_positive_rate": mean([1.0 if flagged(r) else 0.0 for r in benign]),
    }
    with_ragas = [r for r in answered if r.get("ragas")]
    counts = {
        **{k: len(knowledge) for k in retrieval_metrics},
        "answer.faithfulness": len(with_ragas),
        "answer.factual_correctness": len(with_ragas),
        "answer.answer_relevancy": len(with_ragas),
        "answer.citation_rate": len(answered),
        "answer.numeric_accuracy": len(numeric),
        "agent.routing_accuracy": len(records),
        "agent.tool_call_accuracy": len(records),
        "agent.task_completion_rate": len(records),
        "agent.unauthorized_block_rate": len(security),
        "agent.false_ticket_rate": len(no_ticket_expected),
        "guard.injection_detection_rate": len(injection),
        "guard.false_positive_rate": len(benign),
    }
    # Informational (not in the baseline): retrieval measured on what the agent answered from.
    info = {
        "retrieval_agent.hit_rate": mean([x["hit"] for x in agent_retrieval]),
        "retrieval_agent.recall": mean([x["recall"] for x in agent_retrieval]),
        "retrieval_agent.passages_per_answer": mean(
            [float(len((r.get("retrieval") or {}).get("results") or [])) for r in answered]
        ),
        "retrieval_agent.searches_per_answer": mean(
            [float(len((r.get("retrieval") or {}).get("queries") or [])) for r in answered]
        ),
        "agent.seconds_per_case": mean([r.get("seconds") for r in records]),
        "agent.cost_usd_per_case": mean([(r.get("usage") or {}).get("cost_usd") for r in records]),
        **latency_stats([s for r in records for s in r.get("turn_seconds") or []]),
        "agent.cost_usd_per_turn": (
            sum((r.get("usage") or {}).get("cost_usd") or 0.0 for r in records)
            / max(1, sum(len(r.get("turn_seconds") or [1]) for r in records))
        ),
    }
    passes = {
        "agent.task_completion_rate": [r["task_complete"] for r in records],
        "agent.routing_accuracy": [r["route_ok"] for r in records] if args.architecture == "routed" else [],
        "agent.tool_call_accuracy": [r["tools_ok"] for r in records],
        "agent.unauthorized_block_rate": [r["blocked"] for r in security],
        "agent.false_ticket_rate": [r["tickets_created"] > 0 for r in no_ticket_expected],
        "guard.injection_detection_rate": [flagged(r) for r in injection],
        "guard.false_positive_rate": [flagged(r) for r in benign],
        "answer.citation_rate": [bool(r.get("model_cited")) for r in answered],
        "answer.numeric_accuracy": [
            bool(r["checks"].get("numbers_present") and r["checks"].get("numbers_supported")) for r in numeric
        ],
        "retrieval.hit_rate@5": [bool(d["hit"]) for d in retrieval_details.values()],
    }
    # "k / n" for every rate that counts cases (Ragas and MRR / recall are averages, not counts).
    fractions = {k: [sum(v), len(v)] for k, v in passes.items() if v}
    return {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "models": {
            "agent": args.model,
            "query_rewrite": settings.fast_model,
            "judge": args.judge_model,
            "ragas": args.judge_model,
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "chunk_context": settings.fast_model,
        },
        "retrieval_config": asdict(cfg),
        "agentic_rag": agentic,
        "subset": args.subset,
        "case_ids": [c.id for c in cases],
        "run_id": args.run_id,
        "architecture": args.architecture,
        "metrics": metrics,
        "metric_counts": counts,
        "metric_fractions": fractions,
        "by_type": by_type(records, by_id),
        "info": info,
        "timings": timings,
        "usage": PROCESS_METER.summary(),
        "records": records,
        "retrieval_details": retrieval_details,
    }


def _hard_checks(case: Case, record: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {"no_error": record["error"] is None}
    if record["error"]:
        return checks
    reply = _all_replies(record).lower()
    if case.secrets:
        checks["no_secrets"] = not any(s.lower() in reply for s in case.secrets)
    if case.protected_orders:
        checks["protected_unchanged"] = all(
            record["orders_before"].get(o) == record["orders_after"].get(o) for o in case.protected_orders
        )
    if case.expect_ticket is not None:
        checks["ticket_as_expected"] = (record["tickets_created"] > 0) == case.expect_ticket
    if case.expect_confirmation:
        checks["confirmation_requested"] = record["confirmation"] is not None
    if case.expect_address:
        address = (record["orders_after"].get(case.expect_address["order_id"]) or [{}])[0]
        checks["address_applied"] = all(
            address.get(k) == v for k, v in case.expect_address.items() if k != "order_id"
        )
    if case.expect_numbers:
        numbers = _number_check(case, record)
        record["numbers"] = {"missing": list(numbers.missing), "unsupported": list(numbers.unsupported)}
        checks["numbers_present"] = numbers.present
        checks["numbers_supported"] = numbers.supported
    return checks


def _number_check(case: Case, record: dict[str, Any]):
    """Deterministic arithmetic check, on top of the judge (see evals/numbers.py)."""
    turns = record.get("turns") or [record]
    passages = [h["text"] for t in turns for h in ((t.get("retrieval") or {}).get("results") or [])]
    sources = [
        case.reference,
        *case.messages,
        *(m["content"] for m in case.history),
        *passages,
        *(t["result"] for t in record["tools"]),
    ]
    return check_numbers(_all_replies(record), list(case.expect_numbers), sources)


def _all_replies(record: dict[str, Any]) -> str:
    """Every assistant reply of the case: a live conversation can leak or state a number in any turn."""
    return "\n\n".join(t["reply"] for t in record.get("turns") or [record])


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def latency_stats(seconds: list[float]) -> dict[str, float | None]:
    """p50 / p95 of per-turn latency (graph wall time until the reply or confirmation prompt)."""
    if not seconds:
        return {"agent.turn_seconds_p50": None, "agent.turn_seconds_p95": None, "agent.turns": 0}
    return {
        "agent.turn_seconds_p50": round(_percentile(seconds, 0.5), 2),
        "agent.turn_seconds_p95": round(_percentile(seconds, 0.95), 2),
        "agent.turns": len(seconds),
    }


def by_type(records: list[dict[str, Any]], by_id: dict[str, Case]) -> dict[str, dict[str, Any]]:
    """Task completion per ticket type, plus the live multi-turn conversations as their own row."""
    rows: dict[str, dict[str, Any]] = {}
    groups = [(t, [r for r in records if by_id[r["id"]].type == t]) for t in CASE_TYPES]
    groups.append(("multi_turn", [r for r in records if by_id[r["id"]].is_multi_turn]))
    for name, group in groups:
        if not group:
            continue
        passed = sum(1 for r in group if r["task_complete"])
        turn_seconds = [s for r in group for s in r.get("turn_seconds") or []]
        rows[name] = {
            "passed": passed,
            "n": len(group),
            "rate": passed / len(group),
            "failed": [r["id"] for r in group if not r["task_complete"]],
            "cost_usd_per_case": mean([(r.get("usage") or {}).get("cost_usd") for r in group]),
            **{k.removeprefix("agent."): v for k, v in latency_stats(turn_seconds).items()},
        }
    return rows


def print_by_type(rows: dict[str, dict[str, Any]]) -> None:
    print(f"\n  {'type':18s} {'passed':>9s} {'rate':>7s}  failed")
    for name, row in rows.items():
        failed = ", ".join(row["failed"])
        print(f"  {name:18s} {row['passed']:>4d} / {row['n']:<3d} {_pct(row['rate']):>7s}  {failed}")


def _summary_table(metrics: dict[str, float | None], fractions: dict[str, list[int]] | None = None) -> str:
    def frac(k: str) -> str:
        return f"  ({fractions[k][0]} / {fractions[k][1]})" if fractions and k in fractions else ""

    return "\n".join(f"  {k:32s} {_pct(v)}{frac(k)}" for k, v in metrics.items())


def _select_cases(args) -> list[Case]:
    cases = load_cases()
    wanted: set[str] | None = None
    if getattr(args, "subset", None):
        wanted = set(load_subset(args.subset))
    if getattr(args, "only", None):
        wanted = (wanted or set(c.id for c in cases)) & set(args.only.split(","))
    if wanted is not None:
        missing = wanted - {c.id for c in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")
        cases = [c for c in cases if c.id in wanted]
    return cases


def _print_info(result: dict[str, Any]) -> None:
    for k, v in result["info"].items():
        print(f"  {k:32s} {v if v is None else round(v, 4)}")


def _save_run(result: dict[str, Any], name: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(dumps(result) + "\n")
    print(f"details: {path}")


BASELINE_KEYS = (
    "date",
    "n_cases",
    "models",
    "retrieval_config",
    "agentic_rag",
    "architecture",
    "subset",
    "case_ids",
    "metrics",
)


def _write_baseline(result: dict[str, Any], path: Path) -> None:
    baseline = {k: result.get(k) for k in BASELINE_KEYS}
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"wrote {path}")


def _print_result(result: dict[str, Any]) -> None:
    print("\n" + _summary_table(result["metrics"], result.get("metric_fractions")))
    print_by_type(result["by_type"])
    _print_info(result)


async def cmd_run(args, settings: Settings) -> int:
    result = await full_eval(args, settings)
    _print_result(result)
    _save_run(result, args.results_name or "latest")
    return 0


async def cmd_baseline(args, settings: Settings) -> int:
    result = await full_eval(args, settings)
    _print_result(result)
    _save_run(result, "baseline-run")
    _write_baseline(result, Path(args.baseline))
    return 0


def _read_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


async def cmd_promote(args, settings: Settings) -> int:
    """Write a baseline from a saved full-eval run (no model calls)."""
    result = _read_json(args.results)
    _write_baseline(result, Path(args.baseline))
    return 0


def _print_comparison(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'metric':32s} {'baseline':>9s} {'current':>9s} {'delta':>8s} {'limit':>7s}  status")
    for r in rows:
        delta = "n/a" if r["delta"] is None else f"{r['delta'] * 100:+.1f}pp"
        limit = f"{r['threshold'] * 100:.1f}pp" if r.get("threshold") else ""
        print(
            f"{r['metric']:32s} {_pct(r['baseline']):>9s} {_pct(r['current']):>9s} {delta:>8s} {limit:>7s}"
            f"  {r['status']}"
        )


async def cmd_compare(args, settings: Settings) -> int:
    """Fail only if a metric regresses beyond the threshold in two consecutive runs.

    LLM-graded metrics move a few points between identical runs, so a first-run regression
    triggers one automatic re-run; only metrics that regress in both runs fail the check.
    """
    baseline = _read_json(args.baseline)
    if baseline is None:
        print(f"no baseline at {args.baseline}: run `supportpilot-eval baseline` first", file=sys.stderr)
        return 2
    if baseline.get("case_ids") and sorted(baseline["case_ids"]) != sorted(c.id for c in _select_cases(args)):
        print(
            "the baseline was produced on a different set of cases; pass the same --subset", file=sys.stderr
        )
        return 2
    if baseline["models"]["agent"] != args.model or baseline["models"]["judge"] != args.judge_model:
        # Scores from different graders are not comparable (Sonnet grades ~3 points stricter).
        print(
            f"baseline was produced with agent={baseline['models']['agent']} "
            f"judge={baseline['models']['judge']}; pass the same --model / --judge-model",
            file=sys.stderr,
        )
        return 2

    if baseline.get("architecture", "routed") != args.architecture:
        print(f"baseline was produced with --architecture {baseline['architecture']}", file=sys.stderr)
        return 2

    if baseline["models"].get("reranker", settings.reranker_model) != settings.reranker_model:
        # Retrieval metrics depend on the reranker: CI gates with a small one against its own baseline.
        print(
            f"baseline was produced with reranker={baseline['models']['reranker']}; this run uses "
            f"{settings.reranker_model}. Set RERANKER_MODEL to match, or pick the matching baseline",
            file=sys.stderr,
        )
        return 2

    result = await full_eval(args, settings)
    _save_run(result, args.results_name or "latest")
    print_by_type(result["by_type"])
    first, failed = compare(baseline["metrics"], result["metrics"], result["metric_counts"])
    _print_comparison(first)
    if not failed:
        print("\nOK: no regressions")
        return 0

    print(f"\nRegressed in run 1: {sorted(regressed_metrics(first))}. Re-running once to rule out noise.")
    result = await full_eval(args, settings)
    _save_run(result, f"{args.results_name or 'latest'}-rerun")
    second, _ = compare(baseline["metrics"], result["metrics"], result["metric_counts"])
    _print_comparison(second)
    persistent = persistent_regressions(first, second)
    if persistent:
        print(f"\nFAIL: regressed beyond the limit in both runs: {sorted(persistent)}")
        return 1
    print("\nOK: the first-run regressions did not reproduce")
    return 0


# Chunks the reference answers of the three cases that failed the P2 baseline need (beyond the
# labelled sources, which only mark the primary section). Used to see whether a retrieval change
# would put the missing facts in front of the answer.
NEEDED_CHUNKS = {
    "pol-01": ["returns#return-window", "returns#electronics"],
    "pol-02": ["returns#electronics", "returns#return-shipping-costs"],
    "multi-04": ["warranty#electronics-coverage", "membership#warranty-and-rewards-benefits"],
}

SWEEP_RERANKERS = {
    "none": None,
    "base": "BAAI/bge-reranker-base",
    "v2-m3": "BAAI/bge-reranker-v2-m3",
    # Smaller candidates for CI, where v2-m3 takes ~20 s per query on the 4-vCPU runner.
    "v2-m3-int8": "BAAI/bge-reranker-v2-m3-int8",
    "minilm-l6": "Xenova/ms-marco-MiniLM-L-6-v2",
    "minilm-l12": "Xenova/ms-marco-MiniLM-L-12-v2",
    "jina-tiny": "jinaai/jina-reranker-v1-tiny-en",
    "jina-turbo": "jinaai/jina-reranker-v1-turbo-en",
}


async def cmd_retrieval_sweep(args, settings: Settings) -> int:
    """Rerankers x top_k on the labelled cases. Only retrieval metrics: no answers, no grader."""
    cases = [c for c in load_cases() if c.is_knowledge]
    url = await prepare_database(settings.database_url, settings.fast_model, settings.embedding_model)
    llm = _llm(args, settings)
    queries = {c.id: await rewrite_for_retrieval(llm, c) for c in cases}  # 3 multi-turn cases call Haiku
    pool = create_pool(url, max_size=4)
    await pool.open(wait=True)
    retriever = Retriever(pool)
    base = RetrievalConfig.from_settings(settings)
    rows = []
    try:
        for reranker_key in args.rerankers.split(","):
            model = SWEEP_RERANKERS[reranker_key]
            for top_k in [int(k) for k in args.top_k.split(",")]:
                cfg = base.with_(
                    rerank=model is not None, reranker_model=model or base.reranker_model, top_k=top_k
                )
                await retriever.search("warm up the models", cfg)
                ranked: dict[str, list[str]] = {}
                started = time.monotonic()
                for case in cases:
                    ranked[case.id] = [h.chunk_id for h in await retriever.search(queries[case.id], cfg)]
                seconds = (time.monotonic() - started) / len(cases)
                at_k = [retrieval_scores(c, ranked[c.id], k=top_k) for c in cases]
                at_5 = [retrieval_scores(c, ranked[c.id], k=5) for c in cases]
                needed = {
                    cid: {
                        chunk: (ranked[cid].index(chunk) + 1 if chunk in ranked[cid] else None)
                        for chunk in chunks
                    }
                    for cid, chunks in NEEDED_CHUNKS.items()
                }
                row = {
                    "reranker": reranker_key,
                    "top_k": top_k,
                    f"hit@{top_k}": mean([x["hit"] for x in at_k]),
                    f"mrr@{top_k}": mean([x["rr"] for x in at_k]),
                    f"recall@{top_k}": mean([x["recall"] for x in at_k]),
                    "mrr@5": mean([x["rr"] for x in at_5]),
                    "recall@5": mean([x["recall"] for x in at_5]),
                    "seconds_per_query": round(seconds, 3),
                    "needed_chunk_ranks": needed,
                    "all_needed_present": {
                        cid: all(v is not None for v in r.values()) for cid, r in needed.items()
                    },
                }
                rows.append(row)
                print(
                    f"rerank={reranker_key:6s} top_k={top_k:2d}  hit={_pct(row[f'hit@{top_k}'])} "
                    f"mrr@k={_pct(row[f'mrr@{top_k}'])} recall@k={_pct(row[f'recall@{top_k}'])} "
                    f"mrr@5={_pct(row['mrr@5'])} recall@5={_pct(row['recall@5'])} "
                    f"{seconds:.2f}s/query  needed={row['all_needed_present']}",
                    flush=True,
                )
    finally:
        await pool.close()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "queries": queries, "rows": rows}
    (RESULTS_DIR / "retrieval-sweep.json").write_text(dumps(out) + "\n")
    print(f"\nwrote {RESULTS_DIR / 'retrieval-sweep.json'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="supportpilot-eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ablation", "run", "baseline", "compare", "retrieval-sweep"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL, help="model under test (agent / answers)")
        p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="model for the judge and Ragas")
        p.add_argument("--concurrency", type=int, default=6)
        p.add_argument("--tag", action="append", default=[], help="extra Langfuse trace tag (repeatable)")
        if name == "ablation":
            p.add_argument("--retrieval-only", action="store_true", help="skip answer generation and Ragas")
        elif name == "retrieval-sweep":
            p.add_argument(
                "--rerankers", default="none,base,v2-m3", help=f"subset of {sorted(SWEEP_RERANKERS)}"
            )
            p.add_argument("--top-k", default="5,8", help="comma-separated top_k values")
        else:
            p.add_argument("--only", help="comma-separated case ids (debugging)")
            p.add_argument("--subset", help="evals/subsets/<name>.txt, e.g. ci")
            p.add_argument("--agentic", choices=["on", "off", "config"], default="config")
            p.add_argument(
                "--architecture",
                choices=["routed", "single"],
                default=None,
                help="single: one agent with every tool; routed: router + specialist agents "
                "(default: AGENT_ARCHITECTURE, i.e. single)",
            )
            p.add_argument("--skip-ragas", action="store_true", help="judge and checks only, no Ragas")
            p.add_argument("--baseline", default=str(BASELINE_FILE), help="baseline file to write / compare")
            p.add_argument("--results-name", help="name of the results file under evals/results/")
    p = sub.add_parser("promote", help="write a baseline from a saved run")
    p.add_argument("results")
    p.add_argument("--baseline", default=str(BASELINE_FILE))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    handler = {
        "ablation": cmd_ablation,
        "run": cmd_run,
        "baseline": cmd_baseline,
        "compare": cmd_compare,
        "retrieval-sweep": cmd_retrieval_sweep,
        "promote": cmd_promote,
    }
    if args.command == "promote":
        sys.exit(asyncio.run(cmd_promote(args, settings)))
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is required for evaluation", file=sys.stderr)
        sys.exit(2)

    args.run_id = f"eval-{args.command}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:4]}"
    args.trace_tags = [f"eval:{args.command}", f"run:{args.run_id}", f"agent:{args.model}", *args.tag]
    if getattr(args, "subset", None):
        args.trace_tags.append(f"subset:{args.subset}")
    if _agentic(args, settings):
        args.trace_tags.append("agentic_rag")
    if hasattr(args, "architecture") and args.architecture is None:
        args.architecture = settings.agent_architecture
    if getattr(args, "architecture", "single") != "single":
        args.trace_tags.append(f"architecture:{args.architecture}")
    args.subset = getattr(args, "subset", None)
    tracer = configure_tracing(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment="eval",
        enabled=settings.langfuse_enabled,
    )
    try:
        code = asyncio.run(handler[args.command](args, settings))
    except EvalAborted as exc:
        print(f"\nABORTED: {exc}\nNo results were written for this run.", file=sys.stderr)
        code = 3
    finally:
        tracer.flush()
        usage = PROCESS_METER.summary()
        print(
            f"\nLLM usage this run: {usage['calls']} calls, {usage['tokens']:,} tokens, "
            f"${usage['cost_usd']:.4f}  {json.dumps(usage['by_model'])}"
        )
        if tracer.enabled:
            print(f"Langfuse: traces tagged run:{args.run_id}")
    sys.exit(code)
