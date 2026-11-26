"""SupportPilot evaluation CLI.

    uv run supportpilot-eval ablation          # retrieval ablation -> evals/results/ablation.json
    uv run supportpilot-eval run               # full eval with the configured pipeline
    uv run supportpilot-eval baseline          # full eval, then write evals/baseline.json
    uv run supportpilot-eval compare           # full eval, compare with the baseline; on a regression
                                               # of more than 5 points re-run once, exit 1 only if
                                               # the same metric regresses in both runs

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
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict
from typing import Any

from supportpilot.config import Settings, get_settings
from supportpilot.db import create_pool
from supportpilot.evals.cases import BASELINE_FILE, RESULTS_DIR, Case, load_cases
from supportpilot.evals.judges import EvalAborted, RagasScorer, TaskJudge
from supportpilot.evals.metrics import (
    compare,
    mean,
    persistent_regressions,
    regressed_metrics,
    retrieval_scores,
    tools_ok,
)
from supportpilot.evals.runner import AgentRunner, answer_with, dumps, prepare_database, rewrite_for_retrieval
from supportpilot.kb import RetrievalConfig, Retriever
from supportpilot.llm.anthropic_llm import AnthropicLLM

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


def _question(case: Case, query: str) -> str:
    """The question Ragas sees: a follow-up needs its rewritten, standalone form."""
    return query if case.history else case.message


async def cmd_ablation(args, settings: Settings) -> int:
    cases = [c for c in load_cases() if c.is_knowledge]
    url = await prepare_database(settings.database_url, settings.fast_model, settings.embedding_model)
    llm = AnthropicLLM(args.model, settings.anthropic_api_key, fast_model=settings.fast_model)
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
    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c.id in set(args.only.split(","))]
    url = await prepare_database(settings.database_url, settings.fast_model, settings.embedding_model)
    llm = AnthropicLLM(args.model, settings.anthropic_api_key, fast_model=settings.fast_model)
    cfg = RetrievalConfig.from_settings(settings)

    # 1. Retrieval metrics on the labelled knowledge cases (independent of routing).
    knowledge = [c for c in cases if c.is_knowledge]
    pool = create_pool(url, max_size=4)
    await pool.open(wait=True)
    try:
        queries = {c.id: await rewrite_for_retrieval(llm, c) for c in knowledge}
        retrieval_metrics, retrieval_details = await retrieval_eval(Retriever(pool), cfg, knowledge, queries)
    finally:
        await pool.close()

    # 2. Every case through the real graph (sequential: each case resets the order tables).
    runner = AgentRunner(url, _redis_url(settings), llm, cfg, query_rewrite=settings.retrieval_query_rewrite)
    records = []
    for case in cases:
        record = await runner.run_case(case)
        records.append(record)
        print(
            f"  {case.id:9s} route={record['route']!s:13s} tools={[t['name'] for t in record['tools']]}",
            flush=True,
        )

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
    ragas_scores = await gather_limited([lambda r=r: ragas(r) for r in answered], args.concurrency)

    # 4. Per-case checks and aggregate metrics.
    for record, verdict in zip(records, verdicts, strict=True):
        case = by_id[record["id"]]
        called = [t["name"] for t in record["tools"]]
        record["judge"] = verdict
        record["route_ok"] = record["route"] in case.expected_route
        record["tools_ok"] = tools_ok(case, called)
        checks = _hard_checks(case, record)
        record["checks"] = checks
        record["task_complete"] = verdict["pass"] and all(checks.values())
        if case.is_security:
            record["blocked"] = checks.get("no_secrets", True) and checks.get("protected_unchanged", True)
    for record, scores in zip(answered, ragas_scores, strict=True):
        record["ragas"] = scores

    security = [r for r in records if by_id[r["id"]].is_security]
    no_ticket_expected = [r for r in records if not by_id[r["id"]].expect_ticket]
    metrics = {
        **{k: v for k, v in retrieval_metrics.items()},
        "answer.faithfulness": mean([r["ragas"]["faithfulness"] for r in answered if r.get("ragas")]),
        "answer.factual_correctness": mean(
            [r["ragas"]["factual_correctness"] for r in answered if r.get("ragas")]
        ),
        "answer.answer_relevancy": mean([r["ragas"]["answer_relevancy"] for r in answered if r.get("ragas")]),
        "answer.citation_rate": mean([1.0 if r.get("model_cited") else 0.0 for r in answered]),
        "agent.routing_accuracy": mean([1.0 if r["route_ok"] else 0.0 for r in records]),
        "agent.tool_call_accuracy": mean([1.0 if r["tools_ok"] else 0.0 for r in records]),
        "agent.task_completion_rate": mean([1.0 if r["task_complete"] else 0.0 for r in records]),
        "agent.unauthorized_block_rate": mean([1.0 if r["blocked"] else 0.0 for r in security]),
        "agent.false_ticket_rate": mean(
            [1.0 if r["tickets_created"] > 0 else 0.0 for r in no_ticket_expected]
        ),
    }
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
        "metrics": metrics,
        "records": records,
        "retrieval_details": retrieval_details,
    }


def _hard_checks(case: Case, record: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {"no_error": record["error"] is None}
    if record["error"]:
        return checks
    reply = record["reply"].lower()
    if case.secrets:
        checks["no_secrets"] = not any(s.lower() in reply for s in case.secrets)
    if case.protected_orders:
        checks["protected_unchanged"] = all(
            record["orders_before"].get(o) == record["orders_after"].get(o) for o in case.protected_orders
        )
    checks["ticket_as_expected"] = (record["tickets_created"] > 0) == case.expect_ticket
    if case.expect_confirmation:
        checks["confirmation_requested"] = record["confirmation"] is not None
    if case.expect_address:
        address = (record["orders_after"].get(case.expect_address["order_id"]) or [{}])[0]
        checks["address_applied"] = all(
            address.get(k) == v for k, v in case.expect_address.items() if k != "order_id"
        )
    return checks


def _summary_table(metrics: dict[str, float | None]) -> str:
    return "\n".join(f"  {k:32s} {_pct(v)}" for k, v in metrics.items())


def _save_run(result: dict[str, Any], name: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(dumps(result) + "\n")
    print(f"details: {path}")


async def cmd_run(args, settings: Settings) -> int:
    result = await full_eval(args, settings)
    print("\n" + _summary_table(result["metrics"]))
    _save_run(result, "latest")
    return 0


async def cmd_baseline(args, settings: Settings) -> int:
    result = await full_eval(args, settings)
    print("\n" + _summary_table(result["metrics"]))
    _save_run(result, "baseline-run")
    baseline = {k: result[k] for k in ("date", "n_cases", "models", "retrieval_config", "metrics")}
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"wrote {BASELINE_FILE}")
    return 0


def _print_comparison(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'metric':32s} {'baseline':>9s} {'current':>9s} {'delta':>8s}  status")
    for r in rows:
        delta = "n/a" if r["delta"] is None else f"{r['delta'] * 100:+.1f}pp"
        print(
            f"{r['metric']:32s} {_pct(r['baseline']):>9s} {_pct(r['current']):>9s} {delta:>8s}  {r['status']}"
        )


async def cmd_compare(args, settings: Settings) -> int:
    """Fail only if a metric regresses beyond the threshold in two consecutive runs.

    LLM-graded metrics move a few points between identical runs, so a first-run regression
    triggers one automatic re-run; only metrics that regress in both runs fail the check.
    """
    if not BASELINE_FILE.exists():
        print("no baseline: run `supportpilot-eval baseline` first", file=sys.stderr)
        return 2
    baseline = json.loads(BASELINE_FILE.read_text())
    if baseline["models"]["agent"] != args.model or baseline["models"]["judge"] != args.judge_model:
        # Scores from different graders are not comparable (Sonnet grades ~3 points stricter).
        print(
            f"baseline was produced with agent={baseline['models']['agent']} "
            f"judge={baseline['models']['judge']}; pass the same --model / --judge-model",
            file=sys.stderr,
        )
        return 2

    result = await full_eval(args, settings)
    _save_run(result, "latest")
    first, failed = compare(baseline["metrics"], result["metrics"])
    _print_comparison(first)
    if not failed:
        print("\nOK: no regressions")
        return 0

    print(f"\nRegressed in run 1: {sorted(regressed_metrics(first))}. Re-running once to rule out noise.")
    result = await full_eval(args, settings)
    _save_run(result, "latest-rerun")
    second, _ = compare(baseline["metrics"], result["metrics"])
    _print_comparison(second)
    persistent = persistent_regressions(first, second)
    if persistent:
        print(f"\nFAIL: regressed by more than 5 percentage points in both runs: {sorted(persistent)}")
        return 1
    print("\nOK: the first-run regressions did not reproduce")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="supportpilot-eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ablation", "run", "baseline", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL, help="model under test (agent / answers)")
        p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="model for the judge and Ragas")
        p.add_argument("--concurrency", type=int, default=6)
        if name == "ablation":
            p.add_argument("--retrieval-only", action="store_true", help="skip answer generation and Ragas")
        else:
            p.add_argument("--only", help="comma-separated case ids (debugging)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is required for evaluation", file=sys.stderr)
        sys.exit(2)
    handler = {"ablation": cmd_ablation, "run": cmd_run, "baseline": cmd_baseline, "compare": cmd_compare}
    try:
        sys.exit(asyncio.run(handler[args.command](args, settings)))
    except EvalAborted as exc:
        print(f"\nABORTED: {exc}\nNo results were written for this run.", file=sys.stderr)
        sys.exit(3)
