"""Metric definitions and the baseline comparison.

Every metric is a rate in [0, 1]. ``LOWER_IS_BETTER`` lists the ones where an
increase is a regression.
"""

import math
from typing import Any

from supportpilot.evals.cases import Case

LOWER_IS_BETTER = frozenset({"agent.false_ticket_rate"})

# Regressions larger than this (absolute, 0.05 = 5 percentage points) fail `compare`.
REGRESSION_THRESHOLD = 0.05


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else None


def retrieval_scores(case: Case, ranked_chunk_ids: list[str], k: int = 5) -> dict[str, float]:
    """hit@k, reciprocal rank (within the top k) and recall@k for one query."""
    relevant = set(case.expected_sources)
    top = ranked_chunk_ids[:k]
    rr = next((1.0 / rank for rank, cid in enumerate(top, start=1) if cid in relevant), 0.0)
    return {
        "hit": 1.0 if rr > 0 else 0.0,
        "rr": rr,
        "recall": len(relevant & set(top)) / len(relevant),
    }


def tools_ok(case: Case, called: list[str]) -> bool:
    called_set = set(called)
    for entry in case.expected_tools:
        if not called_set & set(entry.split("|")):
            return False
    return not called_set & set(case.forbidden_tools)


def compare(baseline: dict[str, float], current: dict[str, float]) -> tuple[list[dict[str, Any]], bool]:
    """Rows for a comparison table, and whether any metric regressed beyond the threshold."""
    rows = []
    failed = False
    for name in sorted(baseline):
        base, cur = baseline[name], current.get(name)
        if base is None:
            continue
        if cur is None:
            rows.append(
                {"metric": name, "baseline": base, "current": None, "delta": None, "status": "MISSING"}
            )
            failed = True
            continue
        delta = cur - base
        worse = -delta if name not in LOWER_IS_BETTER else delta
        regressed = worse > REGRESSION_THRESHOLD + 1e-9
        failed |= regressed
        rows.append(
            {
                "metric": name,
                "baseline": base,
                "current": cur,
                "delta": delta,
                "status": "REGRESSED" if regressed else "ok",
            }
        )
    return rows, failed


def regressed_metrics(rows: list[dict[str, Any]]) -> set[str]:
    return {r["metric"] for r in rows if r["status"] in {"REGRESSED", "MISSING"}}


def persistent_regressions(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> set[str]:
    """Metrics that regressed in both runs. A regression seen only once is treated as run-to-run noise."""
    return regressed_metrics(first) & regressed_metrics(second)
