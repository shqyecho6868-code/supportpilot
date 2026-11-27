"""Offline checks of the evaluation metrics and the baseline comparison used by CI."""

import anthropic
import httpx
import pytest

from supportpilot.evals.cases import Case, load_cases
from supportpilot.evals.judges import EvalAborted, raise_if_fatal
from supportpilot.evals.metrics import compare, persistent_regressions, retrieval_scores, tools_ok
from supportpilot.kb.documents import chunk_document, load_documents


def _case(**kw) -> Case:
    base = {
        "id": "x",
        "category": "policy",
        "user_id": "u",
        "message": "m",
        "reference": "r",
        "expected_route": "knowledge",
    }
    return Case.from_dict({**base, **kw})


def test_golden_set_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 40
    chunk_ids = {c.id for d in load_documents() for c in chunk_document(d)}
    for case in cases:
        assert set(case.expected_sources) <= chunk_ids, case.id
        assert all(
            r in {"order", "knowledge", "handoff", "smalltalk", "out_of_scope"} for r in case.expected_route
        )
    categories = {c.category for c in cases}
    assert {"policy", "policy_multi_doc", "order", "address_change", "cross_user", "prompt_injection",
            "smalltalk", "out_of_scope", "handoff"} <= categories  # fmt: skip


def test_retrieval_scores():
    case = _case(expected_sources=["a", "b"])
    assert retrieval_scores(case, ["x", "b", "a"]) == {"hit": 1.0, "rr": 0.5, "recall": 1.0}
    assert retrieval_scores(case, ["x", "y", "z", "w", "v", "a"]) == {"hit": 0.0, "rr": 0.0, "recall": 0.0}


def test_tool_expectations_support_alternatives_and_forbidden_tools():
    case = _case(expected_tools=["get_order|list_my_orders"], forbidden_tools=["create_ticket"])
    assert tools_ok(case, ["list_my_orders"])
    assert not tools_ok(case, [])
    assert not tools_ok(case, ["get_order", "create_ticket"])


def test_compare_flags_drops_over_five_points_in_either_direction_of_goodness():
    baseline = {"agent.routing_accuracy": 0.90, "agent.false_ticket_rate": 0.02, "retrieval.mrr": 0.80}
    ok, failed = compare(
        baseline, {"agent.routing_accuracy": 0.86, "agent.false_ticket_rate": 0.06, "retrieval.mrr": 0.95}
    )
    assert not failed
    rows, failed = compare(
        baseline, {"agent.routing_accuracy": 0.84, "agent.false_ticket_rate": 0.02, "retrieval.mrr": 0.8}
    )
    assert failed and {r["metric"] for r in rows if r["status"] == "REGRESSED"} == {"agent.routing_accuracy"}
    rows, failed = compare(
        baseline, {"agent.routing_accuracy": 0.9, "agent.false_ticket_rate": 0.10, "retrieval.mrr": 0.8}
    )
    assert failed and {r["metric"] for r in rows if r["status"] == "REGRESSED"} == {"agent.false_ticket_rate"}


def test_only_regressions_seen_in_both_runs_fail():
    baseline = {"agent.routing_accuracy": 0.90, "answer.faithfulness": 0.80}
    first, _ = compare(baseline, {"agent.routing_accuracy": 0.80, "answer.faithfulness": 0.70})
    second, _ = compare(baseline, {"agent.routing_accuracy": 0.82, "answer.faithfulness": 0.79})
    assert persistent_regressions(first, second) == {"agent.routing_accuracy"}


def test_account_errors_abort_the_run_even_when_wrapped():
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    credit = anthropic.BadRequestError("Your credit balance is too low", response=response, body=None)
    try:
        raise RuntimeError("instructor retry failed") from credit
    except RuntimeError as wrapped:
        with pytest.raises(EvalAborted):
            raise_if_fatal(wrapped)
    raise_if_fatal(ValueError("a parsing problem"))  # ordinary failures do not abort


def test_small_samples_need_more_than_one_case_to_regress():
    from supportpilot.evals.metrics import threshold_for

    baseline = {"answer.factual_correctness": 0.80, "agent.task_completion_rate": 1.0}
    current = {"answer.factual_correctness": 0.66, "agent.task_completion_rate": 0.70}
    rows, failed = compare(
        baseline, current, {"answer.factual_correctness": 7, "agent.task_completion_rate": 17}
    )
    status = {r["metric"]: r["status"] for r in rows}
    # One answer out of 7 is 14 points: noise. Five tasks out of 17 failing is a regression.
    assert status == {"answer.factual_correctness": "ok", "agent.task_completion_rate": "REGRESSED"}
    assert failed
    assert threshold_for(26) == 1.5 / 26 and threshold_for(100) == 0.05 and threshold_for(None) == 0.05


def test_expected_numbers_come_from_the_reference():
    """Every `expect_numbers` quantity is stated in the case's own reference answer."""
    from supportpilot.evals.numbers import quantities

    labelled = [c for c in load_cases() if c.expect_numbers]
    assert len(labelled) >= 20
    for case in labelled:
        reference = quantities(case.reference)
        for item in case.expect_numbers:
            assert quantities(item) and quantities(item) <= reference, (case.id, item)


def test_number_check_fails_the_task_even_when_the_judge_passes():
    from supportpilot.evals.cli import _hard_checks

    case = _case(
        reference="Refunds post 5 to 10 business days after they are issued.",
        expect_numbers=["5 to 10 business days"],
    )
    record = {
        "error": None,
        "reply": "Expect the money in about 7 to 13 business days.",
        "tools": [],
        "retrieval": {"results": [{"text": "Card refunds take 5 to 10 business days to post."}]},
        "tickets_created": 0,
        "confirmation": None,
        "orders_before": {},
        "orders_after": {},
    }
    checks = _hard_checks(case, record)
    assert checks["numbers_present"] is False and checks["numbers_supported"] is False
    assert record["numbers"] == {"missing": ["5 day", "10 day"], "unsupported": ["7 day", "13 day"]}
