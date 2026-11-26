"""Offline checks of the evaluation metrics and the baseline comparison used by CI."""

from supportpilot.evals.cases import Case, load_cases
from supportpilot.evals.metrics import compare, retrieval_scores, tools_ok
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
