"""Agentic RAG: decide whether to search, grade the passages, rewrite and search again."""

from typing import Any

from supportpilot.llm.rule_based import RuleBasedLLM
from tests.conftest import ALICE, chat


class ScriptedLLM(RuleBasedLLM):
    """Rule-based answers, with scripted retrieval decisions and grades."""

    def __init__(self, decision: dict[str, Any], grades: list[dict[str, Any]]):
        self.decision = decision
        self.grades = list(grades)
        self.calls: list[str] = []

    async def structured(self, name, system, messages, schema, *, max_tokens=1024):
        self.calls.append(name)
        if name == "retrieval_decision":
            return self.decision
        return self.grades.pop(0)


async def _state(app, turn) -> dict[str, Any]:
    thread_id = next(d for n, d in turn["events"] if n == "meta")["thread_id"]
    state = await app.state.ctx.graph.aget_state({"configurable": {"thread_id": thread_id}})
    return state.values


async def test_sufficient_first_search_answers_with_citations(make_client):
    llm = ScriptedLLM(
        {"reason": "policy", "retrieve": True, "query": "return window"},
        [{"missing": "", "sufficient": True, "next_query": ""}],
    )
    client, app = await make_client(llm=llm, agentic_rag=True)
    turn = await chat(client, ALICE, "What is your return window policy?")
    values = await _state(app, turn)
    assert values["retrieval"]["queries"] == ["return window"]
    assert llm.calls == ["retrieval_decision", "retrieval_grade"]
    assert "[Returns Policy §" in turn["reply"]


async def test_insufficient_passages_trigger_at_most_two_rewrites(make_client):
    llm = ScriptedLLM(
        {"reason": "policy", "retrieve": True, "query": "return window"},
        [
            {"missing": "electronics", "sufficient": False, "next_query": "electronics return window"},
            {"missing": "fees", "sufficient": False, "next_query": "return label fee"},
            {"missing": "never asked", "sufficient": False, "next_query": "unused"},
        ],
    )
    client, app = await make_client(llm=llm, agentic_rag=True)
    turn = await chat(client, ALICE, "What is your return window policy?")
    retrieval = (await _state(app, turn))["retrieval"]
    assert retrieval["queries"] == ["return window", "electronics return window", "return label fee"]
    # Two grades: the third search is the last allowed one, so it is not graded.
    assert llm.calls == ["retrieval_decision", "retrieval_grade", "retrieval_grade"]
    chunk_ids = [h["chunk_id"] for h in retrieval["results"]]
    assert len(chunk_ids) == len(set(chunk_ids)) <= 10
    assert "returns#return-shipping-costs" in chunk_ids  # found only by the rewritten query
    assert turn["reply"]


async def test_repeated_query_stops_the_loop(make_client):
    llm = ScriptedLLM(
        {"reason": "policy", "retrieve": True, "query": "return window"},
        [{"missing": "x", "sufficient": False, "next_query": "return window"}],
    )
    client, app = await make_client(llm=llm, agentic_rag=True)
    turn = await chat(client, ALICE, "What is your return window policy?")
    assert (await _state(app, turn))["retrieval"]["queries"] == ["return window"]


async def test_follow_up_that_needs_no_policy_skips_retrieval(make_client):
    llm = ScriptedLLM(
        {"reason": "policy", "retrieve": True, "query": "return window"},
        [{"missing": "", "sufficient": True, "next_query": ""}],
    )
    client, app = await make_client(llm=llm, agentic_rag=True)
    await chat(client, ALICE, "What is your return window policy?", session_id="s-skip-rag")
    llm.calls.clear()
    llm.decision = {"reason": "thanks", "retrieve": False, "query": ""}
    turn = await chat(client, ALICE, "Thanks, that's what the policy says then", session_id="s-skip-rag")
    values = await _state(app, turn)
    assert values["retrieval"]["skipped"] is True
    assert llm.calls == ["retrieval_decision"]
    assert not any(n == "sources" and d["sources"] for n, d in turn["events"])


async def test_first_message_always_searches(make_client):
    llm = ScriptedLLM(
        {"reason": "model said no", "retrieve": False, "query": ""},
        [{"missing": "", "sufficient": True, "next_query": ""}],
    )
    client, app = await make_client(llm=llm, agentic_rag=True)
    turn = await chat(client, ALICE, "What is your return window policy?")
    assert (await _state(app, turn))["retrieval"]["results"]


async def test_agentic_rag_off_is_single_shot(client):
    turn = await chat(client, ALICE, "What is your return window policy?")
    assert any(n == "sources" for n in turn["names"])
    assert "[Returns Policy §" in turn["reply"]
