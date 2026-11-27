"""The single-agent architecture (AGENT_ARCHITECTURE=single), the baseline the router is compared with.

One agent holds every tool; the confirmation gate, identity injection and memory work as in the
routed graph.
"""

from supportpilot.llm.rule_based import RuleBasedLLM
from tests.conftest import ALICE, BOB, chat, confirm, fetch_order
from tests.test_flows import CHANGE_1001, NEW_1001


class ToolSpyLLM(RuleBasedLLM):
    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    async def classify_intent(self, system, messages):
        raise AssertionError("the single agent has no router")

    async def respond(self, system, messages, tools, on_text, *, system_suffix=None, name="respond"):
        self.calls.append((name, [t["name"] for t in tools]))
        return await super().respond(system, messages, tools, on_text, system_suffix=system_suffix, name=name)


async def test_one_agent_holds_every_tool_and_no_router_runs(make_client):
    llm = ToolSpyLLM()
    client, _ = await make_client(llm=llm, agent_architecture="single")
    turn = await chat(client, ALICE, "What is the status of ORD-1002?")
    assert turn["route"] == "agent"
    assert "shipped" in turn["reply"]
    name, tools = llm.calls[0]
    assert name == "single_agent"
    assert set(tools) == {"get_order", "list_my_orders", "update_address", "create_ticket", "search_policy"}


async def test_address_change_still_waits_for_confirmation(make_client):
    client, _ = await make_client(agent_architecture="single")
    turn = await chat(client, ALICE, CHANGE_1001)
    assert turn["confirmation"] is not None
    assert (await fetch_order("ORD-1001"))["address_events"] == 0

    resp = await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")
    assert resp.json()["status"] == "completed"
    assert (await fetch_order("ORD-1001"))["address"] == NEW_1001


async def test_other_customers_orders_stay_invisible(make_client):
    client, _ = await make_client(agent_architecture="single")
    turn = await chat(client, BOB, "What is the status of ORD-1002?")
    assert "not found" in turn["reply"].lower() or "couldn't" in turn["reply"].lower()
    assert "1Z999AA10123456784" not in turn["reply"]


async def test_policy_answer_comes_from_search_policy_with_a_citation(make_client):
    client, _ = await make_client(agent_architecture="single")
    turn = await chat(client, ALICE, "What is your return window policy?")
    assert "[Returns Policy §" in turn["reply"]
    assert "sources" in turn["names"]


async def test_request_for_a_human_opens_a_ticket(make_client):
    client, _ = await make_client(agent_architecture="single")
    turn = await chat(client, BOB, "I want to talk to a real person")
    assert "TCK-" in turn["reply"]


def test_single_agent_is_the_default():
    from supportpilot.config import Settings

    assert Settings(_env_file=None).agent_architecture == "single"
