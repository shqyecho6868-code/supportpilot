"""End-to-end flows through FastAPI -> LangGraph -> MCP -> Postgres/Redis.

Uses the deterministic RuleBasedLLM so the flows are reproducible offline.
The same scenarios against real Claude live in tests/test_live_claude.py.
"""

import psycopg

from tests.conftest import ALICE, BOB, TEST_DATABASE_URL, chat, confirm, fetch_order

CHANGE_1001 = "Please change the shipping address of ORD-1001 to 500 Market St, San Francisco, CA 94105, US"
NEW_1001 = {
    "street": "500 Market St",
    "city": "San Francisco",
    "state": "CA",
    "postal_code": "94105",
    "country": "US",
}


# --- Verification 1: confirmation gate + idempotency --------------------------------


async def test_address_change_pauses_for_confirmation_before_writing(client):
    before = await fetch_order("ORD-1001")
    turn = await chat(client, ALICE, CHANGE_1001)

    assert turn["route"] == "order"
    assert turn["confirmation"] is not None
    assert turn["confirmation"]["order_id"] == "ORD-1001"
    assert turn["confirmation"]["new_address"] == NEW_1001
    assert turn["confirmation"]["current_address"] == before["address"]
    assert "final" not in turn["names"]
    assert await fetch_order("ORD-1001") == before  # nothing written while paused


async def test_reject_leaves_address_unchanged(client):
    before = await fetch_order("ORD-1001")
    turn = await chat(client, ALICE, CHANGE_1001)

    resp = await confirm(client, ALICE, turn["confirmation"]["thread_id"], "reject")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert "declined" in resp.json()["reply"]
    assert await fetch_order("ORD-1001") == before


async def test_never_confirming_leaves_address_unchanged(client):
    before = await fetch_order("ORD-1001")
    await chat(client, ALICE, CHANGE_1001)
    # The customer walks away and starts talking about something else.
    await chat(client, ALICE, "What is the status of ORD-1002?")
    assert await fetch_order("ORD-1001") == before


async def test_approve_applies_change(client):
    turn = await chat(client, ALICE, CHANGE_1001)
    resp = await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed" and "San Francisco" in body["reply"]
    row = await fetch_order("ORD-1001")
    assert row["address"] == NEW_1001
    assert row["address_events"] == 1


async def test_approving_twice_changes_address_once(client):
    turn = await chat(client, ALICE, CHANGE_1001)
    thread_id = turn["confirmation"]["thread_id"]

    first = await confirm(client, ALICE, thread_id, "approve")
    second = await confirm(client, ALICE, thread_id, "approve")

    assert first.status_code == 200
    assert second.status_code == 409
    row = await fetch_order("ORD-1001")
    assert row["address"] == NEW_1001
    assert row["address_events"] == 1


async def test_reject_then_approve_does_not_apply(client):
    before = await fetch_order("ORD-1001")
    turn = await chat(client, ALICE, CHANGE_1001)
    thread_id = turn["confirmation"]["thread_id"]
    assert (await confirm(client, ALICE, thread_id, "reject")).status_code == 200
    assert (await confirm(client, ALICE, thread_id, "approve")).status_code == 409
    assert await fetch_order("ORD-1001") == before


async def test_non_pending_order_is_refused_without_asking_for_confirmation(client):
    before = await fetch_order("ORD-1002")
    turn = await chat(client, ALICE, CHANGE_1001.replace("ORD-1001", "ORD-1002"))
    assert turn["confirmation"] is None
    assert "shipped" in turn["reply"]
    assert await fetch_order("ORD-1002") == before


async def test_other_user_cannot_confirm(client):
    before = await fetch_order("ORD-1001")
    turn = await chat(client, ALICE, CHANGE_1001)
    resp = await confirm(client, BOB, turn["confirmation"]["thread_id"], "approve")
    assert resp.status_code == 404
    assert await fetch_order("ORD-1001") == before


async def test_confirm_unknown_thread(client):
    resp = await confirm(client, ALICE, "does-not-exist", "approve")
    assert resp.status_code == 404


# --- Verification 2: no cross-user access -------------------------------------------


async def test_user_a_cannot_read_user_b_order(client):
    turn = await chat(client, ALICE, "What's the status of order ORD-2001?")
    assert turn["route"] == "order"
    assert "not found on your account" in turn["reply"]
    for secret in ("Mountain View", "Amphitheatre", "Keyboard", "pending"):
        assert secret not in turn["reply"]


async def test_user_a_cannot_change_user_b_address(client):
    before = await fetch_order("ORD-2001")
    turn = await chat(client, ALICE, CHANGE_1001.replace("ORD-1001", "ORD-2001"))
    assert turn["confirmation"] is None
    assert "not found on your account" in turn["reply"]
    assert await fetch_order("ORD-2001") == before


async def test_owner_can_read_their_order(client):
    turn = await chat(client, BOB, "What's the status of order ORD-2001?")
    assert "ORD-2001 is pending" in turn["reply"]


# --- Routing -------------------------------------------------------------------------


async def test_knowledge_route_answers_with_a_citation(client):
    turn = await chat(client, ALICE, "What is your refund policy for credit cards?")
    assert turn["route"] == "knowledge"
    sources = next(d for n, d in turn["events"] if n == "sources")
    assert sources["sources"], "retrieval returned nothing"
    assert any(s.startswith("Refunds Policy § ") for s in sources["sources"])
    # The reply cites a retrieved passage as [Document § Section].
    assert f"[{sources['sources'][0]}]" in turn["reply"]


async def test_knowledge_follow_up_is_answered_from_policies(client):
    first = await chat(client, ALICE, "What is the warranty on electronics?")
    turn = await chat(
        client, ALICE, "What is the warranty policy for office chairs?", first["meta"]["session_id"]
    )
    sources = next(d for n, d in turn["events"] if n == "sources")
    assert "Warranty Policy § Furniture coverage" in sources["sources"]


async def _ticket_count(user_id: str) -> int:
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL) as conn:
        cur = await conn.execute("SELECT count(*) FROM tickets WHERE customer_id = %s", (user_id,))
        (count,) = await cur.fetchone()
    return count


async def test_smalltalk_gets_a_polite_reply_and_no_ticket(client):
    turn = await chat(client, ALICE, "Hello there!")
    assert turn["route"] == "smalltalk"
    assert turn["reply"]
    assert await _ticket_count(ALICE) == 0


async def test_out_of_scope_gets_a_polite_reply_and_no_ticket(client):
    turn = await chat(client, ALICE, "What's the weather like in Paris?")
    assert turn["route"] == "out_of_scope"
    assert "orders" in turn["reply"]
    assert await _ticket_count(ALICE) == 0


async def test_handoff_route_opens_ticket(client):
    turn = await chat(client, ALICE, "I want to talk to a human")
    assert turn["route"] == "handoff"
    assert "TCK-" in turn["reply"]
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL) as conn:
        (count,) = await (
            await conn.execute("SELECT count(*) FROM tickets WHERE customer_id = %s", (ALICE,))
        ).fetchone()
    assert count == 1


async def test_reply_is_streamed_as_tokens(client):
    turn = await chat(client, ALICE, "List my orders")
    assert turn["names"][0] == "meta" and turn["names"][-1] == "done"
    assert turn["names"].count("token") > 1
    assert turn["tokens"].strip() == turn["reply"]


# --- Memory and auth -----------------------------------------------------------------


async def test_turns_are_stored_in_short_term_memory(client):
    from redis.asyncio import Redis

    from tests.conftest import TEST_REDIS_URL

    first = await chat(client, ALICE, "List my orders", session_id="s-mem")
    await chat(client, ALICE, "What is your refund policy?", session_id="s-mem")
    redis = Redis.from_url(TEST_REDIS_URL)
    history = await redis.lrange("sp:session:s-mem:history", 0, -1)
    ttl = await redis.ttl("sp:session:s-mem:history")
    await redis.aclose()
    assert len(history) == 4
    assert first["reply"] in history[1].decode()
    assert ttl > 0


async def test_paused_turn_is_recorded_as_pending_then_replaced_on_confirmation(client):
    import json

    from redis.asyncio import Redis

    from tests.conftest import TEST_REDIS_URL

    turn = await chat(client, ALICE, CHANGE_1001, session_id="s-hitl")
    redis = Redis.from_url(TEST_REDIS_URL)
    pending = [json.loads(x) for x in await redis.lrange("sp:session:s-hitl:history", 0, -1)]
    assert len(pending) == 2
    assert pending[0]["content"] == CHANGE_1001
    assert "awaiting the customer's confirmation; it has NOT been applied" in pending[1]["actions"][-1]

    await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")
    done = [json.loads(x) for x in await redis.lrange("sp:session:s-hitl:history", 0, -1)]
    await redis.aclose()
    assert len(done) == 2  # replaced, not appended
    assert any("succeeded" in a for a in done[1]["actions"])
    assert "San Francisco" in done[1]["content"]


async def test_unconfirmed_turn_is_in_memory_for_the_next_turn(client):
    """P1 leftover: a customer who skips the confirmation and moves on must not lose that turn."""
    import json

    from redis.asyncio import Redis

    from tests.conftest import TEST_REDIS_URL

    await chat(client, ALICE, CHANGE_1001, session_id="s-skip")
    await chat(client, ALICE, "What is the status of ORD-1002?", session_id="s-skip")
    redis = Redis.from_url(TEST_REDIS_URL)
    history = [json.loads(x) for x in await redis.lrange("sp:session:s-skip:history", 0, -1)]
    await redis.aclose()
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == CHANGE_1001
    assert "NOT been applied" in history[1]["actions"][-1]
    assert (await fetch_order("ORD-1001"))["address_events"] == 0


async def test_session_cannot_be_hijacked_by_another_user(client):
    await chat(client, ALICE, "List my orders", session_id="s-owned")
    resp = await client.post(
        "/chat", json={"message": "List my orders", "session_id": "s-owned"}, headers={"X-User-Id": BOB}
    )
    assert resp.status_code == 403


async def test_chat_requires_known_user(client):
    assert (await client.post("/chat", json={"message": "hi"})).status_code == 401
    resp = await client.post("/chat", json={"message": "hi"}, headers={"X-User-Id": "cust_999"})
    assert resp.status_code == 401


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "postgres": "ok", "redis": "ok", "llm_provider": "rule_based"}


# --- Action log in memory (regression: model re-submitted a change it had already made) ---


async def test_memory_records_tool_calls_that_ran(client):
    import json

    from redis.asyncio import Redis

    from tests.conftest import TEST_REDIS_URL

    turn = await chat(client, ALICE, CHANGE_1001, session_id="s-log")
    await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")
    await chat(client, ALICE, "What's the status of order ORD-2001?", session_id="s-log")

    redis = Redis.from_url(TEST_REDIS_URL)
    history = [json.loads(x) for x in await redis.lrange("sp:session:s-log:history", 0, -1)]
    await redis.aclose()
    first_actions, second_actions = history[1]["actions"], history[3]["actions"]
    assert any(
        a.startswith("update_address(ORD-1001 -> 500 Market St") and "succeeded" in a for a in first_actions
    )
    assert second_actions == [
        "get_order(order_id=ORD-2001) FAILED: Order ORD-2001 was not found on your account."
    ]


async def test_rejected_change_is_logged_as_failed(client):
    import json

    from redis.asyncio import Redis

    from tests.conftest import TEST_REDIS_URL

    turn = await chat(client, ALICE, CHANGE_1001, session_id="s-rej")
    await confirm(client, ALICE, turn["confirmation"]["thread_id"], "reject")
    redis = Redis.from_url(TEST_REDIS_URL)
    history = [json.loads(x) for x in await redis.lrange("sp:session:s-rej:history", 0, -1)]
    await redis.aclose()
    assert "FAILED: The customer declined" in history[1]["actions"][0]


def test_action_log_is_rendered_into_the_next_turns_prompt():
    from supportpilot.graph.builder import _conversation

    state = {
        "user_message": "And ORD-1002?",
        "history": [
            {"role": "user", "content": "change it"},
            {"role": "assistant", "content": "Done.", "actions": ["update_address(ORD-1001 -> X) succeeded"]},
        ],
    }
    messages = _conversation(state)
    assert messages[1]["content"].startswith("[System record of tool calls in this turn:")
    assert "update_address(ORD-1001 -> X) succeeded" in messages[1]["content"]
    assert set(messages[1]) == {"role", "content"}


async def test_change_to_current_address_does_not_ask_for_confirmation(client):
    turn = await chat(client, ALICE, CHANGE_1001)
    await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")

    again = await chat(client, ALICE, CHANGE_1001)
    assert again["confirmation"] is None
    assert "already this address" in again["reply"]
    assert (await fetch_order("ORD-1001"))["address_events"] == 1
