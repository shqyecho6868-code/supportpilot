"""The three acceptance scenarios against real Claude.

Skipped unless ANTHROPIC_API_KEY is available (environment or .env). Run with:

    uv run pytest -m live -v
"""

import httpx
import pytest

from supportpilot.api.app import create_app
from supportpilot.config import Settings
from tests.conftest import ALICE, TEST_DATABASE_URL, TEST_REDIS_URL, chat, confirm, fetch_order, parse_sse
from tests.test_restart import Server, _free_port, _post

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not Settings().anthropic_api_key, reason="ANTHROPIC_API_KEY not set"),
]

CHANGE = (
    "Hi, please change the shipping address of order ORD-1001 to 500 Market St, San Francisco, CA 94105, US."
)


@pytest.fixture
async def live_client(clean_state):
    settings = Settings(
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        llm_provider="anthropic",
        long_term_memory=False,
        langfuse_enabled=False,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as c:
            yield c


async def _request_change(client) -> str:
    turn = await chat(client, ALICE, CHANGE)
    assert turn["error"] is None
    assert turn["confirmation"] is not None, f"model did not request the change: {turn['tokens']!r}"
    assert turn["confirmation"]["order_id"] == "ORD-1001"
    return turn["confirmation"]["thread_id"]


async def test_live_reject_or_ignore_keeps_address_and_double_approve_applies_once(live_client):
    before = await fetch_order("ORD-1001")

    # Reject.
    thread_id = await _request_change(live_client)
    resp = await confirm(live_client, ALICE, thread_id, "reject")
    assert resp.status_code == 200
    print("\n[reject] reply:", resp.json()["reply"])
    assert await fetch_order("ORD-1001") == before

    # Never confirm.
    await _request_change(live_client)
    assert await fetch_order("ORD-1001") == before

    # Approve twice.
    thread_id = await _request_change(live_client)
    first = await confirm(live_client, ALICE, thread_id, "approve")
    second = await confirm(live_client, ALICE, thread_id, "approve")
    print("[approve] reply:", first.json()["reply"])
    assert first.status_code == 200 and second.status_code == 409
    row = await fetch_order("ORD-1001")
    assert row["address"]["city"] == "San Francisco"
    assert row["address_events"] == 1


async def test_live_user_a_cannot_see_user_b_order(live_client):
    turn = await chat(
        live_client, ALICE, "Can you tell me the status and shipping address of order ORD-2001?"
    )
    print("\n[cross-user] reply:", turn["reply"])
    assert turn["error"] is None and turn["reply"]
    for secret in ("Mountain View", "Amphitheatre", "94043", "Keyboard"):
        assert secret not in turn["reply"]


async def test_live_pending_confirmation_survives_restart(clean_state):
    server = Server(_free_port(), llm_provider="anthropic")
    try:
        server.start()
        resp = await _post(
            server, "/chat", json={"message": CHANGE}, headers={"X-User-Id": ALICE}, timeout=120
        )
        events = dict(parse_sse(resp.text))
        assert "confirmation_required" in events, resp.text
        thread_id = events["confirmation_required"]["thread_id"]

        server.kill()
        assert (await fetch_order("ORD-1001"))["address_events"] == 0
        server.start()
        resp = await _post(
            server,
            "/confirm",
            json={"thread_id": thread_id, "decision": "approve"},
            headers={"X-User-Id": ALICE},
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        print("\n[restart] reply:", resp.json()["reply"])
        row = await fetch_order("ORD-1001")
        assert row["address"]["city"] == "San Francisco" and row["address_events"] == 1
    finally:
        server.kill()


async def test_live_follow_up_after_approved_change_does_not_resubmit(live_client):
    """Regression: after an approved change, an unrelated follow-up made the model 'correct itself'
    and submit the same change again, because memory held only text and no record of the tool call."""
    turn = await chat(live_client, ALICE, CHANGE, session_id="s-live-followup")
    assert turn["confirmation"] is not None
    assert (
        await confirm(live_client, ALICE, turn["confirmation"]["thread_id"], "approve")
    ).status_code == 200

    follow_up = await chat(live_client, ALICE, "What's the status of ORD-2001?", session_id="s-live-followup")
    print("\n[follow-up] reply:", follow_up["reply"])
    assert follow_up["confirmation"] is None, follow_up["tokens"]
    for secret in ("Mountain View", "Amphitheatre", "94043", "Keyboard"):
        assert secret not in (follow_up["reply"] or "")
    assert (await fetch_order("ORD-1001"))["address_events"] == 1
