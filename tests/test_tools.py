"""Tool-layer tests: the MCP server, called through the gateway exactly as the agent calls it."""

import asyncio
import json

import pytest
from mcp import Client

from supportpilot.mcp_server import build_mcp_server
from tests.conftest import ALICE, BOB, fetch_order

NEW_ADDRESS = {
    "street": "1 Infinite Loop",
    "city": "Cupertino",
    "state": "CA",
    "postal_code": "95014",
    "country": "US",
}


def data(result):
    assert not result.is_error, result.content
    return json.loads(result.content)


# --- tool surface -----------------------------------------------------------------


async def test_model_facing_schemas_never_expose_user_id_or_idempotency_key(gateway):
    tools = {t["name"]: t for t in await gateway.model_tools()}
    assert set(tools) == {"get_order", "list_my_orders", "update_address", "create_ticket"}
    for tool in tools.values():
        props = tool["input_schema"].get("properties", {})
        assert "user_id" not in props
        assert "idempotency_key" not in props
        assert "idempotency_key" not in tool["input_schema"].get("required", [])


async def test_mcp_call_without_identity_is_refused(pool):
    async with Client(build_mcp_server(pool)) as client:
        result = await client.call_tool("get_order", {"order_id": "ORD-1001"})
    assert result.is_error
    assert "Missing authenticated user context" in result.content[0].text


# --- get_order / list_my_orders ----------------------------------------------------


async def test_get_order_returns_own_order(gateway):
    order = data(await gateway.call("get_order", {"order_id": "ORD-1002"}, user_id=ALICE))
    assert order["order_id"] == "ORD-1002"
    assert order["status"] == "shipped"
    assert order["tracking_number"] == "1Z999AA10123456784"


async def test_get_order_normalizes_order_id(gateway):
    order = data(await gateway.call("get_order", {"order_id": " ord-1001 "}, user_id=ALICE))
    assert order["order_id"] == "ORD-1001"


async def test_get_order_of_another_user_looks_like_not_found(gateway):
    other = await gateway.call("get_order", {"order_id": "ORD-2001"}, user_id=ALICE)
    missing = await gateway.call("get_order", {"order_id": "ORD-9999"}, user_id=ALICE)
    assert other.is_error and missing.is_error
    assert "not found on your account" in other.content
    assert other.content.replace("2001", "X") == missing.content.replace("9999", "X")
    assert "Mountain View" not in other.content  # Bob's address never leaks


async def test_user_id_smuggled_into_arguments_is_ignored(gateway):
    result = await gateway.call("get_order", {"order_id": "ORD-2001", "user_id": BOB}, user_id=ALICE)
    assert result.is_error


async def test_list_my_orders_only_returns_own_orders(gateway):
    orders = data(await gateway.call("list_my_orders", {}, user_id=ALICE))["orders"]
    assert {o["order_id"] for o in orders} == {"ORD-1001", "ORD-1002", "ORD-1003"}


async def test_list_my_orders_filters_by_status(gateway):
    orders = data(await gateway.call("list_my_orders", {"status": "cancelled"}, user_id=BOB))["orders"]
    assert [o["order_id"] for o in orders] == ["ORD-2002"]


# --- update_address ----------------------------------------------------------------


async def test_update_address_on_pending_order(gateway):
    result = data(
        await gateway.call(
            "update_address", {"order_id": "ORD-1001", **NEW_ADDRESS}, user_id=ALICE, idempotency_key="k1"
        )
    )
    assert result["status"] == "updated" and result["replayed"] is False
    assert result["old_address"]["city"] == "Springfield"
    row = await fetch_order("ORD-1001")
    assert row["address"] == NEW_ADDRESS
    assert row["address_events"] == 1


@pytest.mark.parametrize("order_id,status", [("ORD-1002", "shipped"), ("ORD-1003", "delivered")])
async def test_update_address_rejected_unless_pending(gateway, order_id, status):
    before = await fetch_order(order_id)
    result = await gateway.call(
        "update_address", {"order_id": order_id, **NEW_ADDRESS}, user_id=ALICE, idempotency_key="k1"
    )
    assert result.is_error and status in result.content
    assert await fetch_order(order_id) == before


async def test_update_address_rejected_for_cancelled_order(gateway):
    result = await gateway.call(
        "update_address", {"order_id": "ORD-2002", **NEW_ADDRESS}, user_id=BOB, idempotency_key="k1"
    )
    assert result.is_error and "cancelled" in result.content


async def test_update_address_cannot_touch_another_users_order(gateway):
    before = await fetch_order("ORD-2001")
    result = await gateway.call(
        "update_address", {"order_id": "ORD-2001", **NEW_ADDRESS}, user_id=ALICE, idempotency_key="k1"
    )
    assert result.is_error and "not found on your account" in result.content
    assert await fetch_order("ORD-2001") == before


async def test_update_address_requires_idempotency_key(gateway):
    with pytest.raises(ValueError):
        await gateway.call("update_address", {"order_id": "ORD-1001", **NEW_ADDRESS}, user_id=ALICE)


async def test_model_supplied_idempotency_key_is_overridden(gateway):
    args = {"order_id": "ORD-1001", **NEW_ADDRESS, "idempotency_key": "model-chosen"}
    data(await gateway.call("update_address", args, user_id=ALICE, idempotency_key="server-key"))
    replay = data(await gateway.call("update_address", args, user_id=ALICE, idempotency_key="server-key"))
    assert replay["replayed"] is True


async def test_same_idempotency_key_applies_once(gateway):
    args = {"order_id": "ORD-1001", **NEW_ADDRESS}
    first = data(await gateway.call("update_address", args, user_id=ALICE, idempotency_key="same"))
    second = data(await gateway.call("update_address", args, user_id=ALICE, idempotency_key="same"))
    assert first["replayed"] is False and second["replayed"] is True
    assert second["old_address"] == first["old_address"]
    assert (await fetch_order("ORD-1001"))["address_events"] == 1


async def test_concurrent_calls_with_same_key_apply_once(gateway):
    args = {"order_id": "ORD-1001", **NEW_ADDRESS}
    results = await asyncio.gather(
        *[gateway.call("update_address", args, user_id=ALICE, idempotency_key="race") for _ in range(5)]
    )
    replayed = [data(r)["replayed"] for r in results]
    assert replayed.count(False) == 1 and replayed.count(True) == 4
    assert (await fetch_order("ORD-1001"))["address_events"] == 1


async def test_reusing_key_with_different_payload_is_rejected(gateway):
    data(
        await gateway.call(
            "update_address", {"order_id": "ORD-1001", **NEW_ADDRESS}, user_id=ALICE, idempotency_key="k"
        )
    )
    other = {**NEW_ADDRESS, "street": "2 Different St"}
    result = await gateway.call(
        "update_address", {"order_id": "ORD-1001", **other}, user_id=ALICE, idempotency_key="k"
    )
    assert result.is_error and "different request" in result.content
    assert (await fetch_order("ORD-1001"))["address"] == NEW_ADDRESS


async def test_idempotency_keys_are_scoped_per_user(gateway):
    data(
        await gateway.call(
            "update_address", {"order_id": "ORD-1001", **NEW_ADDRESS}, user_id=ALICE, idempotency_key="k"
        )
    )
    bob = data(
        await gateway.call(
            "update_address", {"order_id": "ORD-2001", **NEW_ADDRESS}, user_id=BOB, idempotency_key="k"
        )
    )
    assert bob["replayed"] is False and bob["order_id"] == "ORD-2001"


async def test_failed_update_does_not_burn_the_key(gateway):
    bad = await gateway.call(
        "update_address", {"order_id": "ORD-1002", **NEW_ADDRESS}, user_id=ALICE, idempotency_key="k"
    )
    assert bad.is_error
    ok = data(
        await gateway.call(
            "update_address", {"order_id": "ORD-1001", **NEW_ADDRESS}, user_id=ALICE, idempotency_key="k"
        )
    )
    assert ok["replayed"] is False


# --- create_ticket -----------------------------------------------------------------


async def test_create_ticket(gateway, pool):
    ticket = data(
        await gateway.call(
            "create_ticket",
            {"subject": "Damaged item", "description": "The mug arrived cracked.", "order_id": "ORD-1002"},
            user_id=ALICE,
        )
    )
    assert ticket["ticket_id"].startswith("TCK-") and ticket["status"] == "open"
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT customer_id, order_id FROM tickets")).fetchone()
    assert row == {"customer_id": ALICE, "order_id": "ORD-1002"}


async def test_create_ticket_rejects_another_users_order(gateway):
    result = await gateway.call(
        "create_ticket", {"subject": "x", "description": "y", "order_id": "ORD-2001"}, user_id=ALICE
    )
    assert result.is_error and "not found on your account" in result.content
