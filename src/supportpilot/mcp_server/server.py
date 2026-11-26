"""MCP server exposing the order tools and the policy knowledge base.

Security model
--------------
No tool accepts a ``user_id`` argument. The caller's identity travels in the
MCP request ``_meta`` under ``USER_ID_META_KEY``, which is set by the trusted
orchestrator from the authenticated session -- never by the model. Every query
is scoped by that id, so the model cannot read or modify another customer's
orders even if it is prompt-injected into asking for them. Orders that exist
but belong to someone else are reported exactly like orders that do not exist,
so the tools cannot be used to probe for valid order numbers.
"""

import hashlib
import json
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import Field

from supportpilot.kb import Retriever

USER_ID_META_KEY = "supportpilot/user_id"
IDEMPOTENCY_KEY_PARAM = "idempotency_key"

ORDER_NOT_FOUND = "Order {order_id} was not found on your account."

OrderStatus = Literal["pending", "shipped", "delivered", "cancelled"]


class ToolInputError(ToolError):
    """An expected failure (not found, wrong status, ...) whose message is shown to the model.

    Any other exception is treated by the MCP SDK as a crash and its text is withheld.
    """


def _current_user(ctx: Context) -> str:
    meta = ctx.request_context.meta or {}
    user_id = meta.get(USER_ID_META_KEY)
    if not isinstance(user_id, str) or not user_id:
        # Fail closed: a call without an authenticated identity is never served.
        raise ToolInputError("Missing authenticated user context.")
    return user_id


def _serialize_order(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": row["id"],
        "status": row["status"],
        "items": row["items"],
        "total": f"{row['total_cents'] / 100:.2f} {row['currency']}",
        "shipping_address": row["shipping_address"],
        "carrier": row["carrier"],
        "tracking_number": row["tracking_number"],
        "placed_at": row["placed_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def _request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_mcp_server(pool: AsyncConnectionPool, retriever: Retriever | None = None) -> MCPServer:
    retriever = retriever or Retriever(pool)
    server = MCPServer(
        "supportpilot",
        instructions=(
            "Order management tools for the authenticated SupportPilot customer, "
            "and search over the store's customer-service policies."
        ),
    )

    @server.tool()
    async def search_policy(
        query: Annotated[str, Field(description="A standalone search query about store policy.")],
        ctx: Context,
        top_k: Annotated[int | None, Field(ge=1, le=10, description="Number of passages to return.")] = None,
    ) -> dict[str, Any]:
        """Search the store's policies (returns, refunds, shipping, exchanges, warranty, payment,
        Larkspur Plus membership, privacy). Returns the best-matching passages, each with a
        `source` ("<document> § <section>") to cite."""
        _current_user(ctx)  # policy text is not per-customer, but unauthenticated calls are still refused
        config = retriever.config if top_k is None else retriever.config.with_(top_k=top_k)
        hits = await retriever.search(query, config)
        return {"query": query, "results": [h.as_dict() for h in hits]}

    @server.tool()
    async def get_order(
        order_id: Annotated[str, Field(description="Order number, e.g. ORD-1001.")],
        ctx: Context,
    ) -> dict[str, Any]:
        """Look up one of the customer's orders: status, items, total, shipping address and tracking."""
        user_id = _current_user(ctx)
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM orders WHERE id = %s AND customer_id = %s",
                (order_id.strip().upper(), user_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise ToolInputError(ORDER_NOT_FOUND.format(order_id=order_id))
        return _serialize_order(row)

    @server.tool()
    async def list_my_orders(
        ctx: Context,
        status: Annotated[OrderStatus | None, Field(description="Only return orders in this status.")] = None,
    ) -> dict[str, Any]:
        """List the customer's orders, newest first, optionally filtered by status."""
        user_id = _current_user(ctx)
        sql = "SELECT * FROM orders WHERE customer_id = %s"
        params: list[Any] = [user_id]
        if status:
            sql += " AND status = %s"
            params.append(status)
        sql += " ORDER BY placed_at DESC"
        async with pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        return {"orders": [_serialize_order(r) for r in rows]}

    @server.tool()
    async def update_address(
        order_id: Annotated[str, Field(description="Order number, e.g. ORD-1001.")],
        street: Annotated[str, Field(description="Street address, including unit/apartment.")],
        city: str,
        state: Annotated[str, Field(description="State or province code, e.g. CA.")],
        postal_code: str,
        country: Annotated[str, Field(description="ISO 3166-1 alpha-2 country code, e.g. US.")],
        idempotency_key: Annotated[
            str, Field(description="Unique key for this change; retries with the same key apply once.")
        ],
        ctx: Context,
    ) -> dict[str, Any]:
        """Change the shipping address of an order. Only orders in `pending` status can be changed."""
        user_id = _current_user(ctx)
        order_id = order_id.strip().upper()
        new_address = {
            "street": street.strip(),
            "city": city.strip(),
            "state": state.strip(),
            "postal_code": postal_code.strip(),
            "country": country.strip().upper(),
        }
        request_hash = _request_hash({"order_id": order_id, "address": new_address})

        async with pool.connection() as conn, conn.transaction():
            # Serialize concurrent calls that share a key, then check for a replay.
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{user_id}:{idempotency_key}",)
            )
            cur = await conn.execute(
                "SELECT tool_name, request_hash, response FROM idempotency_keys "
                "WHERE user_id = %s AND key = %s",
                (user_id, idempotency_key),
            )
            existing = await cur.fetchone()
            if existing is not None:
                if existing["tool_name"] != "update_address" or existing["request_hash"] != request_hash:
                    raise ToolInputError("This idempotency key was already used for a different request.")
                return {**existing["response"], "replayed": True}

            cur = await conn.execute(
                "SELECT * FROM orders WHERE id = %s AND customer_id = %s FOR UPDATE",
                (order_id, user_id),
            )
            order = await cur.fetchone()
            if order is None:
                raise ToolInputError(ORDER_NOT_FOUND.format(order_id=order_id))
            if order["status"] != "pending":
                raise ToolInputError(
                    f"Order {order_id} is {order['status']}; the address can only be changed while "
                    "the order is pending."
                )

            await conn.execute(
                "UPDATE orders SET shipping_address = %s, updated_at = now() WHERE id = %s",
                (Jsonb(new_address), order_id),
            )
            await conn.execute(
                "INSERT INTO order_events (order_id, event_type, payload) VALUES (%s, 'address_updated', %s)",
                (
                    order_id,
                    Jsonb({"old": order["shipping_address"], "new": new_address, "key": idempotency_key}),
                ),
            )
            response = {
                "order_id": order_id,
                "status": "updated",
                "old_address": order["shipping_address"],
                "new_address": new_address,
            }
            await conn.execute(
                "INSERT INTO idempotency_keys (user_id, key, tool_name, request_hash, response) "
                "VALUES (%s, %s, 'update_address', %s, %s)",
                (user_id, idempotency_key, request_hash, Jsonb(response)),
            )
        return {**response, "replayed": False}

    @server.tool()
    async def create_ticket(
        subject: Annotated[str, Field(description="One-line summary of the issue.")],
        description: Annotated[str, Field(description="Details a human agent needs to pick this up.")],
        ctx: Context,
        order_id: Annotated[str | None, Field(description="Related order number, if any.")] = None,
        priority: Literal["low", "normal", "high"] = "normal",
    ) -> dict[str, Any]:
        """Open a support ticket for a human agent to follow up on."""
        user_id = _current_user(ctx)
        if order_id:
            order_id = order_id.strip().upper()
        async with pool.connection() as conn, conn.transaction():
            if order_id:
                cur = await conn.execute(
                    "SELECT 1 FROM orders WHERE id = %s AND customer_id = %s", (order_id, user_id)
                )
                if await cur.fetchone() is None:
                    raise ToolInputError(ORDER_NOT_FOUND.format(order_id=order_id))
            cur = await conn.execute(
                "INSERT INTO tickets (id, customer_id, order_id, subject, description, priority) "
                "VALUES ('TCK-' || nextval('ticket_seq'), %s, %s, %s, %s, %s) RETURNING id, status",
                (user_id, order_id, subject.strip(), description.strip(), priority),
            )
            ticket = await cur.fetchone()
        return {"ticket_id": ticket["id"], "status": ticket["status"], "priority": priority}

    return server
