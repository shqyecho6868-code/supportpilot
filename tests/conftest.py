"""Shared fixtures.

Tests run against the docker-compose Postgres and Redis, but use a separate
database (``supportpilot_test``) and Redis DB 15, rebuilt before every test.
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import psycopg
import pytest
from psycopg import sql
from redis.asyncio import Redis

from supportpilot.config import Settings
from supportpilot.db import create_pool, reset_database
from supportpilot.llm.rule_based import RuleBasedLLM
from supportpilot.mcp_server import build_mcp_server
from supportpilot.tools import ToolGateway

PG_SERVER = os.getenv("TEST_PG_SERVER", "postgresql://supportpilot:supportpilot@localhost:5432")
TEST_DB = "supportpilot_test"
TEST_DATABASE_URL = f"{PG_SERVER}/{TEST_DB}"
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")

ALICE, BOB = "cust_001", "cust_002"


@pytest.fixture(scope="session", autouse=True)
def _test_database() -> None:
    with psycopg.connect(f"{PG_SERVER}/supportpilot", autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))


@pytest.fixture(scope="session", autouse=True)
def _knowledge_base(_test_database) -> None:
    """Index the policy documents into the test database once per session.

    Chunk contexts come from kb/contexts.json (offline); embeddings use the local fastembed model.
    The KB tables are not touched by reset_database(), so they survive per-test resets.
    """
    import asyncio

    from supportpilot.kb.contextualize import Contextualizer
    from supportpilot.kb.ingest import ingest

    defaults = Settings(_env_file=None)
    ctx = Contextualizer(defaults.fast_model, allow_generate=False)
    asyncio.run(ingest(TEST_DATABASE_URL, ctx, defaults.embedding_model))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url=TEST_DATABASE_URL,
        redis_url=TEST_REDIS_URL,
        llm_provider="rule_based",
        anthropic_api_key=None,
    )


@pytest.fixture
async def clean_state() -> AsyncIterator[None]:
    await reset_database(TEST_DATABASE_URL)
    redis = Redis.from_url(TEST_REDIS_URL)
    await redis.flushdb()
    await redis.aclose()
    yield


@pytest.fixture
async def pool(clean_state):
    pool = create_pool(TEST_DATABASE_URL)
    await pool.open(wait=True)
    yield pool
    await pool.close()


@pytest.fixture
def gateway(pool) -> ToolGateway:
    return ToolGateway(build_mcp_server(pool))


@pytest.fixture
async def client(settings, clean_state) -> AsyncIterator[httpx.AsyncClient]:
    from supportpilot.api.app import create_app

    app = create_app(settings, llm=RuleBasedLLM())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
            yield c


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for raw in text.replace("\r\n", "\n").split("\n\n"):
        event, data = None, []
        for line in raw.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].strip())
        if event:
            events.append((event, json.loads("\n".join(data)) if data else {}))
    return events


async def chat(
    client: httpx.AsyncClient, user_id: str, message: str, session_id: str | None = None
) -> dict[str, Any]:
    """POST /chat and collapse the SSE stream into a summary dict."""
    resp = await client.post(
        "/chat", json={"message": message, "session_id": session_id}, headers={"X-User-Id": user_id}
    )
    resp.raise_for_status()
    events = parse_sse(resp.text)
    by_type: dict[str, Any] = {}
    for name, data in events:
        by_type.setdefault(name, data)
    return {
        "events": events,
        "names": [n for n, _ in events],
        "meta": by_type.get("meta"),
        "route": by_type.get("route", {}).get("intent"),
        "tokens": "".join(d["text"] for n, d in events if n == "token"),
        "confirmation": by_type.get("confirmation_required"),
        "reply": by_type.get("final", {}).get("reply"),
        "error": by_type.get("error"),
    }


async def confirm(client: httpx.AsyncClient, user_id: str, thread_id: str, decision: str) -> httpx.Response:
    return await client.post(
        "/confirm", json={"thread_id": thread_id, "decision": decision}, headers={"X-User-Id": user_id}
    )


async def fetch_order(order_id: str) -> dict[str, Any]:
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL) as conn:
        cur = await conn.execute("SELECT shipping_address, status FROM orders WHERE id = %s", (order_id,))
        address, status = await cur.fetchone()
        cur = await conn.execute(
            "SELECT count(*) FROM order_events WHERE order_id = %s AND event_type = 'address_updated'",
            (order_id,),
        )
        (events,) = await cur.fetchone()
    return {"address": address, "status": status, "address_events": events}
