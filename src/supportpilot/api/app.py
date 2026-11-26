"""HTTP API.

Authentication is mocked for this stage: the caller's customer id arrives in
the ``X-User-Id`` header (in production this would come from a verified
session or JWT). It is the *only* source of identity -- the agent and tools
never accept a user id from the model or the request body.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis
from sse_starlette import EventSourceResponse

from supportpilot.api.schemas import ChatRequest, ConfirmRequest, ConfirmResponse
from supportpilot.config import Settings, get_settings
from supportpilot.db import create_pool
from supportpilot.graph import GraphDeps, build_graph
from supportpilot.llm import LLMClient, build_llm
from supportpilot.mcp_server import build_mcp_server
from supportpilot.memory import ConversationMemory, SessionOwnershipError
from supportpilot.tools import ToolGateway

log = logging.getLogger("supportpilot")


@dataclass
class AppContext:
    settings: Settings
    pool: AsyncConnectionPool
    redis: Redis
    memory: ConversationMemory
    graph: Any


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = create_pool(settings.database_url)
        await pool.open(wait=True)
        redis = Redis.from_url(settings.redis_url)
        try:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            memory = ConversationMemory(
                redis, max_messages=settings.memory_max_messages, ttl_seconds=settings.memory_ttl_seconds
            )
            deps = GraphDeps(
                llm=llm or build_llm(settings),
                tools=ToolGateway(build_mcp_server(pool)),
                memory=memory,
                max_agent_steps=settings.max_agent_steps,
            )
            app.state.ctx = AppContext(settings, pool, redis, memory, build_graph(deps, checkpointer))
            yield
        finally:
            await redis.aclose()
            await pool.close()

    app = FastAPI(title="SupportPilot", version="0.1.0", lifespan=lifespan)

    def get_ctx(request: Request) -> AppContext:
        return request.app.state.ctx

    async def current_user(
        ctx: Annotated[AppContext, Depends(get_ctx)],
        x_user_id: Annotated[str | None, Header()] = None,
    ) -> str:
        if not x_user_id:
            raise HTTPException(401, "Missing X-User-Id header")
        async with ctx.pool.connection() as conn:
            cur = await conn.execute("SELECT 1 FROM customers WHERE id = %s", (x_user_id,))
            if await cur.fetchone() is None:
                raise HTTPException(401, "Unknown user")
        return x_user_id

    @app.get("/health")
    async def health(ctx: Annotated[AppContext, Depends(get_ctx)]) -> JSONResponse:
        checks: dict[str, str] = {}
        try:
            async with ctx.pool.connection() as conn:
                await conn.execute("SELECT 1")
            checks["postgres"] = "ok"
        except Exception as exc:  # noqa: BLE001 - report, don't raise
            checks["postgres"] = f"error: {exc.__class__.__name__}"
        try:
            await ctx.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["redis"] = f"error: {exc.__class__.__name__}"
        healthy = all(v == "ok" for v in checks.values())
        body = {
            "status": "ok" if healthy else "degraded",
            **checks,
            "llm_provider": ctx.settings.llm_provider,
        }
        return JSONResponse(body, status_code=200 if healthy else 503)

    @app.post("/chat")
    async def chat(
        body: ChatRequest,
        ctx: Annotated[AppContext, Depends(get_ctx)],
        user_id: Annotated[str, Depends(current_user)],
    ) -> EventSourceResponse:
        session_id = body.session_id or uuid.uuid4().hex
        try:
            await ctx.memory.claim(session_id, user_id)
        except SessionOwnershipError:
            raise HTTPException(403, "This session belongs to another user") from None

        thread_id = f"{session_id}:{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "user_id": user_id,
            "session_id": session_id,
            "user_message": body.message,
            "history": await ctx.memory.history(session_id),
        }

        def sse(event: str, data: dict[str, Any]) -> dict[str, str]:
            return {"event": event, "data": json.dumps(data)}

        async def events() -> AsyncIterator[dict[str, str]]:
            yield sse("meta", {"session_id": session_id, "thread_id": thread_id})
            try:
                async for chunk in ctx.graph.astream(initial, config, stream_mode="custom"):
                    yield sse(chunk["type"], chunk)
                state = await ctx.graph.aget_state(config)
                if state.interrupts:
                    yield sse("confirmation_required", {"thread_id": thread_id, **state.interrupts[0].value})
                else:
                    yield sse("final", {"thread_id": thread_id, "reply": state.values.get("reply")})
            except Exception:
                log.exception("chat turn failed (thread %s)", thread_id)
                yield sse(
                    "error", {"thread_id": thread_id, "message": "Something went wrong. Please try again."}
                )
            yield sse("done", {})

        return EventSourceResponse(events())

    @app.post("/confirm")
    async def confirm(
        body: ConfirmRequest,
        ctx: Annotated[AppContext, Depends(get_ctx)],
        user_id: Annotated[str, Depends(current_user)],
    ) -> ConfirmResponse:
        config = {"configurable": {"thread_id": body.thread_id}}
        state = await ctx.graph.aget_state(config)
        if not state.values or state.values.get("user_id") != user_id:
            raise HTTPException(404, "Unknown thread")

        # One resume per thread at a time; the tool's idempotency key backs this up.
        lock_key = f"sp:lock:confirm:{body.thread_id}"
        if not await ctx.redis.set(lock_key, user_id, nx=True, ex=120):
            raise HTTPException(409, "This confirmation is already being processed")
        try:
            state = await ctx.graph.aget_state(config)
            if not state.interrupts:
                raise HTTPException(409, "No pending confirmation for this thread")
            await ctx.graph.ainvoke(Command(resume=body.decision), config)
            state = await ctx.graph.aget_state(config)
        finally:
            await ctx.redis.delete(lock_key)

        if state.interrupts:
            return ConfirmResponse(
                status="confirmation_required",
                thread_id=body.thread_id,
                decision=body.decision,
                confirmation=state.interrupts[0].value,
            )
        return ConfirmResponse(
            status="completed",
            thread_id=body.thread_id,
            decision=body.decision,
            reply=state.values.get("reply"),
        )

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
