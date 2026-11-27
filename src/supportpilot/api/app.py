"""HTTP API.

Authentication is mocked for this stage: the caller's customer id arrives in
the ``X-User-Id`` header (in production this would come from a verified
session or JWT). It is the *only* source of identity -- the agent and tools
never accept a user id from the model or the request body.
"""

import asyncio
import datetime as dt
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
from supportpilot.graph.builder import _action_log, pending_turn_record
from supportpilot.kb import RetrievalConfig, Retriever
from supportpilot.llm import LLMClient, build_llm
from supportpilot.llm.usage import metered
from supportpilot.longterm import BackgroundTasks, LongTermMemory, build_long_term_memory, worth_remembering
from supportpilot.mcp_server import build_mcp_server
from supportpilot.memory import ConversationMemory, SessionOwnershipError
from supportpilot.observability import configure_tracing
from supportpilot.tools import ToolGateway

log = logging.getLogger("supportpilot")


@dataclass
class AppContext:
    settings: Settings
    pool: AsyncConnectionPool
    redis: Redis
    memory: ConversationMemory
    graph: Any
    long_term: LongTermMemory
    background: BackgroundTasks
    tracer: Any


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    long_term: LongTermMemory | None = None,
) -> FastAPI:
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
            tracer = configure_tracing(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
                environment=settings.app_environment,
                enabled=settings.langfuse_enabled,
            )
            lt_memory = long_term or build_long_term_memory(settings)
            deps = GraphDeps(
                llm=llm or build_llm(settings),
                tools=ToolGateway(
                    build_mcp_server(pool, Retriever(pool, RetrievalConfig.from_settings(settings))),
                    guard=settings.injection_guard,
                ),
                memory=memory,
                max_agent_steps=settings.max_agent_steps,
                query_rewrite=settings.retrieval_query_rewrite,
                agentic_rag=settings.agentic_rag,
                agentic_max_rewrites=settings.agentic_rag_max_rewrites,
                token_limit=settings.conversation_token_limit,
                confirmation_ttl_seconds=settings.confirmation_ttl_seconds,
                injection_guard=settings.injection_guard,
                long_term=lt_memory,
                architecture=settings.agent_architecture,
            )
            background = BackgroundTasks()
            app.state.ctx = AppContext(
                settings, pool, redis, memory, build_graph(deps, checkpointer), lt_memory, background, tracer
            )
            yield
            await background.drain()
        finally:
            tracer.flush()
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

    def trace_tags(kind: str) -> list[str]:
        tags = [kind, f"env:{settings.app_environment}", f"model:{settings.anthropic_model}"]
        if settings.agentic_rag:
            tags.append("agentic_rag")
        return tags

    async def finish_turn(
        ctx: AppContext, user_id: str, session_id: str, thread_id: str, user_message: str, meter, root
    ) -> dict[str, Any]:
        """Book-keeping after a turn (or a resumed turn) ran: tokens, pending record, long-term memory."""
        total = await ctx.memory.add_tokens(session_id, meter.total_tokens)
        state = await ctx.graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = state.values
        if state.interrupts:
            # Paused for confirmation: record the turn now, so it is in the conversation even if the
            # customer never answers. Completing the turn replaces this entry (same turn id).
            prompt, actions = pending_turn_record(values, state.interrupts[0].value["message"])
            await ctx.memory.append_turn(session_id, user_message, prompt, actions=actions, turn_id=thread_id)
            result = {"thread_id": thread_id, **state.interrupts[0].value}
        else:
            reply = values.get("reply") or ""
            result = {"thread_id": thread_id, "reply": reply}
            actions = _action_log(values.get("messages") or [])
            exchange = [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
            if actions:
                exchange[1]["content"] = f"{reply}\n(Actions taken: {'; '.join(actions)})"
            if ctx.settings.long_term_memory_gate == "all" or worth_remembering(user_message, actions):
                remember = ctx.long_term.remember(
                    user_id, exchange, {"session_id": session_id, "thread_id": thread_id}
                )
                if ctx.settings.long_term_memory_sync:
                    await remember
                else:
                    ctx.background.spawn(remember)
        usage = {
            "turn_tokens": meter.total_tokens,
            "turn_cost_usd": round(meter.cost, 6),
            "session_tokens": total,
        }
        root.update(
            output=result.get("reply") or {"confirmation_required": result.get("message")},
            metadata={
                "route": values.get("intent"),
                "injection": values.get("injection"),
                "over_budget": bool(values.get("over_budget")),
                **usage,
                "by_model": meter.by_model(),
            },
        )
        return {"interrupted": bool(state.interrupts), "result": result, "usage": usage}

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
            "turn_id": thread_id,
            "tokens_used": await ctx.memory.tokens_used(session_id),
        }

        def sse(event: str, data: dict[str, Any]) -> dict[str, str]:
            return {"event": event, "data": json.dumps(data)}

        queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()

        async def run_turn() -> None:
            # Runs in its own task, so the trace context and the usage meter cover exactly one turn
            # and are not disturbed by the SSE generator being suspended between events.
            try:
                with (
                    metered() as meter,
                    ctx.tracer.trace(
                        "chat_turn",
                        user_id=user_id,
                        session_id=session_id,
                        tags=trace_tags("chat"),
                        metadata={"thread_id": thread_id},
                        input=body.message,
                    ) as root,
                ):
                    async for chunk in ctx.graph.astream(initial, config, stream_mode="custom"):
                        await queue.put(sse(chunk["type"], chunk))
                    done = await finish_turn(ctx, user_id, session_id, thread_id, body.message, meter, root)
                if done["interrupted"]:
                    await queue.put(sse("confirmation_required", done["result"]))
                else:
                    await queue.put(sse("final", done["result"]))
                await queue.put(sse("usage", done["usage"]))
            except Exception:
                log.exception("chat turn failed (thread %s)", thread_id)
                await queue.put(
                    sse(
                        "error",
                        {"thread_id": thread_id, "message": "Something went wrong. Please try again."},
                    )
                )
            finally:
                ctx.tracer.flush()
                await queue.put(None)

        async def events() -> AsyncIterator[dict[str, str]]:
            yield sse("meta", {"session_id": session_id, "thread_id": thread_id})
            task = ctx.background.spawn(run_turn())
            while (item := await queue.get()) is not None:
                yield item
            await task
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
            pending = state.interrupts[0].value
            expires_at = pending.get("expires_at")
            expired = bool(expires_at) and dt.datetime.now(dt.UTC) > dt.datetime.fromisoformat(expires_at)
            decision = "expired" if expired else body.decision
            session_id = state.values["session_id"]
            with (
                metered() as meter,
                ctx.tracer.trace(
                    "confirm",
                    user_id=user_id,
                    session_id=session_id,
                    tags=trace_tags("confirm"),
                    metadata={"thread_id": body.thread_id, "decision": decision},
                    input={"decision": body.decision, "pending": pending},
                ) as root,
            ):
                # An expired request is resumed too, so the turn ends cleanly (nothing applied) and the
                # conversation records that it expired.
                await ctx.graph.ainvoke(Command(resume=decision), config)
                await finish_turn(
                    ctx, user_id, session_id, body.thread_id, state.values["user_message"], meter, root
                )
            ctx.tracer.flush()
            state = await ctx.graph.aget_state(config)
        finally:
            await ctx.redis.delete(lock_key)

        if expired:
            raise HTTPException(
                410,
                {
                    "error": "confirmation_expired",
                    "message": (
                        f"This confirmation request expired at {expires_at}. Nothing was changed; "
                        "please ask again if you still want the change."
                    ),
                    "expires_at": expires_at,
                },
            )

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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
