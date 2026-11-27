"""Run golden cases through the real graph and record what happened.

Each case runs against a dedicated database (``supportpilot_eval``) whose order
tables are reset before the case, so mutations in one case cannot leak into the
next. Tool calls are recorded by wrapping the ToolGateway; the human
confirmation step is auto-approved so address changes can complete.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from psycopg import sql
from redis.asyncio import Redis

from supportpilot.db import create_pool, reset_database
from supportpilot.evals.cases import Case
from supportpilot.evals.judges import raise_if_fatal
from supportpilot.graph import GraphDeps, build_graph
from supportpilot.graph.builder import _rewrite_input, _sources_message, cited_sources
from supportpilot.graph.prompts import KNOWLEDGE_AGENT_PROMPT, QUERY_REWRITE_PROMPT
from supportpilot.kb import RetrievalConfig, Retriever
from supportpilot.kb.contextualize import Contextualizer
from supportpilot.kb.ingest import ingest
from supportpilot.llm import LLMClient
from supportpilot.llm.usage import metered
from supportpilot.mcp_server import build_mcp_server
from supportpilot.memory import ConversationMemory
from supportpilot.observability import get_tracer
from supportpilot.tools import ToolGateway, ToolResult

log = logging.getLogger(__name__)

EVAL_DB = "supportpilot_eval"


def eval_database_url(server_url: str, name: str = EVAL_DB) -> str:
    return f"{server_url.rsplit('/', 1)[0]}/{name}"


async def prepare_database(
    server_url: str, fast_model: str, embedding_model: str, name: str = EVAL_DB
) -> str:
    """Create the eval database if needed and (re)index the knowledge base offline."""
    async with await psycopg.AsyncConnection.connect(server_url, autocommit=True) as conn:
        cur = await conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        if await cur.fetchone() is None:
            await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = eval_database_url(server_url, name)
    await reset_database(url)
    await ingest(url, Contextualizer(fast_model, allow_generate=False), embedding_model)
    return url


class RecordingGateway(ToolGateway):
    def __init__(self, server, *, guard: bool = True):
        super().__init__(server, guard=guard)
        self.calls: list[dict[str, Any]] = []

    async def call(self, name, arguments, *, user_id, idempotency_key=None) -> ToolResult:
        result = await super().call(name, arguments, user_id=user_id, idempotency_key=idempotency_key)
        self.calls.append(
            {
                "name": name,
                "args": arguments,
                "is_error": result.is_error,
                "result": result.content,
                "injection_flags": list(result.injection_flags),
            }
        )
        return result


async def _snapshot(url: str, order_ids: tuple[str, ...]) -> dict[str, Any]:
    async with await psycopg.AsyncConnection.connect(url) as conn:
        cur = await conn.execute(
            "SELECT id, shipping_address, status FROM orders WHERE id = ANY(%s) ORDER BY id",
            (list(order_ids),),
        )
        orders = {r[0]: [r[1], r[2]] for r in await cur.fetchall()}
        (tickets,) = await (await conn.execute("SELECT count(*) FROM tickets")).fetchone()
    return {"orders": orders, "tickets": tickets}


@dataclass
class AgentRunner:
    database_url: str
    redis_url: str
    llm: LLMClient
    retrieval: RetrievalConfig
    query_rewrite: bool = True
    agentic_rag: bool = False
    injection_guard: bool = True
    architecture: str = "routed"  # "routed" (router + specialist agents) or "single" (one agent)
    trace_tags: tuple[str, ...] = ()
    run_id: str = ""

    async def run_case(self, case: Case) -> dict[str, Any]:
        await reset_database(self.database_url)
        redis = Redis.from_url(self.redis_url)
        await redis.flushdb()
        pool = create_pool(self.database_url, max_size=4)
        await pool.open(wait=True)
        try:
            with (
                metered() as meter,
                get_tracer().trace(
                    f"eval:{case.id}",
                    user_id=case.user_id,
                    session_id=self.run_id or None,
                    tags=["eval", f"category:{case.category}", *self.trace_tags],
                    metadata={"case_id": case.id, "run_id": self.run_id},
                    input=case.message,
                ) as root,
            ):
                record = await self._run(case, pool, redis)
                record["usage"] = meter.summary()
                record["trace_id"] = get_tracer().trace_id()
                root.update(output=record["reply"], metadata={"route": record["route"], **record["usage"]})
            return record
        except Exception as exc:  # noqa: BLE001 - record the failure, keep evaluating
            raise_if_fatal(exc)
            log.exception("case %s crashed", case.id)
            return {
                "id": case.id,
                "error": repr(exc),
                "route": None,
                "routes": [],
                "tools": [],
                "reply": "",
                "retrieval": None,
                "confirmation": None,
                "turns": None,
                "tickets_created": 0,
                "orders_before": {},
                "orders_after": {},
                "turn_seconds": [],
                "injection_flags": {"user_input": [], "tool_results": []},
            }
        finally:
            await pool.close()
            await redis.aclose()

    async def _run(self, case: Case, pool, redis: Redis) -> dict[str, Any]:
        memory = ConversationMemory(redis, max_messages=20, ttl_seconds=3600)
        session_id = f"eval-{case.id}-{uuid.uuid4().hex[:6]}"
        await memory.claim(session_id, case.user_id)
        history = list(case.history)
        for user, assistant in zip(history[::2], history[1::2], strict=True):
            await memory.append_turn(session_id, user["content"], assistant["content"])

        gateway = RecordingGateway(
            build_mcp_server(pool, Retriever(pool, self.retrieval)), guard=self.injection_guard
        )
        graph = build_graph(
            GraphDeps(
                llm=self.llm,
                tools=gateway,
                memory=memory,
                query_rewrite=self.query_rewrite,
                agentic_rag=self.agentic_rag,
                injection_guard=self.injection_guard,
                architecture=self.architecture,
            ),
            InMemorySaver(),
        )
        watched = tuple(
            sorted(
                {*case.protected_orders, *([case.expect_address["order_id"]] if case.expect_address else [])}
            )
        )
        before = await _snapshot(self.database_url, watched)

        # Every live message is one turn: its own graph thread, in the same session, as in the API.
        turns: list[dict[str, Any]] = []
        guard_flags: list[str] = []
        for message in case.messages:
            config = {"configurable": {"thread_id": f"{session_id}:{len(turns)}"}}
            initial = {
                "user_id": case.user_id,
                "session_id": session_id,
                "user_message": message,
                "history": await memory.history(session_id),
                "turn_id": config["configurable"]["thread_id"],
            }
            first_call = len(gateway.calls)
            started = time.monotonic()
            route = None
            async for chunk in graph.astream(initial, config, stream_mode="custom"):
                if chunk["type"] == "route":
                    route = chunk["intent"]
                elif chunk["type"] == "guard":
                    guard_flags.extend(chunk["matches"])
            state = await graph.aget_state(config)
            # The customer-facing latency of the turn: until the reply or the confirmation prompt.
            seconds = time.monotonic() - started
            confirmation = None
            # Auto-approve confirmations (at most a few, in case the agent chains changes).
            for _ in range(3):
                if not state.interrupts:
                    break
                confirmation = state.interrupts[0].value
                resumed = time.monotonic()
                await graph.ainvoke(Command(resume="approve"), config)
                seconds += time.monotonic() - resumed
                state = await graph.aget_state(config)
            values = state.values
            turns.append(
                {
                    "message": message,
                    "route": route,
                    "reply": values.get("reply") or "",
                    "tools": gateway.calls[first_call:],
                    "retrieval": values.get("retrieval"),
                    "confirmation": confirmation,
                    "seconds": round(seconds, 2),
                }
            )
        after = await _snapshot(self.database_url, watched)
        last = turns[-1]
        reply = last["reply"]
        retrieval = last["retrieval"]
        return {
            "id": case.id,
            "error": None,
            "route": last["route"],
            "routes": [t["route"] for t in turns],
            "tools": gateway.calls,
            "reply": reply,
            "retrieval": retrieval,
            "confirmation": next((t["confirmation"] for t in reversed(turns) if t["confirmation"]), None),
            "turns": turns if len(turns) > 1 else None,
            "tickets_created": after["tickets"] - before["tickets"],
            "orders_before": before["orders"],
            "orders_after": after["orders"],
            "model_cited": bool(
                retrieval and cited_sources(reply.split("\n\nSources: ")[0], retrieval["results"])
            ),
            "seconds": round(sum(t["seconds"] for t in turns), 2),
            "turn_seconds": [t["seconds"] for t in turns],
            "injection_flags": {
                "user_input": guard_flags,
                "tool_results": sorted({f for t in gateway.calls for f in t["injection_flags"]}),
            },
        }


async def rewrite_for_retrieval(llm: LLMClient, case: Case) -> str:
    """The search query the knowledge agent would use for this case."""
    if not case.history:
        return case.message
    conversation = [*case.history, {"role": "user", "content": case.message}]
    return await llm.rewrite_query(QUERY_REWRITE_PROMPT, _rewrite_input(conversation)) or case.message


async def answer_with(llm: LLMClient, retriever: Retriever, cfg: RetrievalConfig, case: Case, query: str):
    """The knowledge agent's retrieve-then-answer step, for one retrieval configuration."""
    hits = [h.as_dict() for h in await retriever.search(query, cfg)]
    messages = [*case.history, {"role": "user", "content": _sources_message(hits, case.message)}]
    turn = await llm.respond(KNOWLEDGE_AGENT_PROMPT, messages, [], lambda _t: None)
    return hits, turn.text


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
