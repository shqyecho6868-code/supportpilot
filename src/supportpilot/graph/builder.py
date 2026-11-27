"""LangGraph orchestration.

    START -> router -+-> order_agent <-> confirm   (confirm pauses with interrupt())
                     +-> knowledge_agent            (rewrite query -> search_policy -> cited answer;
                     |                               agentic mode: decide -> search -> grade -> rewrite)
                     +-> chitchat                   (smalltalk / out_of_scope: polite reply, no ticket)
                     +-> handoff                    (opens a ticket for a human)
                     +-> token_limit                (conversation token limit reached: ticket + goodbye)
             all of them end in save_memory -> END

With ``architecture="single"`` (an evaluation baseline) there is no router: one agent holds every
tool (orders, update_address, create_ticket, search_policy) and handles every message itself.

    START -> screen -> agent <-> confirm -> save_memory -> END

Before routing, the router screens the customer's message with the injection guard and recalls
long-term memories; both reach the agents as per-turn system context.

The confirm node is the human-in-the-loop gate: it pauses the graph before any
tool in CONFIRMATION_REQUIRED runs. Because the graph is checkpointed to
Postgres, a paused turn survives a process restart and resumes from /confirm.
"""

import datetime as dt
import html
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from supportpilot.graph.prompts import (
    ACTION_LOG_PREFIX,
    CHITCHAT_PROMPT,
    CONFIRMATION_EXPIRED_RESULT,
    KNOWLEDGE_AGENT_PROMPT,
    KNOWLEDGE_FROM_CONVERSATION_PROMPT,
    LONG_TERM_MEMORY_CONTEXT,
    NO_POLICY_REPLY,
    ORDER_AGENT_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRIEVAL_DECISION_PROMPT,
    RETRIEVAL_GRADE_PROMPT,
    ROUTER_PROMPT,
    SINGLE_AGENT_PROMPT,
    TOKEN_LIMIT_REPLY,
)
from supportpilot.graph.state import TurnState
from supportpilot.guard import USER_NOTICE, neutralize, scan
from supportpilot.llm import LLMClient
from supportpilot.llm.usage import current_meter
from supportpilot.longterm import LongTermMemory, NullLongTermMemory
from supportpilot.memory import ConversationMemory
from supportpilot.observability import get_tracer
from supportpilot.tools import CONFIRMATION_REQUIRED, ORDER_AGENT_TOOLS, SINGLE_AGENT_TOOLS, ToolGateway

STEP_LIMIT_REPLY = "Sorry, I wasn't able to finish that. I can open a ticket for a human agent if you'd like."
DECLINED_RESULT = (
    "The customer declined this change at the confirmation step. Nothing was changed. "
    "Do not retry unless the customer asks again."
)


# Most passages handed to the answer after several agentic retrieval rounds.
MAX_AGENTIC_PASSAGES = 10

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "retrieve": {"type": "boolean"},
        "query": {"type": "string"},
    },
    "required": ["reason", "retrieve", "query"],
    "additionalProperties": False,
}
_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "missing": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "next_query": {"type": "string"},
    },
    "required": ["missing", "sufficient", "next_query"],
    "additionalProperties": False,
}


@dataclass
class GraphDeps:
    llm: LLMClient
    tools: ToolGateway
    memory: ConversationMemory
    max_agent_steps: int = 8
    query_rewrite: bool = True
    agentic_rag: bool = False
    agentic_max_rewrites: int = 2
    token_limit: int = 0  # 0 = no limit
    confirmation_ttl_seconds: int = 900
    injection_guard: bool = True
    long_term: LongTermMemory = NullLongTermMemory()
    # "routed": router + specialist agents (production). "single": one agent with every tool.
    architecture: Literal["routed", "single"] = "routed"


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}


def _render(entry: dict[str, Any]) -> dict[str, Any]:
    """Turn a memory entry into a message, prefixing the system's record of tool activity."""
    content = entry["content"]
    if entry["role"] == "user":
        return {"role": "user", "content": neutralize(content)}
    if entry.get("actions"):
        content = f"{ACTION_LOG_PREFIX} {'; '.join(entry['actions'])}]\n{content}"
    return {"role": entry["role"], "content": content}


def _conversation(state: TurnState) -> list[dict[str, Any]]:
    history = list(state.get("history") or [])
    while history and history[0]["role"] != "user":  # API requires a user turn first
        history.pop(0)
    return [*(_render(e) for e in history), {"role": "user", "content": neutralize(state["user_message"])}]


def _turn_context(state: TurnState) -> str | None:
    """Per-turn system context: long-term memories and the injection-guard notice."""
    parts = []
    if notes := state.get("long_term"):
        parts.append(LONG_TERM_MEMORY_CONTEXT.format(notes="\n".join(f"- {n}" for n in notes)))
    if (verdict := state.get("injection")) and verdict.get("flagged"):
        parts.append(USER_NOTICE.format(matches=", ".join(verdict["matches"])))
    return "\n\n".join(parts) or None


def _over_budget(state: TurnState, limit: int) -> bool:
    if not limit:
        return False
    meter = current_meter()
    return state.get("tokens_used", 0) + (meter.total_tokens if meter else 0) >= limit


def _describe_args(args: dict[str, Any]) -> str:
    if "street" in args:
        address = ", ".join(
            str(args.get(k, "")) for k in ("street", "city", "state", "postal_code", "country")
        )
        return f"{args.get('order_id', '?')} -> {address}"
    return ", ".join(f"{k}={v}" for k, v in args.items() if k != "description") or "-"


def _action_log(messages: list[dict[str, Any]]) -> list[str]:
    """Summarize every tool call that actually ran this turn, and how it ended."""
    calls: dict[str, dict[str, Any]] = {}
    log: list[str] = []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block["type"] == "tool_use":
                calls[block["id"]] = block
            elif block["type"] == "tool_result" and block["tool_use_id"] in calls:
                call = calls[block["tool_use_id"]]
                if block["is_error"]:
                    outcome = "FAILED: " + block["content"].removeprefix(
                        f"Error executing tool {call['name']}: "
                    )
                else:
                    outcome = "succeeded"
                log.append(f"{call['name']}({_describe_args(call['input'])}) {outcome[:200]}")
    return log


_CITATION = re.compile(r"\[([^\[\]]+? § [^\[\]]+?)\]")


def _sources_message(hits: list[dict[str, Any]], question: str) -> str:
    passages = "\n".join(f'<source cite="{html.escape(h["source"])}">\n{h["text"]}\n</source>' for h in hits)
    return f"<sources>\n{passages}\n</sources>\n\nCustomer question: {question}"


def cited_sources(reply: str, hits: list[dict[str, Any]]) -> list[str]:
    """Citations in the reply that match a retrieved passage."""
    valid = {h["source"] for h in hits}
    return [c for c in dict.fromkeys(_CITATION.findall(reply)) if c in valid]


def _rewrite_input(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The whole conversation as one user message, so it is plainly data to the rewriter."""
    earlier = "\n".join(f"{m['role']}: {m['content']}" for m in conversation[:-1])
    return [
        {
            "role": "user",
            "content": f"<conversation>\n{earlier}\n</conversation>\n\n"
            f"<latest_message>\n{conversation[-1]['content']}\n</latest_message>",
        }
    ]


def _emit_text(text: str) -> None:
    get_stream_writer()({"type": "token", "text": text})


def pending_turn_record(values: dict[str, Any], prompt: str) -> tuple[str, list[str]]:
    """What short-term memory stores for a turn paused at the confirmation step."""
    action = values["pending_action"]
    actions = _action_log(values.get("messages") or [])
    actions.append(
        f"{action['name']}({_describe_args(action['input'])}) is awaiting the customer's confirmation; "
        "it has NOT been applied"
    )
    return prompt, actions


def _add_search(retrieval: dict[str, Any] | None, query: str, content: str) -> dict[str, Any]:
    """Record a search_policy call made by a tool-using agent, like the knowledge agent's retrieval."""
    hits = json.loads(content).get("results") or []
    retrieval = retrieval or {"query": query, "queries": [], "results": []}
    seen = {h["chunk_id"] for h in retrieval["results"]}
    return {
        **retrieval,
        "queries": [*retrieval["queries"], query],
        "results": [*retrieval["results"], *(h for h in hits if h["chunk_id"] not in seen)],
    }


def _grade_input(question: str, queries: list[str], hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passages = "\n".join(f'<source cite="{html.escape(h["source"])}">\n{h["text"]}\n</source>' for h in hits)
    earlier = "\n".join(f"- {q}" for q in queries)
    return [
        {
            "role": "user",
            "content": f"<question>\n{question}\n</question>\n\n"
            f"<queries_so_far>\n{earlier}\n</queries_so_far>\n\n<sources>\n{passages}\n</sources>",
        }
    ]


def build_graph(deps: GraphDeps, checkpointer: BaseCheckpointSaver):
    tracer = get_tracer()

    async def screen(state: TurnState) -> dict[str, Any]:
        """Injection guard on the customer's message, and long-term memory recall."""
        writer = get_stream_writer()
        updates: dict[str, Any] = {}
        if deps.injection_guard:
            with tracer.span("guard.user_input", as_type="guardrail", input=state["user_message"]) as obs:
                verdict = scan(state["user_message"])
                obs.update(output=verdict.as_dict(), level="WARNING" if verdict.flagged else None)
            updates["injection"] = verdict.as_dict()
            if verdict.flagged:
                writer({"type": "guard", "source": "user_input", "matches": list(verdict.matches)})
        updates["long_term"] = await deps.long_term.recall(state["user_id"], state["user_message"])
        return updates

    async def router(state: TurnState) -> dict[str, Any]:
        writer = get_stream_writer()
        if _over_budget(state, deps.token_limit):
            writer({"type": "route", "intent": "token_limit"})
            return {"intent": "token_limit"}

        updates = await screen(state)
        with tracer.span("router", as_type="chain", input=state["user_message"]) as obs:
            intent = await deps.llm.classify_intent(ROUTER_PROMPT, _conversation(state))
            obs.update(output={"intent": intent})
        writer({"type": "route", "intent": intent})
        return {**updates, "intent": intent}

    async def screen_only(state: TurnState) -> dict[str, Any]:
        """Single-agent entry: the same guard and recall as the router, but no classification."""
        writer = get_stream_writer()
        if _over_budget(state, deps.token_limit):
            writer({"type": "route", "intent": "token_limit"})
            return {"intent": "token_limit"}
        updates = await screen(state)
        writer({"type": "route", "intent": "agent"})
        return {**updates, "intent": "agent"}

    def tool_agent(prompt: str, tool_names: frozenset[str], name: str):
        """A tool-use loop node: the order agent, or the single agent that holds every tool."""

        async def agent(state: TurnState) -> dict[str, Any]:
            return await _tool_agent_step(state, prompt, tool_names, name)

        return agent

    async def _tool_agent_step(
        state: TurnState, prompt: str, tool_names: frozenset[str], name: str
    ) -> dict[str, Any]:
        messages = state.get("messages") or _conversation(state)
        steps = state.get("steps", 0)
        if steps >= deps.max_agent_steps:
            _emit_text(STEP_LIMIT_REPLY)
            return {"reply": STEP_LIMIT_REPLY}
        if steps and _over_budget(state, deps.token_limit):
            return {"over_budget": True, "messages": messages}

        turn = await deps.llm.respond(
            prompt,
            messages,
            await deps.tools.model_tools(tool_names),
            _emit_text,
            system_suffix=_turn_context(state),
            name=name,
        )
        messages = [*messages, {"role": "assistant", "content": turn.content}]
        if not turn.tool_uses:
            return {"messages": messages, "reply": turn.text, "steps": steps + 1, "pending_action": None}

        # Run read-only tools now; hold back at most one mutating tool for confirmation.
        retrieval = state.get("retrieval")
        results: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None
        for call in turn.tool_uses:
            if call["name"] in CONFIRMATION_REQUIRED:
                if pending is None:
                    pending = {
                        "tool_use_id": call["id"],
                        "name": call["name"],
                        "input": call["input"],
                        # Fixed before the pause, so every retry of this action reuses the same key.
                        "idempotency_key": f"{state['session_id']}:{uuid.uuid4().hex}",
                        # Fixed here too (this node does not re-run on resume; confirm does).
                        "expires_at": time.time() + deps.confirmation_ttl_seconds,
                    }
                else:
                    results.append(
                        _tool_result(call["id"], "Only one change can be confirmed at a time.", True)
                    )
                continue
            result = await deps.tools.call(call["name"], call["input"], user_id=state["user_id"])
            results.append(_tool_result(call["id"], result.content, result.is_error))
            if call["name"] == "search_policy" and not result.is_error:
                retrieval = _add_search(retrieval, call["input"].get("query", ""), result.content)
                hits = json.loads(result.content).get("results") or []
                get_stream_writer()(
                    {
                        "type": "sources",
                        "query": call["input"].get("query"),
                        "sources": [h["source"] for h in hits],
                    }
                )

        if pending is not None:
            pending["completed_results"] = results
            return {
                "messages": messages,
                "pending_action": pending,
                "steps": steps + 1,
                "reply": None,
                "retrieval": retrieval,
            }
        messages.append({"role": "user", "content": results})
        return {
            "messages": messages,
            "steps": steps + 1,
            "pending_action": None,
            "reply": None,
            "retrieval": retrieval,
        }

    async def confirm(state: TurnState) -> dict[str, Any]:
        # NOTE: on resume LangGraph re-runs this node from the top, so everything
        # before interrupt() must be free of side effects.
        action = state["pending_action"]
        args = action["input"]
        user_id = state["user_id"]
        writer = get_stream_writer()

        # Pre-check so we never ask the customer to approve something that cannot happen.
        current = await deps.tools.call("get_order", {"order_id": args.get("order_id", "")}, user_id=user_id)
        precheck_error: str | None = None
        if current.is_error:
            precheck_error = current.content
        elif (status := json.loads(current.content)["status"]) != "pending":
            precheck_error = (
                f"Order {args['order_id']} is {status}; "
                "the address can only be changed while the order is pending."
            )

        elif json.loads(current.content)["shipping_address"] == {
            k: args.get(k) for k in ("street", "city", "state", "postal_code", "country")
        }:
            precheck_error = (
                f"The shipping address of {args['order_id']} is already this address; nothing to change."
            )

        if precheck_error is not None:
            outcome = _tool_result(action["tool_use_id"], precheck_error, True)
        else:
            expires_at = action.get("expires_at")
            decision = interrupt(
                {
                    "type": "confirmation_required",
                    "tool": action["name"],
                    "expires_at": (
                        dt.datetime.fromtimestamp(expires_at, dt.UTC).isoformat(timespec="seconds")
                        if expires_at
                        else None
                    ),
                    "order_id": args["order_id"],
                    "current_address": json.loads(current.content)["shipping_address"],
                    "new_address": {
                        k: args.get(k) for k in ("street", "city", "state", "postal_code", "country")
                    },
                    "message": f"Please confirm the new shipping address for order {args['order_id']}.",
                }
            )
            if decision == "approve" and expires_at and time.time() > expires_at:
                decision = "expired"  # the API checks this too; this is the authoritative check
            if decision == "approve":
                result = await deps.tools.call(
                    action["name"], args, user_id=user_id, idempotency_key=action["idempotency_key"]
                )
                outcome = _tool_result(action["tool_use_id"], result.content, result.is_error)
            elif decision == "expired":
                outcome = _tool_result(action["tool_use_id"], CONFIRMATION_EXPIRED_RESULT, True)
            else:
                outcome = _tool_result(action["tool_use_id"], DECLINED_RESULT, True)
            writer({"type": "action_result", "decision": decision, "is_error": outcome["is_error"]})

        # Return every tool_result for the assistant turn in one user message, in call order.
        order = [b["id"] for b in state["messages"][-1]["content"] if b["type"] == "tool_use"]
        results = sorted([*action["completed_results"], outcome], key=lambda r: order.index(r["tool_use_id"]))
        return {
            "messages": [*state["messages"], {"role": "user", "content": results}],
            "pending_action": None,
        }

    async def search(query: str, user_id: str) -> list[dict[str, Any]]:
        with tracer.span("retrieval", as_type="retriever", input=query) as obs:
            result = await deps.tools.call("search_policy", {"query": query}, user_id=user_id)
            hits = [] if result.is_error else json.loads(result.content)["results"]
            obs.update(output=[{"source": h["source"], "score": h["score"]} for h in hits])
        return hits

    async def agentic_retrieve(
        state: TurnState, conversation: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Decide whether to search; search, grade, and rewrite the query while passages are missing."""
        with tracer.span("agentic_rag", as_type="chain", input=state["user_message"]) as obs:
            decision = await deps.llm.structured(
                "retrieval_decision",
                RETRIEVAL_DECISION_PROMPT,
                _rewrite_input(conversation),
                _DECISION_SCHEMA,
            )
            # A first message routed here is a policy question: only follow-ups may skip the search.
            if decision and not decision["retrieve"] and len(conversation) > 1:
                retrieval = {"query": None, "queries": [], "results": [], "skipped": True, "rounds": []}
                obs.update(output={"skipped": True, "reason": decision["reason"]})
                return [], retrieval

            query = (decision or {}).get("query", "").strip() or state["user_message"]
            question = query if len(conversation) > 1 else state["user_message"]
            merged: dict[str, dict[str, Any]] = {}
            queries: list[str] = []
            rounds: list[dict[str, Any]] = []
            for round_no in range(1 + deps.agentic_max_rewrites):
                queries.append(query)
                found = await search(query, state["user_id"])
                for hit in found:
                    merged.setdefault(hit["chunk_id"], hit)
                rounds.append({"query": query, "found": [h["source"] for h in found]})
                if round_no == deps.agentic_max_rewrites:
                    break  # no rewrites left: grading would not change anything
                grade = await deps.llm.structured(
                    "retrieval_grade",
                    RETRIEVAL_GRADE_PROMPT,
                    _grade_input(question, queries, list(merged.values())),
                    _GRADE_SCHEMA,
                )
                rounds[-1]["grade"] = grade
                if grade is None or grade["sufficient"]:
                    break
                next_query = grade["next_query"].strip()
                if not next_query or next_query in queries:
                    break
                query = next_query
            hits = list(merged.values())[:MAX_AGENTIC_PASSAGES]
            obs.update(output={"queries": queries, "rounds": rounds, "passages": [h["source"] for h in hits]})
        return hits, {"query": queries[0], "queries": queries, "results": hits, "rounds": rounds}

    async def knowledge_agent(state: TurnState) -> dict[str, Any]:
        conversation = _conversation(state)
        context = _turn_context(state)
        if deps.agentic_rag:
            hits, retrieval = await agentic_retrieve(state, conversation)
            if retrieval.get("skipped"):
                get_stream_writer()({"type": "sources", "query": None, "sources": []})
                turn = await deps.llm.respond(
                    KNOWLEDGE_FROM_CONVERSATION_PROMPT,
                    conversation,
                    [],
                    _emit_text,
                    system_suffix=context,
                    name="knowledge_agent",
                )
                return {"reply": turn.text, "retrieval": retrieval}
        else:
            query = state["user_message"]
            # Only follow-ups need rewriting; a first message is already standalone.
            if deps.query_rewrite and len(conversation) > 1:
                query = (
                    await deps.llm.rewrite_query(QUERY_REWRITE_PROMPT, _rewrite_input(conversation)) or query
                )
            hits = await search(query, state["user_id"])
            retrieval = {"query": query, "queries": [query], "results": hits}

        get_stream_writer()(
            {"type": "sources", "query": retrieval["query"], "sources": [h["source"] for h in hits]}
        )
        if not hits:
            _emit_text(NO_POLICY_REPLY)
            return {"reply": NO_POLICY_REPLY, "retrieval": retrieval}

        messages = [
            *conversation[:-1],
            {"role": "user", "content": _sources_message(hits, neutralize(state["user_message"]))},
        ]
        turn = await deps.llm.respond(
            KNOWLEDGE_AGENT_PROMPT, messages, [], _emit_text, system_suffix=context, name="knowledge_agent"
        )
        reply = turn.text
        if turn.stop_reason != "refusal" and not cited_sources(reply, hits):
            # Every policy answer carries its sources, even if the model forgot to cite.
            footer = "\n\nSources: " + "; ".join(f"[{h['source']}]" for h in hits[:2])
            _emit_text(footer)
            reply += footer
        return {"reply": reply, "retrieval": retrieval}

    async def chitchat(state: TurnState) -> dict[str, Any]:
        turn = await deps.llm.respond(
            CHITCHAT_PROMPT,
            _conversation(state),
            [],
            _emit_text,
            system_suffix=_turn_context(state),
            name="chitchat",
        )
        return {"reply": turn.text}

    async def token_limit(state: TurnState) -> dict[str, Any]:
        """The conversation reached its token budget: hand it to a human and end politely."""
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in _conversation(state))
        result = await deps.tools.call(
            "create_ticket",
            {
                "subject": "Conversation handed off: token limit reached",
                "description": (
                    f"SupportPilot reached the conversation token limit.\n\nConversation:\n{transcript}"
                ),
                "priority": "normal",
            },
            user_id=state["user_id"],
        )
        ticket = "" if result.is_error else f" (ticket {json.loads(result.content)['ticket_id']})"
        reply = TOKEN_LIMIT_REPLY.format(ticket=ticket)
        _emit_text(reply)
        return {"reply": reply, "over_budget": True}

    async def handoff(state: TurnState) -> dict[str, Any]:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in _conversation(state))
        result = await deps.tools.call(
            "create_ticket",
            {
                "subject": "Customer asked for a human agent",
                "description": f"Escalated by SupportPilot.\n\nConversation:\n{transcript}",
                "priority": "normal",
            },
            user_id=state["user_id"],
        )
        if result.is_error:
            reply = "I'm connecting you with a human agent. Please hold on while I find someone."
        else:
            ticket_id = json.loads(result.content)["ticket_id"]
            reply = (
                f"I've passed this to a human agent (ticket {ticket_id}). "
                "They'll follow up by email, usually within one business day."
            )
        _emit_text(reply)
        return {"reply": reply}

    async def save_memory(state: TurnState) -> dict[str, Any]:
        await deps.memory.append_turn(
            state["session_id"],
            state["user_message"],
            state.get("reply") or "",
            actions=_action_log(state.get("messages") or []),
            turn_id=state.get("turn_id"),
        )
        return {}

    def route_intent(
        state: TurnState,
    ) -> Literal["order_agent", "knowledge_agent", "chitchat", "handoff", "token_limit"]:
        return {
            "token_limit": "token_limit",
            "knowledge": "knowledge_agent",
            "handoff": "handoff",
            "smalltalk": "chitchat",
            "out_of_scope": "chitchat",
        }.get(state["intent"], "order_agent")

    agent_node = "order_agent" if deps.architecture == "routed" else "agent"

    def after_agent(state: TurnState) -> str:
        if state.get("over_budget"):
            return "token_limit"
        if state.get("pending_action"):
            return "confirm"
        if state.get("reply") is not None:
            return "save_memory"
        return agent_node

    graph = StateGraph(TurnState)
    graph.add_node("confirm", confirm)
    graph.add_node("token_limit", token_limit)
    graph.add_node("save_memory", save_memory)
    graph.add_edge("token_limit", "save_memory")
    graph.add_edge("save_memory", END)
    agent_targets = ["confirm", agent_node, "save_memory", "token_limit"]

    if deps.architecture == "single":
        graph.add_node("screen", screen_only)
        graph.add_node("agent", tool_agent(SINGLE_AGENT_PROMPT, SINGLE_AGENT_TOOLS, "single_agent"))
        graph.add_edge(START, "screen")
        graph.add_conditional_edges(
            "screen",
            lambda s: "token_limit" if s["intent"] == "token_limit" else "agent",
            ["agent", "token_limit"],
        )
        graph.add_conditional_edges("agent", after_agent, agent_targets)
        graph.add_edge("confirm", "agent")
        return graph.compile(checkpointer=checkpointer)

    graph.add_node("router", router)
    graph.add_node("order_agent", tool_agent(ORDER_AGENT_PROMPT, ORDER_AGENT_TOOLS, "order_agent"))
    graph.add_node("knowledge_agent", knowledge_agent)
    graph.add_node("chitchat", chitchat)
    graph.add_node("handoff", handoff)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_intent)
    graph.add_conditional_edges("order_agent", after_agent, agent_targets)
    graph.add_edge("confirm", "order_agent")
    graph.add_edge("knowledge_agent", "save_memory")
    graph.add_edge("chitchat", "save_memory")
    graph.add_edge("handoff", "save_memory")
    return graph.compile(checkpointer=checkpointer)
