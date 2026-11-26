"""LangGraph orchestration.

    START -> router -+-> order_agent <-> confirm   (confirm pauses with interrupt())
                     +-> knowledge_agent
                     +-> handoff
             all three end in save_memory -> END

The confirm node is the human-in-the-loop gate: it pauses the graph before any
tool in CONFIRMATION_REQUIRED runs. Because the graph is checkpointed to
Postgres, a paused turn survives a process restart and resumes from /confirm.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from supportpilot.graph.prompts import (
    ACTION_LOG_PREFIX,
    KNOWLEDGE_PLACEHOLDER_REPLY,
    ORDER_AGENT_PROMPT,
    ROUTER_PROMPT,
)
from supportpilot.graph.state import TurnState
from supportpilot.llm import LLMClient
from supportpilot.memory import ConversationMemory
from supportpilot.tools import CONFIRMATION_REQUIRED, ToolGateway

STEP_LIMIT_REPLY = "Sorry, I wasn't able to finish that. I can open a ticket for a human agent if you'd like."
DECLINED_RESULT = (
    "The customer declined this change at the confirmation step. Nothing was changed. "
    "Do not retry unless the customer asks again."
)


@dataclass
class GraphDeps:
    llm: LLMClient
    tools: ToolGateway
    memory: ConversationMemory
    max_agent_steps: int = 8


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}


def _render(entry: dict[str, Any]) -> dict[str, Any]:
    """Turn a memory entry into a message, prefixing the system's record of tool activity."""
    content = entry["content"]
    if entry.get("actions"):
        content = f"{ACTION_LOG_PREFIX} {'; '.join(entry['actions'])}]\n{content}"
    return {"role": entry["role"], "content": content}


def _conversation(state: TurnState) -> list[dict[str, Any]]:
    history = list(state.get("history") or [])
    while history and history[0]["role"] != "user":  # API requires a user turn first
        history.pop(0)
    return [*(_render(e) for e in history), {"role": "user", "content": state["user_message"]}]


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


def _emit_text(text: str) -> None:
    get_stream_writer()({"type": "token", "text": text})


def build_graph(deps: GraphDeps, checkpointer: BaseCheckpointSaver):
    async def router(state: TurnState) -> dict[str, Any]:
        intent = await deps.llm.classify_intent(ROUTER_PROMPT, _conversation(state))
        get_stream_writer()({"type": "route", "intent": intent})
        return {"intent": intent}

    async def order_agent(state: TurnState) -> dict[str, Any]:
        messages = state.get("messages") or _conversation(state)
        steps = state.get("steps", 0)
        if steps >= deps.max_agent_steps:
            _emit_text(STEP_LIMIT_REPLY)
            return {"reply": STEP_LIMIT_REPLY}

        turn = await deps.llm.respond(
            ORDER_AGENT_PROMPT, messages, await deps.tools.model_tools(), _emit_text
        )
        messages = [*messages, {"role": "assistant", "content": turn.content}]
        if not turn.tool_uses:
            return {"messages": messages, "reply": turn.text, "steps": steps + 1, "pending_action": None}

        # Run read-only tools now; hold back at most one mutating tool for confirmation.
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
                    }
                else:
                    results.append(
                        _tool_result(call["id"], "Only one change can be confirmed at a time.", True)
                    )
                continue
            result = await deps.tools.call(call["name"], call["input"], user_id=state["user_id"])
            results.append(_tool_result(call["id"], result.content, result.is_error))

        if pending is not None:
            pending["completed_results"] = results
            return {"messages": messages, "pending_action": pending, "steps": steps + 1, "reply": None}
        messages.append({"role": "user", "content": results})
        return {"messages": messages, "steps": steps + 1, "pending_action": None, "reply": None}

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
            decision = interrupt(
                {
                    "type": "confirmation_required",
                    "tool": action["name"],
                    "order_id": args["order_id"],
                    "current_address": json.loads(current.content)["shipping_address"],
                    "new_address": {
                        k: args.get(k) for k in ("street", "city", "state", "postal_code", "country")
                    },
                    "message": f"Please confirm the new shipping address for order {args['order_id']}.",
                }
            )
            if decision == "approve":
                result = await deps.tools.call(
                    action["name"], args, user_id=user_id, idempotency_key=action["idempotency_key"]
                )
                outcome = _tool_result(action["tool_use_id"], result.content, result.is_error)
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

    async def knowledge_agent(state: TurnState) -> dict[str, Any]:
        _emit_text(KNOWLEDGE_PLACEHOLDER_REPLY)
        return {"reply": KNOWLEDGE_PLACEHOLDER_REPLY}

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
        )
        return {}

    def route_intent(state: TurnState) -> Literal["order_agent", "knowledge_agent", "handoff"]:
        return {"knowledge": "knowledge_agent", "handoff": "handoff"}.get(state["intent"], "order_agent")

    def after_order_agent(state: TurnState) -> Literal["confirm", "order_agent", "save_memory"]:
        if state.get("pending_action"):
            return "confirm"
        if state.get("reply") is not None:
            return "save_memory"
        return "order_agent"

    graph = StateGraph(TurnState)
    graph.add_node("router", router)
    graph.add_node("order_agent", order_agent)
    graph.add_node("confirm", confirm)
    graph.add_node("knowledge_agent", knowledge_agent)
    graph.add_node("handoff", handoff)
    graph.add_node("save_memory", save_memory)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_intent)
    graph.add_conditional_edges("order_agent", after_order_agent)
    graph.add_edge("confirm", "order_agent")
    graph.add_edge("knowledge_agent", "save_memory")
    graph.add_edge("handoff", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile(checkpointer=checkpointer)
