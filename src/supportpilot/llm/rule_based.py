"""Deterministic, offline stand-in for Claude.

This is a test double, not a model: it maps a few fixed phrasings to tool calls
so the graph, MCP tools, confirmation flow and restart recovery can be tested
end to end without network access or API cost. Production uses ``AnthropicLLM``.
"""

import html
import json
import re
import uuid
from typing import Any

from supportpilot.llm.base import Intent, LLMTurn, TextCallback
from supportpilot.llm.usage import CallUsage, record

MODEL_NAME = "rule-based"

_ORDER_ID = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
_ADDRESS = re.compile(
    r"\bto\s+(?P<street>[^,]+),\s*(?P<city>[^,]+),\s*(?P<state>[A-Za-z]{2})\s+(?P<postal_code>[\w-]+)"
    r"(?:,\s*(?P<country>[A-Za-z]{2}))?",
    re.IGNORECASE,
)
_HANDOFF = re.compile(r"\b(human|real person|representative|live agent|speak to (an )?agent)\b", re.I)
_KNOWLEDGE = re.compile(
    r"\b(policy|policies|return window|refund policy|warranty|how long does shipping)\b", re.I
)
_SMALLTALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|bye)\b", re.I)
_OUT_OF_SCOPE = re.compile(r"\b(weather|poem|joke|recipe|stock price)\b", re.I)
_SOURCE = re.compile(r'<source cite="(?P<cite>[^"]+)">\n(?P<text>.*?)\n</source>', re.S)
_LATEST = re.compile(r"<latest_message>\n(?P<text>.*?)\n</latest_message>", re.S)


def _text_of(message: dict[str, Any]) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if b.get("type") == "text")


def _record(messages: list[dict[str, Any]], system: str, output: str) -> None:
    """Pretend token usage (~4 characters per token) so token limits can be tested offline."""
    size = len(system) + len(json.dumps(messages, default=str))
    record(CallUsage(MODEL_NAME, input_tokens=size // 4, output_tokens=len(output) // 4))


def _emit(text: str, on_text: TextCallback) -> LLMTurn:
    for word in text.split(" "):
        on_text(word + " ")
    return LLMTurn(content=[{"type": "text", "text": text}], stop_reason="end_turn")


def _tool_call(name: str, args: dict[str, Any]) -> LLMTurn:
    block = {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:12]}", "name": name, "input": args}
    return LLMTurn(content=[block], stop_reason="tool_use")


class RuleBasedLLM:
    async def classify_intent(self, system: str, messages: list[dict[str, Any]]) -> Intent:
        _record(messages, system, "intent")
        text = _text_of(messages[-1])
        if _HANDOFF.search(text):
            return "handoff"
        if _KNOWLEDGE.search(text):
            return "knowledge"
        if _OUT_OF_SCOPE.search(text):
            return "out_of_scope"
        if _SMALLTALK.search(text):
            return "smalltalk"
        return "order"

    async def rewrite_query(self, system: str, messages: list[dict[str, Any]]) -> str:
        match = _LATEST.search(_text_of(messages[-1]))
        return match["text"].strip() if match else ""

    async def structured(
        self,
        name: str,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | None:
        """Agentic-RAG decisions: always retrieve with the latest message; passages are enough
        unless none were found, then retry once with "policy" appended."""
        _record(messages, system, name)
        text = _text_of(messages[-1])
        if name == "retrieval_decision":
            latest = _LATEST.search(text)
            return {"retrieve": True, "query": latest["text"].strip() if latest else text, "reason": "rule"}
        if name == "retrieval_grade":
            sufficient = "<source " in text
            return {
                "sufficient": sufficient,
                "missing": "" if sufficient else "everything",
                "next_query": "policy",
            }
        return None

    async def respond(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: TextCallback,
        *,
        system_suffix: str | None = None,
        name: str = "respond",
    ) -> LLMTurn:
        turn = await self._respond(messages, tools, on_text)
        _record(messages, system + (system_suffix or ""), turn.text)
        return turn

    async def _respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], on_text: TextCallback
    ) -> LLMTurn:
        last = messages[-1]
        if isinstance(last["content"], list) and any(b.get("type") == "tool_result" for b in last["content"]):
            return _emit(self._summarize(messages), on_text)

        text = _text_of(last)
        if not tools:
            return _emit(self._answer_without_tools(text), on_text)
        order_match = _ORDER_ID.search(text)
        address_match = _ADDRESS.search(text)
        if order_match and address_match and "address" in text.lower():
            parts = address_match.groupdict()
            return _tool_call(
                "update_address",
                {
                    "order_id": order_match.group(0).upper(),
                    "street": parts["street"].strip(),
                    "city": parts["city"].strip(),
                    "state": parts["state"].upper(),
                    "postal_code": parts["postal_code"],
                    "country": (parts["country"] or "US").upper(),
                },
            )
        if order_match:
            return _tool_call("get_order", {"order_id": order_match.group(0).upper()})
        if "orders" in text.lower():
            return _tool_call("list_my_orders", {})
        return _emit("I can help you track orders or update a pending order's shipping address.", on_text)

    @staticmethod
    def _answer_without_tools(text: str) -> str:
        source = _SOURCE.search(text)
        if source is None:  # small talk / out of scope
            return "I can help with your orders and with questions about our store policies."
        body = source["text"].split("\n\n", 1)[-1]
        first_sentence = body.split(". ")[0].rstrip(".")
        return f"{first_sentence}. [{html.unescape(source['cite'])}]"

    @staticmethod
    def _summarize(messages: list[dict[str, Any]]) -> str:
        names = {b["id"]: b["name"] for b in messages[-2]["content"] if b.get("type") == "tool_use"}
        lines = []
        for block in messages[-1]["content"]:
            if block.get("type") != "tool_result":
                continue
            raw = block["content"] if isinstance(block["content"], str) else block["content"][0]["text"]
            name = names.get(block["tool_use_id"], "")
            if block.get("is_error"):
                lines.append(f"Sorry, I couldn't complete that: {raw}")
                continue
            data = json.loads(raw)
            if name == "update_address":
                a = data["new_address"]
                lines.append(
                    f"Done. The shipping address for {data['order_id']} is now "
                    f"{a['street']}, {a['city']}, {a['state']} {a['postal_code']}, {a['country']}."
                )
            elif name == "get_order":
                lines.append(f"Order {data['order_id']} is {data['status']}.")
            elif name == "list_my_orders":
                summary = ", ".join(f"{o['order_id']} ({o['status']})" for o in data["orders"])
                lines.append(f"Your orders: {summary}.")
            else:
                lines.append("Done.")
        return " ".join(lines)
