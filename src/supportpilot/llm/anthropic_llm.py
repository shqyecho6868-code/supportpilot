"""Claude via the official Anthropic SDK."""

import json
import logging
from typing import Any

import anthropic

from supportpilot.llm.base import INTENTS, Intent, LLMTurn, TextCallback

log = logging.getLogger(__name__)

REFUSAL_REPLY = (
    "I'm sorry, I can't help with that request. Is there anything else about your orders I can do?"
)

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": list(INTENTS)}},
    "required": ["intent"],
    "additionalProperties": False,
}


def _to_param(block: Any) -> dict[str, Any]:
    """Convert a response content block to the param shape the API accepts back."""
    match block.type:
        case "text":
            return {"type": "text", "text": block.text}
        case "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        case "thinking":
            # Passed back unchanged so the model keeps its reasoning across tool calls.
            return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
        case "redacted_thinking":
            return {"type": "redacted_thinking", "data": block.data}
        case _:
            return block.model_dump(mode="json", exclude_none=True)


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None):
        # With api_key=None the SDK resolves credentials from the environment.
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def classify_intent(self, system: str, messages: list[dict[str, Any]]) -> Intent:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _INTENT_SCHEMA}},
        )
        if response.stop_reason == "refusal":
            return "handoff"
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            intent = json.loads(text)["intent"]
        except (json.JSONDecodeError, KeyError, TypeError):
            log.warning("router returned unparseable output: %r", text)
            return "order"
        return intent if intent in INTENTS else "order"

    async def respond(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: TextCallback,
    ) -> LLMTurn:
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=16000,
            # Cache the stable prefix (tools + system) across turns.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
            output_config={"effort": "low"},
        ) as stream:
            async for text in stream.text_stream:
                on_text(text)
            final = await stream.get_final_message()

        if final.stop_reason == "refusal":
            on_text(REFUSAL_REPLY)
            return LLMTurn(content=[{"type": "text", "text": REFUSAL_REPLY}], stop_reason="refusal")
        return LLMTurn(content=[_to_param(b) for b in final.content], stop_reason=final.stop_reason)
