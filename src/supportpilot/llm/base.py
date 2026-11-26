from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

Intent = Literal["order", "knowledge", "handoff", "smalltalk", "out_of_scope"]
INTENTS: tuple[Intent, ...] = ("order", "knowledge", "handoff", "smalltalk", "out_of_scope")

TextCallback = Callable[[str], None]


@dataclass
class LLMTurn:
    """One assistant response, with content blocks in Messages API *param* form."""

    content: list[dict[str, Any]]
    stop_reason: str | None

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b["type"] == "text")

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b["type"] == "tool_use"]


class LLMClient(Protocol):
    async def classify_intent(self, system: str, messages: list[dict[str, Any]]) -> Intent: ...

    async def respond(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: TextCallback,
    ) -> LLMTurn: ...

    async def rewrite_query(self, system: str, messages: list[dict[str, Any]]) -> str:
        """Turn the latest message (plus conversation context) into a standalone search query."""
        ...
