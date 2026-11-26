from typing import Any, TypedDict


class TurnState(TypedDict, total=False):
    """State for one customer turn. One LangGraph thread == one turn."""

    user_id: str
    session_id: str
    user_message: str
    history: list[dict[str, Any]]  # prior turns from Redis, plain text
    intent: str
    messages: list[dict[str, Any]]  # order agent transcript, Messages API param form
    pending_action: dict[str, Any] | None  # tool call awaiting human confirmation
    steps: int
    reply: str | None
    retrieval: dict[str, Any] | None  # knowledge agent: the search query and passages it used
