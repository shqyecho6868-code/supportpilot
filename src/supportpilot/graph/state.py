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
    turn_id: str  # the LangGraph thread id; keys this turn's entry in short-term memory
    tokens_used: int  # tokens the conversation had used before this turn
    over_budget: bool  # the conversation token limit was reached during this turn
    injection: dict[str, Any] | None  # injection-guard verdict on the customer's message
    long_term: list[str]  # recalled long-term memories about this customer
