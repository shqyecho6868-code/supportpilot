from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class ConfirmRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    decision: Literal["approve", "reject"]


class ConfirmResponse(BaseModel):
    status: Literal["completed", "confirmation_required"]
    thread_id: str
    decision: Literal["approve", "reject"]
    reply: str | None = None
    confirmation: dict | None = None
