"""Short-term conversation memory in Redis.

Each session keeps its last N messages with a sliding TTL. Full tool payloads and
reasoning stay in the LangGraph checkpoint for the turn; later turns get the text
the customer saw plus a one-line log per tool call that ran (name, arguments,
outcome), so the model knows which actions really happened.
"""

import json
from typing import Any

from redis.asyncio import Redis


class SessionOwnershipError(Exception):
    pass


class ConversationMemory:
    def __init__(self, redis: Redis, *, max_messages: int, ttl_seconds: int):
        self._redis = redis
        self._max = max_messages
        self._ttl = ttl_seconds

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"sp:session:{session_id}:history"

    @staticmethod
    def _owner_key(session_id: str) -> str:
        return f"sp:session:{session_id}:owner"

    async def claim(self, session_id: str, user_id: str) -> None:
        """Bind a session to its first user; reject any other user afterwards."""
        key = self._owner_key(session_id)
        await self._redis.set(key, user_id, nx=True, ex=self._ttl)
        owner = await self._redis.get(key)
        if owner is not None and owner.decode() != user_id:
            raise SessionOwnershipError(session_id)
        await self._redis.expire(key, self._ttl)

    async def history(self, session_id: str) -> list[dict[str, Any]]:
        raw = await self._redis.lrange(self._history_key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    async def append_turn(
        self, session_id: str, user_text: str, assistant_text: str, *, actions: list[str] | None = None
    ) -> None:
        """Store one turn. ``actions`` is the system's log of tool calls that ran during the turn."""
        key = self._history_key(session_id)
        assistant: dict[str, Any] = {"role": "assistant", "content": assistant_text}
        if actions:
            assistant["actions"] = actions
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(
                key,
                json.dumps({"role": "user", "content": user_text}),
                json.dumps(assistant),
            )
            pipe.ltrim(key, -self._max, -1)
            pipe.expire(key, self._ttl)
            pipe.expire(self._owner_key(session_id), self._ttl)
            await pipe.execute()
