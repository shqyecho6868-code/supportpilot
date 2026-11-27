"""Retries, timeouts, model fallback, the conversation token limit and confirmation expiry."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import anthropic
import httpx2
import psycopg
import pytest
from langgraph.types import Command

from supportpilot.llm.anthropic_llm import AnthropicLLM
from supportpilot.llm.usage import CallUsage, UsageMeter, metered
from tests.conftest import ALICE, TEST_DATABASE_URL, chat, confirm, fetch_order

CHANGE_1001 = "Please change the shipping address of ORD-1001 to 500 Market St, San Francisco, CA 94105, US"

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls, status: int):
    return cls(f"HTTP {status}", response=httpx2.Response(status, request=_REQUEST), body=None)


def _message(text: str, model: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
        model=model,
    )


class FakeMessages:
    """Fails according to a script of per-model errors, then answers."""

    def __init__(self, script: dict[str, list[Exception]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def _next_error(self, model: str):
        errors = self.script.get(model, [])
        return errors.pop(0) if errors else None

    async def create(self, *, model: str, **_):
        self.calls.append(model)
        if (error := self._next_error(model)) is not None:
            raise error
        return _message('{"intent": "knowledge"}', model)

    def stream(self, *, model: str, **_):
        self.calls.append(model)
        error = self._next_error(model)

        @asynccontextmanager
        async def ctx():
            if error is not None and not isinstance(error, MidStreamError):
                raise error

            async def text_stream():
                yield f"hello from {model}"
                if isinstance(error, MidStreamError):
                    raise error.inner

            async def final():
                return _message(f"hello from {model}", model)

            yield SimpleNamespace(text_stream=text_stream(), get_final_message=final)

        return ctx()


class MidStreamError(Exception):
    def __init__(self, inner: Exception):
        self.inner = inner


def _llm(script, model="claude-sonnet-5-5", **kw) -> tuple[AnthropicLLM, FakeMessages]:
    messages = FakeMessages(script)
    llm = AnthropicLLM(
        model,
        fast_model="claude-haiku-5-5",
        fallback_model="claude-haiku-5-5",
        retry_base_delay=0.001,
        retry_max_delay=0.002,
        client=SimpleNamespace(messages=messages),
        **kw,
    )
    return llm, messages


async def test_transient_errors_are_retried_with_backoff():
    llm, fake = _llm(
        {
            "claude-sonnet-5-5": [
                _status_error(anthropic.InternalServerError, 529),
                anthropic.APIConnectionError(request=_REQUEST),
            ]
        }
    )
    assert await llm.classify_intent("route", [{"role": "user", "content": "hi"}]) == "knowledge"
    assert fake.calls == ["claude-sonnet-5-5"] * 3
    assert llm.fallbacks_used == 0


async def test_timeouts_count_as_transient():
    llm, fake = _llm({"claude-sonnet-5-5": [anthropic.APITimeoutError(request=_REQUEST)]})
    assert await llm.classify_intent("route", [{"role": "user", "content": "hi"}]) == "knowledge"
    assert fake.calls == ["claude-sonnet-5-5"] * 2


async def test_main_model_falls_back_to_haiku_after_retries():
    errors = [_status_error(anthropic.InternalServerError, 500) for _ in range(4)]
    llm, fake = _llm({"claude-sonnet-5-5": errors}, max_retries=3)
    assert await llm.classify_intent("route", [{"role": "user", "content": "hi"}]) == "knowledge"
    assert fake.calls == ["claude-sonnet-5-5"] * 4 + ["claude-haiku-5-5"]
    assert llm.fallbacks_used == 1


async def test_unknown_model_falls_back_without_retrying():
    """A mistyped main model name (404) is not retried, it falls back to Haiku straight away."""
    llm, fake = _llm(
        {"claude-sonnet-5-5-typo": [_status_error(anthropic.NotFoundError, 404)]},
        model="claude-sonnet-5-5-typo",
    )
    turn = await llm.respond("system", [{"role": "user", "content": "hi"}], [], lambda _t: None)
    assert turn.text == "hello from claude-haiku-5-5"
    assert fake.calls == ["claude-sonnet-5-5-typo", "claude-haiku-5-5"]


async def test_bad_requests_are_not_retried_or_masked():
    llm, fake = _llm({"claude-sonnet-5-5": [_status_error(anthropic.BadRequestError, 400)]})
    with pytest.raises(anthropic.BadRequestError):
        await llm.classify_intent("route", [{"role": "user", "content": "hi"}])
    assert fake.calls == ["claude-sonnet-5-5"]


async def test_stream_does_not_fall_back_after_text_was_sent():
    """Falling back mid-reply would show the customer a second, different answer."""
    error = MidStreamError(_status_error(anthropic.InternalServerError, 500))
    llm, fake = _llm({"claude-sonnet-5-5": [error]}, max_retries=0)
    seen: list[str] = []
    with pytest.raises(anthropic.InternalServerError):
        await llm.respond("system", [{"role": "user", "content": "hi"}], [], seen.append)
    assert fake.calls == ["claude-sonnet-5-5"]
    assert seen == ["hello from claude-sonnet-5-5"]


async def test_fast_model_has_no_fallback_to_itself():
    errors = [_status_error(anthropic.InternalServerError, 500) for _ in range(2)]
    llm, fake = _llm({"claude-haiku-5-5": errors}, max_retries=1)
    with pytest.raises(anthropic.InternalServerError):
        await llm.rewrite_query("rewrite", [{"role": "user", "content": "hi"}])
    assert fake.calls == ["claude-haiku-5-5"] * 2


async def test_usage_is_metered_per_turn_and_priced():
    llm, _ = _llm({})
    with metered() as meter:
        await llm.classify_intent("route", [{"role": "user", "content": "hi"}])
    assert meter.total_tokens == 110
    assert meter.cost == pytest.approx(100 * 2 / 1e6 + 10 * 10 / 1e6)
    haiku = UsageMeter()
    haiku.add(CallUsage("claude-haiku-5-5", input_tokens=1_000_000, cache_read_tokens=1_000_000))
    assert haiku.cost == pytest.approx(0.10 + 0.01)


# --- conversation token limit ---------------------------------------------------------------


async def _tickets(user_id: str) -> list[str]:
    async with await psycopg.AsyncConnection.connect(TEST_DATABASE_URL) as conn:
        cur = await conn.execute("SELECT subject FROM tickets WHERE customer_id = %s", (user_id,))
        return [r[0] for r in await cur.fetchall()]


async def test_token_limit_ends_the_conversation_politely_and_hands_off(make_client):
    # Deterministic: the session's usage is set directly, independent of prompt length.
    client, app = await make_client(conversation_token_limit=100_000)
    first = await chat(client, ALICE, "What is the status of ORD-1002?", session_id="s-budget")
    assert first["route"] == "order"
    usage = next(d for n, d in first["events"] if n == "usage")
    assert 0 < usage["session_tokens"] < 100_000
    await app.state.ctx.memory.add_tokens("s-budget", 100_000)

    second = await chat(client, ALICE, "And ORD-1003?", session_id="s-budget")
    assert second["route"] == "token_limit"
    assert "passed it to a human agent" in second["reply"] and "ticket TCK-" in second["reply"]
    assert await _tickets(ALICE) == ["Conversation handed off: token limit reached"]


async def test_token_limit_reached_mid_turn_hands_off_after_the_current_step(make_client):
    """The order agent checks the budget between steps, so one runaway turn cannot overspend."""
    client, _ = await make_client(conversation_token_limit=1)
    turn = await chat(client, ALICE, "What is the status of ORD-1002?")
    assert turn["route"] == "order"  # the turn starts: nothing was used before it
    assert "passed it to a human agent" in turn["reply"]
    assert await _tickets(ALICE) == ["Conversation handed off: token limit reached"]


async def test_token_limit_is_per_conversation(make_client):
    client, app = await make_client(conversation_token_limit=100_000)
    await chat(client, ALICE, "What is the status of ORD-1002?", session_id="s-a")
    await app.state.ctx.memory.add_tokens("s-a", 100_000)
    assert (await chat(client, ALICE, "What is the status of ORD-1002?", session_id="s-a"))[
        "route"
    ] == "token_limit"
    fresh = await chat(client, ALICE, "What is the status of ORD-1002?", session_id="s-b")
    assert fresh["route"] == "order"


# --- confirmation expiry (P1 leftover) ------------------------------------------------------


async def test_expired_confirmation_returns_410_and_changes_nothing(make_client):
    client, _ = await make_client(confirmation_ttl_seconds=0)
    before = await fetch_order("ORD-1001")
    turn = await chat(client, ALICE, CHANGE_1001, session_id="s-exp")
    assert turn["confirmation"]["expires_at"]

    resp = await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")
    assert resp.status_code == 410
    detail = resp.json()["detail"]
    assert detail["error"] == "confirmation_expired"
    assert "Nothing was changed" in detail["message"]
    assert await fetch_order("ORD-1001") == before

    # The thread is closed: a second attempt finds nothing pending.
    again = await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")
    assert again.status_code == 409


async def test_expired_turn_is_recorded_in_memory(make_client):
    from redis.asyncio import Redis

    from tests.conftest import TEST_REDIS_URL

    client, _ = await make_client(confirmation_ttl_seconds=0)
    turn = await chat(client, ALICE, CHANGE_1001, session_id="s-exp-mem")
    await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")
    redis = Redis.from_url(TEST_REDIS_URL)
    history = [json.loads(x) for x in await redis.lrange("sp:session:s-exp-mem:history", 0, -1)]
    await redis.aclose()
    assert len(history) == 2
    assert "FAILED: The confirmation request expired" in history[1]["actions"][-1]


async def test_graph_rejects_an_approval_that_arrives_after_expiry(make_client):
    """The confirm node re-checks the expiry itself, so a stale approve cannot slip past the API check."""
    client, app = await make_client(confirmation_ttl_seconds=0)
    turn = await chat(client, ALICE, CHANGE_1001)
    config = {"configurable": {"thread_id": turn["confirmation"]["thread_id"]}}
    await app.state.ctx.graph.ainvoke(Command(resume="approve"), config)
    assert (await fetch_order("ORD-1001"))["address_events"] == 0


async def test_confirmation_within_ttl_still_works(make_client):
    client, _ = await make_client(confirmation_ttl_seconds=900)
    turn = await chat(client, ALICE, CHANGE_1001)
    resp = await confirm(client, ALICE, turn["confirmation"]["thread_id"], "approve")
    assert resp.status_code == 200
    assert (await fetch_order("ORD-1001"))["address_events"] == 1
