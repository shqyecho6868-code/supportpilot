"""Server-side refusal fallback: Sonnet 5.5 requests opt in with `fallbacks: "default"`.

A policy decline is an HTTP 200 with stop_reason "refusal". With the opt-in, the API re-runs the
request on its fallback model inside the same call; these tests use a fake client that returns
the response shapes the API documents for that (usage.iterations, fallback content blocks).
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from supportpilot.llm.anthropic_llm import REFUSAL_REPLY, SERVER_FALLBACK_BETA, AnthropicLLM
from supportpilot.llm.usage import metered


def _usage(input_tokens=100, output_tokens=10, iterations=None):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        iterations=iterations,
    )


def _iteration(kind: str, model: str, input_tokens: int, output_tokens: int):
    return SimpleNamespace(type=kind, model=model, **vars(_usage(input_tokens, output_tokens)))


class _Block(SimpleNamespace):
    def model_dump(self, *, mode="json", by_alias=False, exclude_none=False) -> dict[str, Any]:
        data = dict(vars(self))
        if by_alias and "from_" in data:
            data["from"] = data.pop("from_")
        return data


def _text(text):
    return _Block(type="text", text=text)


def _tool_use(id_, name="get_order"):
    return _Block(type="tool_use", id=id_, name=name, input={"order_id": "ORD-1001"})


def _fallback(src="claude-sonnet-5-5", dst="claude-sonnet-5"):
    return _Block(type="fallback", from_={"model": src}, to={"model": dst}, trigger={"type": "refusal"})


# The declined Sonnet 5.5 hop and the Sonnet 5 hop that served the reply, billed separately.
SERVED_BY_FALLBACK = dict(
    content=[_text("Here is how the warranty works.")],
    stop_reason="end_turn",
    model="claude-sonnet-5",
    usage=_usage(
        1300,
        60,
        iterations=[
            _iteration("message", "claude-sonnet-5-5", 1000, 5),
            _iteration("fallback_message", "claude-sonnet-5", 300, 55),
        ],
    ),
)


class FakeMessages:
    """Records every request's params and answers each with the next scripted response."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def _next(self, params):
        self.requests.append(params)
        return SimpleNamespace(**self.responses.pop(0))

    async def create(self, **params):
        return self._next(params)

    def stream(self, **params):
        message = self._next(params)

        @asynccontextmanager
        async def ctx():
            async def text_stream():
                for block in message.content:
                    if block.type == "text":
                        yield block.text

            async def final():
                return message

            yield SimpleNamespace(text_stream=text_stream(), get_final_message=final)

        return ctx()


def _llm(responses, model="claude-sonnet-5-5"):
    plain, beta = FakeMessages(responses), FakeMessages(responses)
    llm = AnthropicLLM(
        model,
        fast_model="claude-haiku-5-5",
        client=SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta)),
    )
    return llm, plain, beta


def _ok(text="ok"):
    return dict(content=[_text(text)], stop_reason="end_turn", model="claude-sonnet-5-5", usage=_usage())


async def test_sonnet_requests_opt_into_server_side_fallback():
    llm, plain, beta = _llm([_ok('{"intent": "order"}')])
    assert await llm.classify_intent("route", [{"role": "user", "content": "hi"}]) == "order"
    assert plain.requests == []
    (request,) = beta.requests
    assert request["betas"] == [SERVER_FALLBACK_BETA]
    assert request["fallbacks"] == "default"


async def test_haiku_has_no_server_side_fallback_and_uses_the_plain_endpoint():
    llm, plain, beta = _llm([_ok('{"intent": "order"}')], model="claude-haiku-5-5")
    await llm.classify_intent("route", [{"role": "user", "content": "hi"}])
    assert beta.requests == []
    (request,) = plain.requests
    assert "fallbacks" not in request and "betas" not in request


async def test_refusal_is_answered_by_the_fallback_model_and_billed_per_model():
    llm, _, beta = _llm([SERVED_BY_FALLBACK])
    emitted: list[str] = []
    with metered() as meter:
        turn = await llm.respond("system", [{"role": "user", "content": "warranty?"}], [], emitted.append)
    assert beta.requests[0]["fallbacks"] == "default"
    assert turn.stop_reason == "end_turn"
    assert turn.text == "Here is how the warranty works."
    assert "".join(emitted) == turn.text  # the customer gets the fallback model's answer, not a refusal
    assert llm.server_fallbacks_used == 1
    assert llm.fallbacks_used == 0  # the client-side (error) fallback did not run
    by_model = meter.by_model()
    assert by_model["claude-sonnet-5-5"]["tokens"] == 1005
    assert by_model["claude-sonnet-5"]["tokens"] == 355


async def test_refusal_of_the_whole_chain_still_ends_politely():
    """Sonnet 5.5 falls back only for some categories (cyber, frontier_llm); others stay declined."""
    refused = dict(
        content=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(type="refusal", category="general_harms", explanation=None),
        model="claude-sonnet-5-5",
        usage=_usage(),
    )
    llm, _, _ = _llm([refused])
    emitted: list[str] = []
    turn = await llm.respond("system", [{"role": "user", "content": "..."}], [], emitted.append)
    assert turn.stop_reason == "refusal"
    assert turn.text == REFUSAL_REPLY
    assert emitted == [REFUSAL_REPLY]
    assert llm.server_fallbacks_used == 0


async def test_mid_output_fallback_drops_the_declined_tool_call():
    """Before the last fallback block, the declined model's tool_use is neither run nor sent back."""
    mid_stream = dict(
        content=[
            _text("Let me check. "),
            _tool_use("toolu_declined"),
            _fallback(),
            _text("Checking your order."),
            _tool_use("toolu_served"),
        ],
        stop_reason="tool_use",
        model="claude-sonnet-5",
        usage=_usage(
            iterations=[
                _iteration("message", "claude-sonnet-5-5", 80, 8),
                _iteration("fallback_message", "claude-sonnet-5", 20, 2),
            ]
        ),
    )
    llm, _, _ = _llm([mid_stream])
    turn = await llm.respond("system", [{"role": "user", "content": "where is it?"}], [], lambda _t: None)
    assert [b["type"] for b in turn.content] == ["text", "fallback", "text", "tool_use"]
    assert [t["id"] for t in turn.tool_uses] == ["toolu_served"]
    assert turn.content[1]["from"] == {"model": "claude-sonnet-5-5"}  # serialized by alias, in place


async def test_fallback_blocks_are_stripped_for_a_model_without_server_fallback():
    """After a Sonnet turn fell back, a Haiku call (e.g. the client-side fallback) must not see the marker."""
    llm, plain, _ = _llm([_ok()], model="claude-haiku-5-5")
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [{"type": "fallback", "from": {}, "to": {}}, {"type": "text", "text": "x"}],
        },
        {"role": "user", "content": "and?"},
    ]
    await llm.respond("system", history, [], lambda _t: None)
    sent = plain.requests[0]["messages"]
    assert sent[1]["content"] == [{"type": "text", "text": "x"}]
    assert history[1]["content"][0]["type"] == "fallback"  # the caller's list is not mutated
