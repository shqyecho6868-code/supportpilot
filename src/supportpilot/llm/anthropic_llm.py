"""Claude via the official Anthropic SDK.

Reliability
-----------
* Every request has a timeout (``llm_timeout_seconds``).
* Transient failures -- rate limits, 5xx / overloaded, timeouts, connection errors -- are
  retried with exponential backoff and jitter (``llm_max_retries``). The SDK's own retries are
  turned off so there is exactly one retry policy, and it is visible in logs and traces.
* If the primary model still fails, or does not exist (404, e.g. a mistyped model name), the
  call is repeated once on ``fallback_model`` (Haiku). A streamed reply falls back only if no
  text has reached the customer yet, so a reply is never emitted twice.
* Errors that a retry cannot fix (400, 401, 403) are raised immediately.

Refusals
--------
A policy decline is not an error: it is an HTTP 200 with ``stop_reason == "refusal"``. On models
that support it (Sonnet 5.5, Opus 5.5, Fable) every request opts into the API's server-side
fallback (``fallbacks: "default"``): the API re-runs a declined request on the fallback model it
picks for the refusal category, inside the same call. Sonnet 5.5 falls back to Sonnet 5 for
``cyber`` and ``frontier_llm`` declines; other categories stay declined, and a refusal of the
whole chain still ends in ``REFUSAL_REPLY``. Haiku has no server-side fallback, so dev and CI
(Haiku) use the plain endpoint.
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import anthropic

from supportpilot.llm.base import INTENTS, Intent, LLMTurn, TextCallback
from supportpilot.llm.usage import CallUsage, record
from supportpilot.observability import get_tracer, preview

log = logging.getLogger(__name__)

T = TypeVar("T")

REFUSAL_REPLY = (
    "I'm sorry, I can't help with that request. Is there anything else about your orders I can do?"
)

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": list(INTENTS)}},
    "required": ["intent"],
    "additionalProperties": False,
}

SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_SERVER_FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5")

_TRANSIENT = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,  # 5xx, including 529 overloaded
    anthropic.APIConnectionError,  # includes APITimeoutError
)


def thinking_off(model: str) -> dict[str, Any] | None:
    """The `thinking` value that turns thinking off for this model, if it can be turned off.

    Sonnet 5.5 rejects {"type": "disabled"} and uses "between_tools" instead; Opus 5.5 and Fable
    cannot turn thinking off at all (None: leave the parameter out and rely on low effort).
    """
    if model.startswith(("claude-opus-5-5", "claude-fable")):
        return None
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}
    return {"type": "disabled"}


def supports_server_fallback(model: str) -> bool:
    return model.startswith(_SERVER_FALLBACK_MODELS)


def served_by_fallback(response: Any) -> str | None:
    """The model that served the reply if a server-side fallback ran, else None."""
    iterations = getattr(getattr(response, "usage", None), "iterations", None) or []
    if any(getattr(it, "type", None) == "fallback_message" for it in iterations):
        return getattr(response, "model", None)
    return None


def _assistant_content(blocks: list[Any]) -> list[dict[str, Any]]:
    """Response blocks in the form they are sent back (and acted on) in the next request.

    After a mid-output server-side fallback, the declined model's thinking and tool_use blocks
    before the last ``fallback`` block are left out (its tool calls are never run), and the
    ``fallback`` block stays where it was: the API uses its position to validate the turn.
    """
    params = [_to_param(b) for b in blocks]
    boundary = max((i for i, b in enumerate(params) if b["type"] == "fallback"), default=-1)
    dropped = {"thinking", "redacted_thinking", "tool_use"}
    return [b for i, b in enumerate(params) if i >= boundary or b["type"] not in dropped]


def _without_fallback_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip ``fallback`` markers for a model called without the fallback beta (it rejects them)."""

    def is_marker(block: Any) -> bool:
        return isinstance(block, dict) and block.get("type") == "fallback"

    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list) and any(is_marker(b) for b in content):
            m = {**m, "content": [b for b in content if not is_marker(b)]}
        out.append(m)
    return out


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT):
        return True
    return isinstance(exc, anthropic.APIStatusError) and (
        exc.status_code in (408, 409) or exc.status_code >= 500
    )


def _should_fall_back(exc: BaseException) -> bool:
    """After retries: fall back for transient errors and for a model that does not exist."""
    return _is_transient(exc) or isinstance(exc, anthropic.NotFoundError)


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
        case _:  # e.g. "fallback": by_alias keeps its "from" key
            return block.model_dump(mode="json", by_alias=True, exclude_none=True)


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        fast_model: str | None = None,
        *,
        fallback_model: str | None = "claude-haiku-5-5",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        server_fallback: bool = True,
        client: Any | None = None,
    ):
        # With api_key=None the SDK resolves credentials from the environment.
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self._model = model
        self._fast_model = fast_model or model
        self._fallback = fallback_model if fallback_model and fallback_model != model else None
        self._max_retries = max_retries
        self._base_delay = retry_base_delay
        self._max_delay = retry_max_delay
        self._server_fallback = server_fallback
        self.fallbacks_used = 0  # observable in tests and logs
        self.server_fallbacks_used = 0  # replies served by the API's server-side fallback model

    # -- retry / fallback -------------------------------------------------------------------------

    async def _with_retries(self, name: str, model: str, fn: Callable[[str], Awaitable[T]]) -> T:
        attempt = 0
        while True:
            try:
                return await fn(model)
            except Exception as exc:
                if not _is_transient(exc) or attempt >= self._max_retries:
                    raise
                delay = min(self._base_delay * 2**attempt, self._max_delay) * (0.5 + random.random() / 2)
                attempt += 1
                log.warning(
                    "%s on %s failed (%s); retry %d/%d in %.2fs",
                    name,
                    model,
                    exc.__class__.__name__,
                    attempt,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _call(
        self,
        name: str,
        model: str,
        fn: Callable[[str], Awaitable[T]],
        *,
        can_fall_back: Callable[[], bool] = lambda: True,
    ) -> T:
        try:
            return await self._with_retries(name, model, fn)
        except Exception as exc:
            fallback = self._fallback if model == self._model else None
            if fallback is None or not _should_fall_back(exc) or not can_fall_back():
                raise
            self.fallbacks_used += 1
            log.error(
                "%s: %s failed (%s: %s); falling back to %s", name, model, type(exc).__name__, exc, fallback
            )
            with get_tracer().span(
                "model_fallback",
                as_type="span",
                metadata={"from": model, "to": fallback, "error": f"{type(exc).__name__}: {exc}"[:500]},
            ):
                return await self._with_retries(name, fallback, fn)

    def _messages_api(self, model: str, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """The messages resource to call and the final params: opts into server-side fallback when
        the model supports it, else the plain endpoint (which rejects ``fallback`` blocks)."""
        if self._server_fallback and supports_server_fallback(model):
            opt_in = {"betas": [SERVER_FALLBACK_BETA], "fallbacks": "default"}
            return self._client.beta.messages, {**params, **opt_in}
        if "messages" in params:
            params = {**params, "messages": _without_fallback_blocks(params["messages"])}
        return self._client.messages, params

    def _meter(self, name: str, model: str, response: Any) -> dict[str, Any]:
        """Record usage per model that ran; return the Langfuse fields for the generation."""
        usages = CallUsage.per_model(model, response.usage)
        for usage in usages:
            record(usage)
        details: dict[str, int] = {}
        cost: dict[str, float] = {}
        for usage in usages:
            for k, v in usage.usage_details().items():
                details[k] = details.get(k, 0) + v
            for k, v in usage.cost().items():
                cost[k] = cost.get(k, 0.0) + v
        metadata: dict[str, Any] = {"stop_reason": response.stop_reason}
        if served := served_by_fallback(response):
            self.server_fallbacks_used += 1
            metadata["server_fallback"] = served
            log.warning("%s: %s declined; served by server-side fallback %s", name, model, served)
        return {"usage_details": details, "cost_details": cost, "metadata": metadata}

    async def _create(self, name: str, model: str, **params: Any) -> Any:
        """One non-streaming request, traced and metered."""
        with get_tracer().generation(
            name, model=model, input=preview(params.get("messages", [])), model_parameters=_params_of(params)
        ) as gen:
            started = time.monotonic()
            api, params = self._messages_api(model, params)
            response = await api.create(model=model, **params)
            fields = self._meter(name, model, response)
            fields["metadata"]["seconds"] = round(time.monotonic() - started, 3)
            gen.update(output="".join(getattr(b, "text", "") for b in response.content), **fields)
            return response

    # -- public API -------------------------------------------------------------------------------

    async def rewrite_query(self, system: str, messages: list[dict[str, Any]]) -> str:
        async def run(model: str):
            return await self._create(
                "rewrite_query",
                model,
                max_tokens=256,
                system=system,
                messages=messages,
                # A short, deterministic-as-possible rewrite: no thinking, low effort.
                **({"thinking": t} if (t := thinking_off(model)) else {}),
                output_config={"effort": "low"},
            )

        response = await self._call("rewrite_query", self._fast_model, run)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal" or not text:
            return ""
        return text.splitlines()[0].strip().strip('"')

    async def structured(
        self,
        name: str,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | None:
        """A small JSON decision from the fast model. None if the model refused or broke the schema."""

        async def run(model: str):
            return await self._create(
                name,
                model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                **({"thinking": t} if (t := thinking_off(model)) else {}),
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            )

        response = await self._call(name, self._fast_model, run)
        if response.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("%s returned unparseable output: %r", name, text)
            return None

    async def classify_intent(self, system: str, messages: list[dict[str, Any]]) -> Intent:
        async def run(model: str):
            return await self._create(
                "classify_intent",
                model,
                max_tokens=1024,
                system=system,
                messages=messages,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": _INTENT_SCHEMA}},
            )

        response = await self._call("classify_intent", self._model, run)
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
        *,
        system_suffix: str | None = None,
        name: str = "respond",
    ) -> LLMTurn:
        # Cache the stable prefix (tools + system); per-turn context goes in a second, uncached block.
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        if system_suffix:
            system_blocks.append({"type": "text", "text": system_suffix})
        emitted = False

        def emit(text: str) -> None:
            nonlocal emitted
            emitted = True
            on_text(text)

        async def run(model: str):
            with get_tracer().generation(
                name, model=model, input=preview(messages), model_parameters={"max_tokens": 16000}
            ) as gen:
                started = time.monotonic()
                first_token: float | None = None
                api, params = self._messages_api(
                    model,
                    {
                        "max_tokens": 16000,
                        "system": system_blocks,
                        "messages": messages,
                        **({"tools": tools} if tools else {}),
                        "output_config": {"effort": "low"},
                    },
                )
                async with api.stream(model=model, **params) as stream:
                    async for text in stream.text_stream:
                        first_token = first_token or time.monotonic()
                        emit(text)
                    final = await stream.get_final_message()
                fields = self._meter(name, model, final)
                fields["metadata"].update(
                    seconds=round(time.monotonic() - started, 3),
                    time_to_first_token=round(first_token - started, 3) if first_token else None,
                )
                gen.update(output=[_to_param(b) for b in final.content if b.type != "thinking"], **fields)
                return final

        final = await self._call(name, self._model, run, can_fall_back=lambda: not emitted)
        if final.stop_reason == "refusal":
            on_text(REFUSAL_REPLY)
            return LLMTurn(content=[{"type": "text", "text": REFUSAL_REPLY}], stop_reason="refusal")
        # Thinking blocks go back unchanged; after a fallback the API ignores blocks the other model wrote.
        return LLMTurn(content=_assistant_content(final.content), stop_reason=final.stop_reason)


def _params_of(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"max_tokens": params.get("max_tokens")}
    if "thinking" in params:
        out["thinking"] = params["thinking"]["type"]
    if effort := params.get("output_config", {}).get("effort"):
        out["effort"] = effort
    return out
