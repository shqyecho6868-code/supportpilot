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
        case _:
            return block.model_dump(mode="json", exclude_none=True)


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
        self.fallbacks_used = 0  # observable in tests and logs

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

    async def _create(self, name: str, model: str, **params: Any) -> Any:
        """One non-streaming request, traced and metered."""
        with get_tracer().generation(
            name, model=model, input=preview(params.get("messages", [])), model_parameters=_params_of(params)
        ) as gen:
            started = time.monotonic()
            response = await self._client.messages.create(model=model, **params)
            usage = CallUsage.from_response(model, response.usage)
            record(usage)
            gen.update(
                output="".join(getattr(b, "text", "") for b in response.content),
                usage_details=usage.usage_details(),
                cost_details=usage.cost(),
                metadata={
                    "stop_reason": response.stop_reason,
                    "seconds": round(time.monotonic() - started, 3),
                },
            )
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
                async with self._client.messages.stream(
                    model=model,
                    max_tokens=16000,
                    system=system_blocks,
                    messages=messages,
                    **({"tools": tools} if tools else {}),
                    output_config={"effort": "low"},
                ) as stream:
                    async for text in stream.text_stream:
                        first_token = first_token or time.monotonic()
                        emit(text)
                    final = await stream.get_final_message()
                usage = CallUsage.from_response(model, final.usage)
                record(usage)
                gen.update(
                    output=[_to_param(b) for b in final.content if b.type != "thinking"],
                    usage_details=usage.usage_details(),
                    cost_details=usage.cost(),
                    metadata={
                        "stop_reason": final.stop_reason,
                        "seconds": round(time.monotonic() - started, 3),
                        "time_to_first_token": round(first_token - started, 3) if first_token else None,
                    },
                )
                return final

        final = await self._call(name, self._model, run, can_fall_back=lambda: not emitted)
        if final.stop_reason == "refusal":
            on_text(REFUSAL_REPLY)
            return LLMTurn(content=[{"type": "text", "text": REFUSAL_REPLY}], stop_reason="refusal")
        # Thinking blocks go back unchanged; after a fallback the API ignores blocks the other model wrote.
        return LLMTurn(content=[_to_param(b) for b in final.content], stop_reason=final.stop_reason)


def _params_of(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"max_tokens": params.get("max_tokens")}
    if "thinking" in params:
        out["thinking"] = params["thinking"]["type"]
    if effort := params.get("output_config", {}).get("effort"):
        out["effort"] = effort
    return out
