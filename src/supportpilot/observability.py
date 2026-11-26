"""Langfuse tracing.

One trace per conversation turn (``/chat``) or confirmation (``/confirm``), grouped into a
Langfuse session by ``session_id``. Inside it: the router decision, every Claude call
(a generation with model, tokens, cost and latency), every tool call, retrieval results and
guard decisions. Evaluation runs report one trace per case, tagged ``eval``.

Tracing is off unless ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set; then every
helper here is a cheap no-op, so the offline test suite never needs Langfuse.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)


class _NoopObservation:
    def update(self, **_: Any) -> "_NoopObservation":
        return self


_NOOP = _NoopObservation()


class Tracer:
    def __init__(self, client: Any | None = None):
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        input: Any = None,
    ) -> Iterator[Any]:
        """A root observation whose trace carries user, session, tags and metadata."""
        if self._client is None:
            yield _NOOP
            return
        from langfuse import propagate_attributes

        with (
            self._client.start_as_current_observation(name=name, as_type="agent", input=input) as root,
            propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                tags=tags,
                metadata={k: str(v)[:200] for k, v in (metadata or {}).items()},
                trace_name=name,
            ),
        ):
            yield root

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", input: Any = None, metadata: Any = None):
        """A child observation. ``as_type``: span, tool, retriever, guardrail, chain, evaluator."""
        if self._client is None:
            yield _NOOP
            return
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        ) as obs:
            yield obs

    @contextmanager
    def generation(self, name: str, *, model: str, input: Any = None, model_parameters: Any = None):
        if self._client is None:
            yield _NOOP
            return
        with self._client.start_as_current_observation(
            name=name, as_type="generation", model=model, input=input, model_parameters=model_parameters
        ) as gen:
            yield gen

    def trace_id(self) -> str | None:
        return self._client.get_current_trace_id() if self._client is not None else None

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001 - tracing must never break the app
                log.warning("langfuse flush failed", exc_info=True)


_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    global _tracer
    _tracer = tracer


def configure_tracing(
    *,
    public_key: str | None,
    secret_key: str | None,
    base_url: str | None = None,
    environment: str | None = None,
    enabled: bool = True,
    **client_options: Any,
) -> Tracer:
    """Install the process-wide tracer. Without keys (or with enabled=False) tracing stays off."""
    global _tracer
    if not (enabled and public_key and secret_key):
        _tracer = Tracer()
        return _tracer
    from langfuse import Langfuse

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url or None,
        environment=environment or None,
        **client_options,  # tests pass span_exporter / tracer_provider
    )
    _tracer = Tracer(client)
    log.info("Langfuse tracing on (%s)", base_url or "https://cloud.langfuse.com")
    return _tracer


def preview(messages: list[dict[str, Any]], limit: int = 4000) -> list[dict[str, Any]]:
    """A compact copy of a message list for trace inputs (long tool results are truncated)."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content[:limit]})
            continue
        blocks = []
        for b in content:
            b = dict(b)
            for key in ("text", "content"):
                if isinstance(b.get(key), str):
                    b[key] = b[key][:limit]
            if b.get("type") == "thinking":
                b = {"type": "thinking"}
            blocks.append(b)
        out.append({"role": m["role"], "content": blocks})
    return out
