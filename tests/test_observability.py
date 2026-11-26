"""Tracing: one trace per turn with router, guard, tool and retrieval observations.

Uses the real Langfuse client with an in-memory OpenTelemetry exporter, so nothing leaves the
machine and no keys are needed.
"""

import uuid

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from supportpilot import observability
from tests.conftest import ALICE, chat


@pytest.fixture
def exported(monkeypatch):
    exporter = InMemorySpanExporter()
    tracer = observability.configure_tracing(
        public_key=f"pk-lf-test-{uuid.uuid4().hex}",  # Langfuse caches one client per public key
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    # The app configures tracing from settings at startup; keep this tracer instead.
    monkeypatch.setattr("supportpilot.api.app.configure_tracing", lambda **_: tracer)
    yield exporter
    observability.set_tracer(observability.Tracer())


def test_tracing_is_off_without_keys():
    assert not observability.configure_tracing(public_key=None, secret_key=None).enabled
    assert not observability.configure_tracing(public_key="pk", secret_key="sk", enabled=False).enabled


async def test_a_turn_is_one_trace_with_nested_observations(exported, client):
    turn = await chat(client, ALICE, "What is the status of ORD-1002?", session_id="s-trace")
    assert turn["error"] is None, turn["names"]
    observability.get_tracer().flush()
    spans = exported.get_finished_spans()
    names = [s.name for s in spans]
    assert {"chat_turn", "guard.user_input", "router", "tool:get_order"} <= set(names)
    assert len({s.context.trace_id for s in spans}) == 1
    root = next(s for s in spans if s.name == "chat_turn")
    assert all(s.parent is not None for s in spans if s is not root)
    attrs = dict(root.attributes)
    assert attrs["session.id"] == "s-trace" and attrs["user.id"] == ALICE
    assert "chat" in attrs["langfuse.trace.tags"]
    # Every observation carries the session, so Langfuse groups the turn under the conversation.
    assert all(dict(s.attributes).get("session.id") == "s-trace" for s in spans)


async def test_retrieval_results_are_traced(exported, client):
    await chat(client, ALICE, "What is your return window policy?")
    observability.get_tracer().flush()
    names = [s.name for s in exported.get_finished_spans()]
    assert "retrieval" in names and "tool:search_policy" in names
