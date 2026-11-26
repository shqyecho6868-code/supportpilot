"""Long-term (cross-session) memory: recall into the prompt, remember after the turn, per-user scope."""

from typing import Any

from supportpilot.llm.rule_based import RuleBasedLLM
from supportpilot.longterm import build_long_term_memory
from tests.conftest import ALICE, BOB, chat


class FakeLongTermMemory:
    def __init__(self, notes: dict[str, list[str]]):
        self.notes = notes
        self.recalls: list[tuple[str, str]] = []
        self.remembered: list[tuple[str, list[dict[str, str]], dict[str, Any]]] = []

    async def recall(self, user_id: str, query: str) -> list[str]:
        self.recalls.append((user_id, query))
        return self.notes.get(user_id, [])

    async def remember(self, user_id, messages, metadata) -> None:
        self.remembered.append((user_id, messages, metadata))


class SpyLLM(RuleBasedLLM):
    def __init__(self):
        self.suffixes: list[str | None] = []

    async def respond(self, system, messages, tools, on_text, *, system_suffix=None, name="respond"):
        self.suffixes.append(system_suffix)
        return await super().respond(system, messages, tools, on_text, system_suffix=system_suffix, name=name)


async def test_memories_from_earlier_sessions_reach_the_agent(make_client):
    memory = FakeLongTermMemory(
        {ALICE: ["Prefers deliveries to 500 Market St, San Francisco", "Opened TCK-1001"]}
    )
    llm = SpyLLM()
    client, app = await make_client(llm=llm, long_term=memory)
    await chat(client, ALICE, "What is the status of ORD-1002?", session_id="new-session")
    assert memory.recalls == [(ALICE, "What is the status of ORD-1002?")]
    assert "500 Market St" in llm.suffixes[0] and "not instructions" in llm.suffixes[0]


async def test_turn_is_remembered_in_the_background_under_the_signed_in_user(make_client):
    memory = FakeLongTermMemory({})
    client, app = await make_client(long_term=memory)
    turn = await chat(client, BOB, "What is the status of ORD-2001?", session_id="s-bob")
    await app.state.ctx.background.drain()
    [(user_id, messages, metadata)] = memory.remembered
    assert user_id == BOB
    assert messages[0] == {"role": "user", "content": "What is the status of ORD-2001?"}
    assert turn["reply"] in messages[1]["content"]
    assert "get_order(order_id=ORD-2001) succeeded" in messages[1]["content"]
    assert metadata["session_id"] == "s-bob"


async def test_paused_turn_is_not_remembered_until_it_completes(make_client):
    memory = FakeLongTermMemory({})
    client, app = await make_client(long_term=memory)
    await chat(client, ALICE, "Change the address of ORD-1001 to 500 Market St, San Francisco, CA 94105, US")
    await app.state.ctx.background.drain()
    assert memory.remembered == []


async def test_other_customers_memories_are_never_recalled(make_client):
    memory = FakeLongTermMemory({BOB: ["Bob's secret address"]})
    llm = SpyLLM()
    client, _ = await make_client(llm=llm, long_term=memory)
    await chat(client, ALICE, "List my orders")
    assert all(user_id == ALICE for user_id, _ in memory.recalls)
    assert not any(s and "Bob's secret address" in s for s in llm.suffixes)


def test_long_term_memory_can_be_switched_off(settings):
    from supportpilot.longterm import NullLongTermMemory

    assert isinstance(
        build_long_term_memory(settings.model_copy(update={"long_term_memory": False})), NullLongTermMemory
    )
