"""Golden evaluation cases (evals/golden.jsonl)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals"
GOLDEN_FILE = EVALS_DIR / "golden.jsonl"
BASELINE_FILE = EVALS_DIR / "baseline.json"
RESULTS_DIR = EVALS_DIR / "results"
SUBSETS_DIR = EVALS_DIR / "subsets"

KNOWLEDGE_CATEGORIES = frozenset({"policy", "policy_multi_doc", "policy_multi_turn"})
SECURITY_CATEGORIES = frozenset({"cross_user", "prompt_injection"})

# The seven ticket types every case is labelled with (the "type" field), in report order.
CASE_TYPES = (
    "order_lookup",
    "address_change",
    "policy",
    "after_sales",
    "mixed_intent",
    "adversarial",
    "smalltalk_handoff",
)


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    user_id: str
    message: str
    reference: str
    expected_route: tuple[str, ...]
    type: str = ""  # one of CASE_TYPES
    expected_tools: tuple[str, ...] = ()  # each entry may be "a|b": at least one of them must run
    forbidden_tools: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()  # relevant chunk ids, for retrieval metrics
    history: tuple[dict[str, str], ...] = ()  # earlier turns given as text (not run)
    # Earlier customer messages that are run live through the graph before ``message``, in the
    # same session, so their tool calls and confirmations really happen.
    turns: tuple[str, ...] = ()
    turn_routes: tuple[tuple[str, ...], ...] = ()  # accepted routes for each of ``turns``
    expect_ticket: bool | None = False  # None: a ticket is acceptable either way
    expect_confirmation: bool = False
    expect_address: dict[str, str] | None = None
    protected_orders: tuple[str, ...] = ()
    secrets: tuple[str, ...] = ()
    # Quantities the reply must state ("30 days", "$6.95", "15%"), checked deterministically.
    expect_numbers: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_knowledge(self) -> bool:
        """Cases answered from the knowledge base (they have labelled sources)."""
        return bool(self.expected_sources)

    @property
    def messages(self) -> tuple[str, ...]:
        """Every customer message run live, in order (the last one is ``message``)."""
        return (*self.turns, self.message)

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.turns or self.history)

    @property
    def is_security(self) -> bool:
        return self.category in SECURITY_CATEGORIES and bool(self.protected_orders or self.secrets)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        route = d["expected_route"]
        if d.get("type") not in CASE_TYPES:
            raise ValueError(f"case {d['id']}: type must be one of {CASE_TYPES}")
        turn_routes = tuple((r,) if isinstance(r, str) else tuple(r) for r in d.get("turn_routes", []))
        if turn_routes and len(turn_routes) != len(d.get("turns", [])):
            raise ValueError(f"case {d['id']}: turn_routes needs one entry per turn")
        return cls(
            id=d["id"],
            category=d["category"],
            user_id=d["user_id"],
            message=d["message"],
            reference=d["reference"],
            expected_route=tuple(route) if isinstance(route, list) else (route,),
            type=d["type"],
            expected_tools=tuple(d.get("expected_tools", [])),
            forbidden_tools=tuple(d.get("forbidden_tools", [])),
            expected_sources=tuple(d.get("expected_sources", [])),
            history=tuple(d.get("history", [])),
            turns=tuple(d.get("turns", [])),
            turn_routes=turn_routes,
            expect_ticket=d.get("expect_ticket", d["category"] == "handoff"),
            expect_confirmation=d.get("expect_confirmation", False),
            expect_address=d.get("expect_address"),
            protected_orders=tuple(d.get("protected_orders", [])),
            secrets=tuple(d.get("secrets", [])),
            expect_numbers=tuple(d.get("expect_numbers", [])),
            raw=d,
        )


def load_cases(path: Path = GOLDEN_FILE) -> list[Case]:
    cases = [Case.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in golden set")
    return cases


def load_subset(name: str) -> list[str]:
    """Case ids listed in evals/subsets/<name>.txt (one per line, # comments allowed)."""
    lines = (SUBSETS_DIR / f"{name}.txt").read_text().splitlines()
    return [line.split("#", 1)[0].strip() for line in lines if line.split("#", 1)[0].strip()]
