"""Golden evaluation cases (evals/golden.jsonl)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals"
GOLDEN_FILE = EVALS_DIR / "golden.jsonl"
BASELINE_FILE = EVALS_DIR / "baseline.json"
RESULTS_DIR = EVALS_DIR / "results"

KNOWLEDGE_CATEGORIES = frozenset({"policy", "policy_multi_doc", "policy_multi_turn"})
SECURITY_CATEGORIES = frozenset({"cross_user", "prompt_injection"})


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    user_id: str
    message: str
    reference: str
    expected_route: tuple[str, ...]
    expected_tools: tuple[str, ...] = ()  # each entry may be "a|b": at least one of them must run
    forbidden_tools: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()  # relevant chunk ids, for retrieval metrics
    history: tuple[dict[str, str], ...] = ()
    expect_ticket: bool = False
    expect_confirmation: bool = False
    expect_address: dict[str, str] | None = None
    protected_orders: tuple[str, ...] = ()
    secrets: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_knowledge(self) -> bool:
        """Cases answered from the knowledge base (they have labelled sources)."""
        return bool(self.expected_sources)

    @property
    def is_security(self) -> bool:
        return self.category in SECURITY_CATEGORIES and bool(self.protected_orders or self.secrets)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        route = d["expected_route"]
        return cls(
            id=d["id"],
            category=d["category"],
            user_id=d["user_id"],
            message=d["message"],
            reference=d["reference"],
            expected_route=tuple(route) if isinstance(route, list) else (route,),
            expected_tools=tuple(d.get("expected_tools", [])),
            forbidden_tools=tuple(d.get("forbidden_tools", [])),
            expected_sources=tuple(d.get("expected_sources", [])),
            history=tuple(d.get("history", [])),
            expect_ticket=d.get("expect_ticket", d["category"] == "handoff"),
            expect_confirmation=d.get("expect_confirmation", False),
            expect_address=d.get("expect_address"),
            protected_orders=tuple(d.get("protected_orders", [])),
            secrets=tuple(d.get("secrets", [])),
            raw=d,
        )


def load_cases(path: Path = GOLDEN_FILE) -> list[Case]:
    cases = [Case.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in golden set")
    return cases
