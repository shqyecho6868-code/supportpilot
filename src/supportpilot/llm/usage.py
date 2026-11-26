"""Token and cost accounting for Claude calls.

Every model call records its usage into the ``UsageMeter`` of the current context (one per
conversation turn, set by the API or the eval runner) and into a process-wide meter. The
per-turn meter feeds the conversation token limit and the Langfuse trace; the process meter
reports what an evaluation run cost.
"""

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# USD per million tokens: (input, output). Cache writes cost 1.25x input, cache reads 0.1x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def _price(model: str) -> tuple[float, float]:
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price
    return PRICES["claude-sonnet-5-5"]  # unknown model: assume a mid-tier price rather than zero


@dataclass(frozen=True)
class CallUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def from_response(cls, model: str, usage: Any) -> "CallUsage":
        """From an Anthropic ``Usage`` object (or None)."""
        if usage is None:
            return cls(model)
        return cls(
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    def cost(self) -> dict[str, float]:
        """Cost in USD, split the way Langfuse expects ``cost_details``."""
        price_in, price_out = _price(self.model)
        input_cost = (
            self.input_tokens
            + self.cache_write_tokens * CACHE_WRITE_MULTIPLIER
            + self.cache_read_tokens * CACHE_READ_MULTIPLIER
        ) * (price_in / 1e6)
        output_cost = self.output_tokens * (price_out / 1e6)
        return {"input": input_cost, "output": output_cost, "total": input_cost + output_cost}

    def usage_details(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_tokens,
            "cache_creation_input_tokens": self.cache_write_tokens,
        }


@dataclass
class UsageMeter:
    calls: list[CallUsage] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: CallUsage) -> None:
        with self._lock:
            self.calls.append(usage)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def cost(self) -> float:
        return sum(c.cost()["total"] for c in self.calls)

    def by_model(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for c in self.calls:
            row = out.setdefault(c.model, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["tokens"] += c.total_tokens
            row["cost_usd"] += c.cost()["total"]
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "tokens": self.total_tokens,
            "cost_usd": round(self.cost, 6),
            "by_model": self.by_model(),
        }


PROCESS_METER = UsageMeter()
_current: contextvars.ContextVar[UsageMeter | None] = contextvars.ContextVar("usage_meter", default=None)


def current_meter() -> UsageMeter | None:
    return _current.get()


@contextmanager
def metered() -> Iterator[UsageMeter]:
    """Give the enclosed code (and the tasks it starts) a fresh per-turn meter."""
    meter = UsageMeter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def record(usage: CallUsage) -> None:
    PROCESS_METER.add(usage)
    if (meter := _current.get()) is not None:
        meter.add(usage)
