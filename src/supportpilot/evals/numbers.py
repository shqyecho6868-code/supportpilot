"""Deterministic number check for answers about days and amounts.

The LLM task judge checks key facts, but it passed a wrong computed total once ("roughly 7 to 13
business days" where the policy adds up to 10 to 15). For cases that list ``expect_numbers`` in
the golden set, two checks run in addition to the judge:

* every expected quantity appears in the reply (``numbers_present``);
* every quantity with a unit in the reply (days, weeks, months, years, hours, $, %) appears in
  the material the reply may draw on: the reference, the retrieved passages and tool results,
  and the conversation (``numbers_supported``). A total or conversion the model computed itself
  shows up here as unsupported.

A quantity is a (value, unit) pair: "15-day window" -> (15, "day"), "$1,000" -> (1000, "$"),
"5 to 10 business days" -> (5, "day") and (10, "day").
"""

import re
from dataclasses import dataclass

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty-five": 45, "sixty": 60, "ninety": 90,
}  # fmt: skip
_UNITS = {
    "day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month",
    "months": "month", "year": "year", "years": "year", "hour": "hour", "hours": "hour",
    "percent": "%", "%": "%",
}  # fmt: skip

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")"
_UNIT = r"(?:days?|weeks?|months?|years?|hours?|percent|%)"
# "5 to 10 business days", "15-day", "2-3 business days", "30 or 60 days"
_WITH_UNIT = re.compile(
    rf"\b(?P<a>{_NUM})(?:\s*(?:-|–|—|to|and|or)\s*(?P<b>{_NUM}))?\s*-?\s*"
    rf"(?:(?:business|calendar|working)\s+)?(?P<unit>{_UNIT})(?![a-z])",
    re.I,
)
_MONEY = re.compile(rf"\$\s?(?P<a>{_NUM})", re.I)
_BARE = re.compile(rf"(?<![\w.]){_NUM}(?![\w])", re.I)
# Citations like [Returns Policy § Electronics] and the Sources footer carry no claims.
_CITATION = re.compile(r"\[[^\[\]]*§[^\[\]]*\]")
_SOURCES = re.compile(r"\n\nSources: .*\Z", re.S)

Quantity = tuple[float, str]


def _value(token: str) -> float:
    token = token.lower()
    return float(_WORDS[token]) if token in _WORDS else float(token.replace(",", ""))


def _clean(text: str) -> str:
    return _CITATION.sub(" ", _SOURCES.sub("", text))


def quantities(text: str) -> set[Quantity]:
    """Every (value, unit) pair in the text."""
    text = _clean(text)
    found: set[Quantity] = set()
    for m in _WITH_UNIT.finditer(text):
        unit = _UNITS[m["unit"].lower()]
        found.add((_value(m["a"]), unit))
        if m["b"]:
            found.add((_value(m["b"]), unit))
    for m in _MONEY.finditer(text):
        found.add((_value(m["a"]), "$"))
    return found


def bare_numbers(text: str) -> set[float]:
    return {_value(m.group(0)) for m in _BARE.finditer(_clean(text))}


def _fmt(q: Quantity) -> str:
    value = f"{q[0]:g}"
    return f"${value}" if q[1] == "$" else f"{value}{q[1]}" if q[1] == "%" else f"{value} {q[1]}"


@dataclass(frozen=True)
class NumberCheck:
    missing: tuple[str, ...]  # expected quantities the reply does not state
    unsupported: tuple[str, ...]  # quantities in the reply found nowhere in the allowed material

    @property
    def present(self) -> bool:
        return not self.missing

    @property
    def supported(self) -> bool:
        return not self.unsupported

    @property
    def ok(self) -> bool:
        return self.present and self.supported


def check_numbers(reply: str, expected: list[str], allowed_sources: list[str]) -> NumberCheck:
    """Compare the reply's numbers with the expected ones and with what the sources support.

    An expected quantity also counts as stated when its value appears without a unit ("30 days
    (60 for Plus members)"), since the unit is then carried by the sentence.
    """
    in_reply = quantities(reply)
    reply_values = bare_numbers(reply)
    wanted: set[Quantity] = set()
    for item in expected:
        wanted |= quantities(item)
    missing = sorted(q for q in wanted if q not in in_reply and q[0] not in reply_values)
    allowed = set().union(*(quantities(s) for s in allowed_sources)) if allowed_sources else set()
    unsupported = sorted(in_reply - allowed - wanted)
    return NumberCheck(tuple(_fmt(q) for q in missing), tuple(_fmt(q) for q in unsupported))
