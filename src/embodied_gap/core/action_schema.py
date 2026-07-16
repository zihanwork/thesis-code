from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

Fact = str
Action = str

_CALL_RE = re.compile(r"^\s*([a-zA-Z_][\w-]*)\((.*)\)\s*$")


@dataclass(frozen=True)
class ActionCall:
    """Grounded action or fact-like predicate call."""

    name: str
    args: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "ActionCall":
        match = _CALL_RE.match(text)
        if not match:
            return cls(name=text.strip(), args=())
        raw_args = match.group(2).strip()
        args = tuple(arg.strip() for arg in raw_args.split(",")) if raw_args else ()
        return cls(name=match.group(1), args=args)

    def format(self) -> str:
        if not self.args:
            return f"{self.name}()"
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True)
class ActionSpec:
    """Symbolic transition model for one grounded action."""

    preconditions: tuple[Fact, ...] = ()
    add_effects: tuple[Fact, ...] = ()
    del_effects: tuple[Fact, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpec":
        return cls(
            preconditions=tuple(data.get("preconditions", [])),
            add_effects=tuple(data.get("add_effects", [])),
            del_effects=tuple(data.get("del_effects", [])),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "preconditions": list(self.preconditions),
            "add_effects": list(self.add_effects),
            "del_effects": list(self.del_effects),
        }


def parse_call(text: str) -> ActionCall:
    return ActionCall.parse(text)


def predicate_name(text: str) -> str:
    return parse_call(text).name


def predicate_args(text: str) -> tuple[str, ...]:
    return parse_call(text).args


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
