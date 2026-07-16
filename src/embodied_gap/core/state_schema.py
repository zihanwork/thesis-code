from __future__ import annotations

from dataclasses import dataclass

from .action_schema import ActionSpec, Fact


@dataclass(frozen=True)
class WorldState:
    """Immutable symbolic state represented as a set of grounded facts."""

    facts: frozenset[Fact]

    @classmethod
    def from_facts(cls, facts: tuple[Fact, ...] | list[Fact] | set[Fact]) -> "WorldState":
        return cls(facts=frozenset(facts))

    def contains_all(self, facts: tuple[Fact, ...]) -> bool:
        return set(facts).issubset(self.facts)

    def missing(self, facts: tuple[Fact, ...]) -> tuple[Fact, ...]:
        return tuple(fact for fact in facts if fact not in self.facts)

    def apply(self, spec: ActionSpec) -> "WorldState":
        next_facts = set(self.facts)
        next_facts.difference_update(spec.del_effects)
        next_facts.update(spec.add_effects)
        return WorldState(facts=frozenset(next_facts))

    def to_list(self) -> list[Fact]:
        return sorted(self.facts)
