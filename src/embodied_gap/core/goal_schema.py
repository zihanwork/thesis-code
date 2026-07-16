from __future__ import annotations

from dataclasses import dataclass

from .action_schema import Fact
from .state_schema import WorldState


@dataclass(frozen=True)
class GoalSpec:
    """Final-state goals for the current symbolic implementation."""

    final_facts: tuple[Fact, ...] = ()

    def satisfaction(self, state: WorldState) -> float:
        if not self.final_facts:
            return 1.0
        achieved = sum(1 for fact in self.final_facts if self._fact_satisfied(fact, state))
        return achieved / len(self.final_facts)

    def is_satisfied(self, state: WorldState) -> bool:
        return self.satisfaction(state) == 1.0

    def _fact_satisfied(self, fact: Fact, state: WorldState) -> bool:
        if fact.startswith("not(") and fact.endswith(")"):
            positive = fact[4:-1]
            return positive not in state.facts or fact in state.facts
        return fact in state.facts
