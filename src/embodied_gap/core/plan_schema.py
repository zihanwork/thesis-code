from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .action_schema import Action


@dataclass(frozen=True)
class PlanCandidate:
    """Initial or repaired plan emitted by a planner/repair strategy."""

    planner_name: str
    actions: tuple[Action, ...]
    raw_response: str = ""
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.actions == ("reject()",)

    def with_name(self, planner_name: str) -> "PlanCandidate":
        return PlanCandidate(
            planner_name=planner_name,
            actions=self.actions,
            raw_response=self.raw_response,
            prompt=self.prompt,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_name": self.planner_name,
            "actions": list(self.actions),
            "raw_response": self.raw_response,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }
