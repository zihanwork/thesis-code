from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .action_schema import Action, Fact


class ViolationType(StrEnum):
    PARSING = "parsing_error"
    HALLUCINATION = "hallucination"
    ACTION_ARG_NUM = "action_argument_number_error"
    AFFORDANCE = "affordance_error"
    ADDITIONAL_STEP = "additional_step"
    MISSING_STEP = "missing_step"
    WRONG_ORDER = "wrong_order"
    SAFETY = "safety_violation"
    GOAL_UNSATISFIED = "goal_unsatisfied"
    NO_PLAN = "no_plan"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Violation:
    type: ViolationType
    message: str
    step_index: int | None = None
    action: Action | None = None
    missing_preconditions: tuple[Fact, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "message": self.message,
            "step_index": self.step_index,
            "action": self.action,
            "missing_preconditions": list(self.missing_preconditions),
            "details": self.details,
        }
