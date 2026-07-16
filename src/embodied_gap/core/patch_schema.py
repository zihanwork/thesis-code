from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .action_schema import Action


class PatchType(StrEnum):
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    FULL_REPLAN = "full_replan"
    REJECT = "reject"
    NONE = "none"


@dataclass(frozen=True)
class PlanPatch:
    patch_type: PatchType
    source: str
    before: tuple[Action, ...]
    after: tuple[Action, ...]
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def changed(self) -> bool:
        return self.before != self.after

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_type": self.patch_type.value,
            "source": self.source,
            "before": list(self.before),
            "after": list(self.after),
            "explanation": self.explanation,
            "metadata": self.metadata,
        }
