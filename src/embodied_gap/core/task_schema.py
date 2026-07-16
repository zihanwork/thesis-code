from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .action_schema import Action, ActionSpec, Fact
from .goal_schema import GoalSpec


@dataclass(frozen=True)
class SafetyRules:
    should_reject: bool = False
    forbidden_actions: tuple[Action, ...] = ()
    forbidden_goal_facts: tuple[Fact, ...] = ()
    hazard_keywords: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SafetyRules":
        if not data:
            return cls()
        return cls(
            should_reject=bool(data.get("should_reject", False)),
            forbidden_actions=tuple(data.get("forbidden_actions", [])),
            forbidden_goal_facts=tuple(data.get("forbidden_goal_facts", [])),
            hazard_keywords=tuple(data.get("hazard_keywords", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_reject": self.should_reject,
            "forbidden_actions": list(self.forbidden_actions),
            "forbidden_goal_facts": list(self.forbidden_goal_facts),
            "hazard_keywords": list(self.hazard_keywords),
        }


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    initial_facts: tuple[Fact, ...]
    goal: GoalSpec
    allowed_actions: tuple[Action, ...]
    action_model: dict[Action, ActionSpec]
    split: str = "eval"
    tags: tuple[str, ...] = ()
    slots: dict[str, str] = field(default_factory=dict)
    gold_plan: tuple[Action, ...] = ()
    safety_rules: SafetyRules = field(default_factory=SafetyRules)
    source: str = "jsonl"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def goal_facts(self) -> tuple[Fact, ...]:
        return self.goal.final_facts

    @property
    def is_hazardous(self) -> bool:
        return self.safety_rules.should_reject or "hazard" in self.tags

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            id=data["id"],
            split=data.get("split", "eval"),
            instruction=data["instruction"],
            tags=tuple(data.get("tags", [])),
            slots=dict(data.get("slots", {})),
            initial_facts=tuple(data.get("initial_facts", [])),
            goal=GoalSpec(tuple(data.get("goal_facts", []))),
            allowed_actions=tuple(data.get("allowed_actions", [])),
            gold_plan=tuple(data.get("gold_plan", [])),
            safety_rules=SafetyRules.from_dict(data.get("safety_rules")),
            action_model={
                name: ActionSpec.from_dict(spec)
                for name, spec in data.get("action_model", {}).items()
            },
            source=data.get("source", "jsonl"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "split": self.split,
            "instruction": self.instruction,
            "tags": list(self.tags),
            "slots": dict(self.slots),
            "initial_facts": list(self.initial_facts),
            "goal_facts": list(self.goal.final_facts),
            "allowed_actions": list(self.allowed_actions),
            "gold_plan": list(self.gold_plan),
            "action_model": {
                name: spec.to_dict() for name, spec in self.action_model.items()
            },
            "source": self.source,
            "metadata": self.metadata,
        }
        if self.safety_rules != SafetyRules():
            payload["safety_rules"] = self.safety_rules.to_dict()
        return payload


def load_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(Task.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return tasks


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
