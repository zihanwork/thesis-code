from __future__ import annotations

from dataclasses import dataclass

from embodied_gap.core.task_schema import Task


@dataclass(frozen=True)
class FailurePattern:
    name: str
    diagnosis: str
    macro_hint: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "diagnosis": self.diagnosis,
            "macro_hint": self.macro_hint,
        }


PATTERNS = {
    "virtualhome_appliance_surface_activation": FailurePattern(
        name="virtualhome_appliance_surface_activation",
        diagnosis="Appliance tasks combine put_on goals with closed/on/plugged_in goals.",
        macro_hint="Open appliance only when needed to retrieve objects, put objects on target, close, plug in, then switch on.",
    ),
}


def classify_failure_patterns(task: Task) -> tuple[FailurePattern, ...]:
    names: set[str] = set()
    dataset = task.slots.get("dataset")
    goal_names = {predicate_name(goal) for goal in task.goal_facts}

    if dataset == "virtualhome":
        if "obj_ontop" in goal_names and ({"closed", "on", "plugged_in"} & goal_names):
            names.add("virtualhome_appliance_surface_activation")

    return tuple(PATTERNS[name] for name in sorted(names))


def predicate_name(fact: str) -> str:
    text = fact
    if text.startswith("not(") and text.endswith(")"):
        text = text[4:-1]
    if "(" not in text:
        return text
    return text.split("(", 1)[0]
