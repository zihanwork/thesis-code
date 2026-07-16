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
    "behavior_negative_cleaning": FailurePattern(
        name="behavior_negative_cleaning",
        diagnosis="Goal requires removing dusty/stained state; search often misses tool preparation and soaking.",
        macro_hint="Select feasible cleaning tool, prepare/soak it when needed, then apply clean_* action to target.",
    ),
    "behavior_soaking": FailurePattern(
        name="behavior_soaking",
        diagnosis="Goal requires soaked objects; search often misses sink activation and held-object sequencing.",
        macro_hint="Navigate to sink, toggle it on, hold each target object, then apply soak.",
    ),
    "behavior_container_transfer": FailurePattern(
        name="behavior_container_transfer",
        diagnosis="Object starts inside a closed source container or must be placed into an opened destination.",
        macro_hint="Open source/destination containers, hold object, navigate to destination, then place_inside.",
    ),
    "behavior_surface_or_nextto_placement": FailurePattern(
        name="behavior_surface_or_nextto_placement",
        diagnosis="Goal requires ontop/nextto relation; search can miss grasp and target reachability steps.",
        macro_hint="Hold source object, navigate to target, then use place_ontop/place_nextto.",
    ),
    "behavior_floor_placement": FailurePattern(
        name="behavior_floor_placement",
        diagnosis="Goal requires placing objects on a floor surface; search can miss repeated grasp/place_onfloor cycles.",
        macro_hint="For each object, hold it, navigate to target floor, then apply place_onfloor.",
    ),
    "behavior_food_processing": FailurePattern(
        name="behavior_food_processing",
        diagnosis="Goal requires slicing/cooking/freezing before final placement.",
        macro_hint="Apply required transformation actions before container placement goals.",
    ),
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

    if dataset == "behavior":
        if {"dusty", "stained"} & goal_names:
            names.add("behavior_negative_cleaning")
        if "soaked" in goal_names:
            names.add("behavior_soaking")
        if "inside" in goal_names:
            names.add("behavior_container_transfer")
        if {"ontop", "nextto"} & goal_names:
            names.add("behavior_surface_or_nextto_placement")
        if "onfloor" in goal_names:
            names.add("behavior_floor_placement")
        if {"sliced", "cooked", "frozen"} & goal_names:
            names.add("behavior_food_processing")
    elif dataset == "virtualhome":
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
