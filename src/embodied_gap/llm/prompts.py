from __future__ import annotations

from embodied_gap.core.action_schema import predicate_name
from embodied_gap.core.task_schema import Task
from embodied_gap.execution.pddl_executor import (
    load_domain,
    object_types,
    resolve_domain_path,
    type_compatible,
)


OBJECT_CANDIDATE_LIMIT = 8
PLANNING_PROMPT_VERSION = "planning_prompt_v2"
REPAIR_PROMPT_VERSION = "repair_prompt_v2"


def render_planning_prompt(
    task: Task,
    strategy: str,
    extra_context: str = "",
    *,
    profile: str = "structured",
) -> str:
    if profile == "minimal":
        return "\n".join(
            [
                "Return only a JSON list of grounded high-level actions.",
                f"Instruction: {task.instruction}",
                f"Allowed action names: {list(task.allowed_actions)}",
            ]
        )
    if profile not in {"structured", "engineered"}:
        raise ValueError(f"Unsupported planning prompt profile: {profile}")

    objects = task.metadata.get("objects", {})
    object_line = ""
    if isinstance(objects, dict) and objects:
        rendered_objects = ", ".join(
            f"{name}:{type_name}" for name, type_name in sorted(objects.items())
        )
        object_line = f"Objects: {rendered_objects}"
    lines = [
        "You are an embodied task planner.",
        "Return only a JSON list of grounded high-level actions.",
        "Use exactly the action names, argument counts, and argument order provided below.",
        "Do not invent object names. Do not include explanations or markdown.",
        f"Strategy: {strategy}",
        f"Instruction: {task.instruction}",
        f"Initial facts: {list(task.initial_facts)}",
        f"Goal facts: {list(task.goal_facts)}",
        f"Allowed action names: {list(task.allowed_actions)}",
    ]
    if object_line:
        lines.append(object_line)
    action_schema_context = render_action_schema_context(task)
    if action_schema_context:
        lines.append(f"PDDL action signatures:\n{action_schema_context}")
    if extra_context:
        lines.append(f"Extra context:\n{extra_context}")
    if profile == "engineered":
        lines.extend(
            [
                "Before returning the final list, internally verify that:",
                "1. every action uses a listed signature and the exact argument count;",
                "2. every object is present in the provided object candidates;",
                "3. actions are ordered so their preconditions are established;",
                "4. the final state satisfies every goal fact;",
                "5. the response contains only the final JSON list.",
            ]
        )
    return "\n".join(lines)


def render_action_schema_context(task: Task) -> str:
    domain_path = resolve_domain_path(task)
    if domain_path is None:
        return ""

    domain = load_domain(domain_path)
    objects = object_types(task)
    allowed_names = {predicate_name(action) for action in task.allowed_actions}
    if not allowed_names:
        allowed_names = set(domain.actions)

    lines: list[str] = []
    for name in sorted(allowed_names):
        schema = domain.actions.get(name)
        if schema is None:
            continue
        parameters = [
            f"{parameter.name.lstrip('?')}:{parameter.type_name}"
            for parameter in schema.parameters
        ]
        candidates = []
        feasible = True
        for parameter in schema.parameters:
            matching_objects = sorted(
                object_name
                for object_name, object_type in objects.items()
                if type_compatible(object_type, parameter.type_name)
            )
            if not matching_objects:
                feasible = False
                break
            candidates.append(
                f"{parameter.name.lstrip('?')} in [{format_object_candidates(matching_objects)}]"
            )
        if not feasible:
            continue
        candidate_text = "; ".join(candidates) if candidates else "no arguments"
        lines.append(f"- {name}({', '.join(parameters)}) | candidates: {candidate_text}")
    return "\n".join(lines)


def format_object_candidates(objects: list[str]) -> str:
    if not objects:
        return ""
    visible = objects[:OBJECT_CANDIDATE_LIMIT]
    suffix = ""
    remaining = len(objects) - len(visible)
    if remaining > 0:
        suffix = f", ... (+{remaining})"
    return ", ".join(visible) + suffix


def render_repair_prompt(
    task: Task,
    actions: tuple[str, ...],
    feedback: str,
    *,
    repair_guidance: str = "",
    memory_context: str = "",
) -> str:
    schema_context = render_action_schema_context(task)
    lines = [
        "Repair the embodied action plan using the explicit execution feedback.",
        "Return only a JSON list of grounded high-level actions.",
        "Do not explain the repair and do not use markdown.",
        f"Instruction: {task.instruction}",
        f"Initial facts: {list(task.initial_facts)}",
        f"Goal facts: {list(task.goal_facts)}",
        f"Current plan: {list(actions)}",
        f"Execution feedback: {feedback}",
        f"Allowed action names: {list(task.allowed_actions)}",
    ]
    if schema_context:
        lines.append(f"PDDL action signatures:\n{schema_context}")
    if repair_guidance:
        lines.append(f"Error-specific repair guidance:\n{repair_guidance}")
    if memory_context:
        lines.append(f"Frozen failure memory:\n{memory_context}")
    lines.append(
        "Return a complete replacement plan that fixes the reported error and still achieves all goals."
    )
    return "\n".join(lines)
