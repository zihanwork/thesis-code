#!/usr/bin/env python3
"""Prompt variants for the EAI/VirtualHome improvement experiments.

Each variant returns a ``PromptVariant`` describing the system prompt,
optional pre-processing of the EAI ``llm_prompt`` field, an optional
critique template (for two-pass variants), and a label used for output
file naming.  Variants are programmatic counterparts to the templates
documented in ``output/diagnostics/multimodel_prompt_templates.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PromptVariant:
    label: str
    eval_type: str
    system_prompt: str
    # Wrappers accept arbitrary keyword context (``identifier`` etc.) so that
    # knowledge-grounded variants can pull scene data without breaking the
    # simple single-arg signature used by existing baseline wrappers.
    user_wrapper: Callable[..., str]
    requires_second_pass: bool = False
    critique_system_prompt: Optional[str] = None
    critique_user_wrapper: Optional[Callable[..., str]] = None
    description: str = ""


def _identity(prompt: str, **_: Any) -> str:
    return prompt


def _few_shot_action_wrapper(prompt: str, **_: Any) -> str:
    examples = (
        "Examples of valid VirtualHome action sequences:\n"
        "Task: Turn on light\n"
        'Output: {"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}\n\n'
        "Task: Sit on chair\n"
        'Output: {"WALK":["chair","245"]}{"SIT":["chair","245"]}\n\n'
        "Task: Put cup in dishwasher\n"
        'Output: {"WALK":["cup","112"]}{"GRAB":["cup","112"]}{"WALK":["dishwasher","104"]}'
        '{"OPEN":["dishwasher","104"]}{"PUTIN":["cup","112","dishwasher","104"]}'
        '{"CLOSE":["dishwasher","104"]}\n\n'
        "Now solve the user's task with the same compact format.\n\n"
    )
    return examples + prompt


def _few_shot_goal_wrapper(prompt: str, **_: Any) -> str:
    examples = (
        "Reference symbolic goal format:\n"
        '{"node goals": [{"name": "washing_machine", "state": "PLUGGED_IN"},'
        ' {"name": "washing_machine", "state": "ON"}],'
        ' "edge goals": [{"from_name": "pants", "relation": "INSIDE",'
        ' "to_name": "washing_machine"}],'
        ' "action goals": [{"action": "SWITCHON",'
        ' "description": "switch on the washing machine"}]}\n\n'
        "Use the same JSON keys (node goals, edge goals, action goals).\n\n"
    )
    return examples + prompt


def _self_check_action_wrapper(prompt: str, **_: Any) -> str:
    return prompt


def _self_check_critique_user(original_prompt: str, draft: str, **_: Any) -> str:
    return (
        "You will revise a VirtualHome action sequence draft.\n"
        "Original task and constraints:\n" + original_prompt + "\n\n"
        "Draft action sequence:\n" + draft + "\n\n"
        "Apply these checks before finalising:\n"
        "1. Every action is in the supported action list and uses the correct number of arguments.\n"
        "2. Object names and ids come from the prompt; never invent ids.\n"
        "3. WALK to an object before any action that operates on it.\n"
        "4. OPEN containers before PUTIN, CLOSE them when finished if required by the goal.\n"
        "5. Remove redundant repeated actions.\n"
        "6. Output ONLY the corrected compact JSON action sequence with no explanation."
    )


def _plan_then_ground_wrapper(prompt: str, **_: Any) -> str:
    return (
        "Step 1: Privately think through a short high-level plan.\n"
        "Step 2: Convert the plan into the compact JSON action sequence.\n"
        "Output ONLY the JSON sequence; never reveal the plan.\n\n"
    ) + prompt


def _state_checklist_plan_wrapper(prompt: str, **_: Any) -> str:
    return (
        "Before writing the final JSON, internally run this grounded execution checklist:\n"
        "1. For every node goal, identify the exact action that makes the final state true "
        "(for example SWITCHON for ON, OPEN for OPEN, GRAB/PUTBACK/PUTIN for holding or location goals).\n"
        "2. For every edge goal such as HOLDS_RH, INSIDE, ON or FACING, include the concrete action that creates that relation.\n"
        "3. Before any object operation, WALK to the operated object unless the object is already being held.\n"
        "4. Before READ, DRINK, EAT, PUTIN, or PUTBACK, make sure the target object has been GRABbed.\n"
        "5. Before PUTIN, OPEN the destination container; close it only if closing does not prevent the goal.\n"
        "6. Before TYPE, WATCH, or other device-use actions, explicitly satisfy required device states such as SWITCHON when the goal requires ON.\n"
        "7. Do not add actions that are not required by the listed goals or object properties.\n"
        "8. Finally, verify that every listed node, edge, and action goal is caused by at least one action in the sequence.\n"
        "Output ONLY the compact JSON action sequence; do not reveal the checklist or any explanation.\n\n"
    ) + prompt


def _goal_conditioned_scaffold_wrapper(prompt: str, **_: Any) -> str:
    examples = (
        "Use goal-conditioned action skeletons internally. Work backwards from the formal goals:\n"
        "- First find the action that directly makes each node, edge, or action goal true.\n"
        "- Then add only the minimal precondition actions needed before that action.\n"
        "- Do not add actions that do not support a listed goal.\n\n"
        "Internal examples, not output format:\n"
        "Goal pattern: book is held/read -> skeleton: WALK book, GRAB book, READ book.\n"
        "Goal pattern: object INSIDE container -> skeleton: WALK object, GRAB object, WALK container, OPEN container, PUTIN object container.\n"
        "Goal pattern: device ON and use device -> skeleton: WALK device, SWITCHON device, use device.\n\n"
        "Now solve the task by internally mapping goals to a minimal action skeleton, "
        "then output ONLY the compact JSON action sequence.\n\n"
    )
    return examples + prompt


def _bidirectional_causal_planning_wrapper(prompt: str, **_: Any) -> str:
    return (
        "Use bidirectional causal planning internally. Combine backward goal reasoning with forward action grounding:\n"
        "1. Backward pass (由果倒推因): inspect the formal node, edge, and action goals. Identify the exact causal actions that must happen for each final goal to become true.\n"
        "2. Precondition pass: for each causal action, add only the necessary prerequisites, such as WALK before operating on an object, GRAB before READ/DRINK/EAT/PUTIN/PUTBACK, OPEN before PUTIN, and SWITCHON before using an ON device.\n"
        "3. Forward pass (种因得果): reorder the required actions from the initial scene into a natural executable sequence that achieves the goals without unnecessary steps.\n"
        "4. Final check: every listed final goal must be caused by at least one action, and every action must use only objects and ids from the prompt.\n\n"
        "Internal examples, not output format:\n"
        "Goal: book is read/held -> backward causal action READ book -> prerequisites WALK book, GRAB book -> forward sequence WALK, GRAB, READ.\n"
        "Goal: cup INSIDE dishwasher -> causal action PUTIN cup dishwasher -> prerequisites WALK cup, GRAB cup, WALK dishwasher, OPEN dishwasher -> forward sequence WALK cup, GRAB cup, WALK dishwasher, OPEN dishwasher, PUTIN cup dishwasher.\n"
        "Goal: computer ON and TYPE computer -> causal actions SWITCHON computer and TYPE computer -> forward sequence WALK computer, SWITCHON computer, TYPE computer.\n\n"
        "Output ONLY the compact JSON action sequence; do not reveal the reasoning.\n\n"
    ) + prompt


def _decompose_goal_wrapper(prompt: str, **_: Any) -> str:
    return (
        "Process the goal in three internal stages before answering:\n"
        "1. List the target node states implied by the goal.\n"
        "2. List the spatial relations implied by the goal.\n"
        "3. List any necessary explicit actions.\n"
        "Then merge the three lists into the required JSON structure.\n"
        "Return ONLY the final JSON; do not output the intermediate stages.\n\n"
    ) + prompt


def _sg_rag_action_wrapper(prompt: str, identifier: Optional[str] = None, **_: Any) -> str:
    """Inject a scene-subgraph block retrieved for ``identifier`` before the prompt."""
    from scene_graph_rag import SceneGraphRetriever

    block = SceneGraphRetriever.shared().retrieve(identifier, task_prompt=prompt)
    if not block:
        return prompt
    return block + "\n\n" + prompt


def _relevant_kg_guidance(prompt: str) -> str:
    text = prompt.upper()
    lines = [
        "[Relevant Precondition KG]",
        "Use only these task-relevant action rules as planning constraints; do not output this block.",
        "- For any object operation, WALK to the operated object first unless the object is already held.",
        "- Use only object names and numeric ids that appear in the prompt or retrieved scene subgraph.",
    ]
    action_rules = [
        ("READ", "- READ object requires the object to be held; normally use WALK object, GRAB object, READ object."),
        ("DRINK", "- DRINK object requires the object to be held; normally use WALK object, GRAB object, DRINK object."),
        ("EAT", "- EAT object requires the object to be held; normally use WALK object, GRAB object, EAT object."),
        ("PUTIN", "- PUTIN object container requires holding the object, walking to the container, and opening the container first."),
        ("PUTBACK", "- PUTBACK object target requires holding the object and walking to the target first."),
        ("OPEN", "- OPEN can only be applied to openable containers or devices after walking to them."),
        ("CLOSE", "- CLOSE can only be applied to an open openable object after walking to it."),
        ("SWITCHON", "- SWITCHON device requires walking to a switchable device first."),
        ("SIT", "- SIT target requires walking to a sittable object first."),
        ("LIE", "- LIE target requires walking to a lieable object first."),
    ]
    for keyword, rule in action_rules:
        if keyword in text:
            lines.append(rule)
    if not any(keyword in text for keyword, _ in action_rules):
        lines.extend(
            [
                "- GRAB requires WALK to the grabbable object first.",
                "- If the final goal places an object inside or on another object, first GRAB the moved object, then move to the destination.",
            ]
        )
    lines.append("[/Relevant Precondition KG]")
    return "\n".join(lines)


def _kg_rag_plan_then_ground_wrapper(prompt: str, identifier: Optional[str] = None, **_: Any) -> str:
    from scene_graph_rag import SceneGraphRetriever

    scene_block = SceneGraphRetriever.shared().retrieve(
        identifier,
        task_prompt=prompt,
        k_neighbours=1,
        max_objects=12,
    )
    knowledge_parts = []
    if scene_block:
        knowledge_parts.append(scene_block)
    knowledge_parts.append(_relevant_kg_guidance(prompt))
    return (
        "Use the retrieved scene graph and relevant precondition KG as external grounding knowledge.\n"
        "First, privately make a short plan that uses the correct object ids from the scene/prompt.\n"
        "Second, ground the plan into a compact executable VirtualHome JSON action sequence.\n"
        "Prefer the minimum sufficient actions; do not add unrelated actions; output JSON only.\n\n"
        + "\n\n".join(knowledge_parts)
        + "\n\n"
        + prompt
    )


def _pc_kg_critique_user(
    original_prompt: str,
    draft: str,
    identifier: Optional[str] = None,
    **_: Any,
) -> str:
    """Critique prompt whose feedback block comes from the precondition KG."""
    from precondition_kg import PreconditionKG
    from scene_graph_rag import SceneGraphRetriever

    scene_objects = SceneGraphRetriever.shared().load_scene_objects(identifier)
    violations = PreconditionKG.shared().verify(draft, scene_objects)
    feedback = PreconditionKG.summarise(violations)
    return (
        "You will revise a VirtualHome action sequence draft.\n"
        "Original task and constraints:\n" + original_prompt + "\n\n"
        "Draft action sequence:\n" + draft + "\n\n"
        + feedback + "\n\n"
        "Apply these checks before finalising:\n"
        "1. Address every violation listed above; keep untouched steps unchanged.\n"
        "2. Object names and ids must come from the prompt; never invent ids.\n"
        "3. WALK to an object before acting on it; OPEN containers before PUTIN.\n"
        "4. Output ONLY the corrected compact JSON action sequence with no explanation."
    )


def _selective_recovery_wrapper(prompt: str, identifier: Optional[str] = None, **_: Any) -> str:
    return _plan_then_ground_wrapper(prompt, identifier=identifier)


def _selective_recovery_critique_user(
    original_prompt: str,
    draft: str,
    identifier: Optional[str] = None,
    **_: Any,
) -> str:
    """Return the deterministic local recovery as a no-op second-pass prompt."""
    from scene_graph_rag import SceneGraphRetriever
    from selective_recovery import recover_sequence

    scene_objects = SceneGraphRetriever.shared().load_scene_objects(identifier)
    recovered, report = recover_sequence(draft, scene_objects)
    audit = "; ".join(report.inserted_steps) if report.inserted_steps else report.skipped_reason
    return (
        "Return EXACTLY this conservative locally repaired VirtualHome sequence.\n"
        "Do not add, remove, reorder, or explain anything.\n"
        f"Audit: {audit}\n"
        f"Sequence:\n{recovered}"
    )


_ACTION_SELECTIVE_RECOVERY_SYSTEM = (
    "You copy the sequence supplied by the user exactly. "
    "Output ONLY the compact JSON action sequence and no commentary."
)


_ACTION_BASELINE_SYSTEM = (
    "You output ONLY the VirtualHome action sequence requested by the user. "
    "Do not include explanations, markdown, or extra text."
)

_ACTION_FORMAT_SYSTEM = (
    "You output ONLY a compact JSON action sequence for VirtualHome. "
    "Format: concatenate one or more JSON objects with no separator, e.g. "
    '{"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}. '
    "Rules: action names must be uppercase; each value is a JSON array of strings; "
    "parameters alternate object_name and numeric id; one-object actions use 2 strings, "
    "two-object actions use 4 strings; STANDUP uses []. Never wrap the answer in markdown."
)

_ACTION_FEW_SHOT_SYSTEM = _ACTION_FORMAT_SYSTEM + (
    " Follow the in-context examples exactly when constructing your output."
)

_ACTION_SELF_CHECK_SYSTEM = _ACTION_FORMAT_SYSTEM + (
    " You will produce a draft, then apply a self-check before finalising."
)

_ACTION_PLAN_SYSTEM = _ACTION_FORMAT_SYSTEM + (
    " First plan internally, then output the compact JSON action sequence only."
)

_ACTION_STATE_CHECKLIST_SYSTEM = _ACTION_FORMAT_SYSTEM + (
    " Internally verify that the sequence creates every requested final node, edge, "
    "and action goal. Pay special attention to missing precondition steps and final-state actions."
)

_ACTION_GOAL_SCAFFOLD_SYSTEM = _ACTION_FORMAT_SYSTEM + (
    " Work backwards from the listed goals to a minimal action skeleton, then output JSON only."
)

_ACTION_BIDIRECTIONAL_CAUSAL_SYSTEM = _ACTION_FORMAT_SYSTEM + (
    " Internally reason backward from final goals to necessary causal actions, "
    "then reason forward from the initial scene to order and ground those actions."
)

_ACTION_KG_RAG_PLAN_SYSTEM = _ACTION_FORMAT_SYSTEM + (
    " Use retrieved scene information and relevant precondition rules as grounding knowledge. "
    "Plan internally, then output only the executable compact JSON action sequence."
)

_GOAL_BASELINE_SYSTEM = (
    "You output ONLY a JSON object with keys node goals, edge goals, action goals. "
    "Never include markdown or commentary."
)

_GOAL_SCHEMA_SYSTEM = (
    "You output ONLY a JSON object with keys 'node goals', 'edge goals', 'action goals'. "
    "node goals is an array of {name, state} objects whose state is one of "
    "CLOSED, OPEN, ON, OFF, SITTING, DIRTY, CLEAN, LYING, PLUGGED_IN, PLUGGED_OUT. "
    "edge goals is an array of {from_name, relation, to_name} objects whose relation "
    "is one of ON, INSIDE, BETWEEN, CLOSE, FACING, HOLDS_RH, HOLDS_LH. "
    "action goals is an array of {action, description} objects. "
    "Use double-quoted JSON only; do not omit any required key, return [] if a list is empty."
)

_GOAL_FEW_SHOT_SYSTEM = _GOAL_SCHEMA_SYSTEM + (
    " Follow the format shown in the in-context example."
)

_GOAL_DECOMPOSE_SYSTEM = _GOAL_SCHEMA_SYSTEM + (
    " Reason internally before merging the three lists; output JSON only."
)


ACTION_VARIANTS: Dict[str, PromptVariant] = {
    "baseline": PromptVariant(
        label="baseline",
        eval_type="action_sequencing",
        system_prompt=_ACTION_BASELINE_SYSTEM,
        user_wrapper=_identity,
        description="Direct generation with a minimal output-only system prompt.",
    ),
    "format_constraints": PromptVariant(
        label="format_constraints",
        eval_type="action_sequencing",
        system_prompt=_ACTION_FORMAT_SYSTEM,
        user_wrapper=_identity,
        description="Strict JSON format constraints to reduce parsing and grammar errors.",
    ),
    "few_shot_valid_actions": PromptVariant(
        label="few_shot",
        eval_type="action_sequencing",
        system_prompt=_ACTION_FEW_SHOT_SYSTEM,
        user_wrapper=_few_shot_action_wrapper,
        description="Format constraints plus three valid VirtualHome action examples.",
    ),
    "self_check_rewrite": PromptVariant(
        label="self_check",
        eval_type="action_sequencing",
        system_prompt=_ACTION_SELF_CHECK_SYSTEM,
        user_wrapper=_self_check_action_wrapper,
        requires_second_pass=True,
        critique_system_prompt=_ACTION_FORMAT_SYSTEM,
        critique_user_wrapper=_self_check_critique_user,
        description="Two-pass generation: produce draft, then revise with executable checks.",
    ),
    "plan_then_ground": PromptVariant(
        label="plan_then_ground",
        eval_type="action_sequencing",
        system_prompt=_ACTION_PLAN_SYSTEM,
        user_wrapper=_plan_then_ground_wrapper,
        description="Encourage internal high-level planning before grounding into JSON actions.",
    ),
    "state_checklist_plan": PromptVariant(
        label="state_checklist_plan",
        eval_type="action_sequencing",
        system_prompt=_ACTION_STATE_CHECKLIST_SYSTEM,
        user_wrapper=_state_checklist_plan_wrapper,
        description=(
            "Prompt-only improvement: plan internally, then run a final-state and "
            "precondition checklist before outputting the compact JSON action sequence."
        ),
    ),
    "goal_conditioned_scaffold": PromptVariant(
        label="goal_conditioned_scaffold",
        eval_type="action_sequencing",
        system_prompt=_ACTION_GOAL_SCAFFOLD_SYSTEM,
        user_wrapper=_goal_conditioned_scaffold_wrapper,
        description=(
            "Prompt-only improvement inspired by programmatic planning: map formal goals "
            "to minimal action skeletons before emitting the compact JSON sequence."
        ),
    ),
    "bidirectional_causal_planning": PromptVariant(
        label="bidirectional_causal_planning",
        eval_type="action_sequencing",
        system_prompt=_ACTION_BIDIRECTIONAL_CAUSAL_SYSTEM,
        user_wrapper=_bidirectional_causal_planning_wrapper,
        description=(
            "Prompt-only improvement: reason backward from final goals to causal actions, "
            "then reason forward to order and ground them into an executable sequence."
        ),
    ),
    "kg_rag_plan_then_ground": PromptVariant(
        label="kg_rag_plan_then_ground",
        eval_type="action_sequencing",
        system_prompt=_ACTION_KG_RAG_PLAN_SYSTEM,
        user_wrapper=_kg_rag_plan_then_ground_wrapper,
        description=(
            "Knowledge-grounded planning: inject compact scene-graph RAG and task-relevant "
            "precondition KG rules before plan-then-ground generation."
        ),
    ),
    "sg_rag": PromptVariant(
        label="sg_rag",
        eval_type="action_sequencing",
        system_prompt=_ACTION_FORMAT_SYSTEM,
        user_wrapper=_sg_rag_action_wrapper,
        description=(
            "Scene-Graph RAG: inject a task-relevant object subgraph "
            "(ids, properties, states, relations) into the user prompt."
        ),
    ),
    "pc_kg_self_check": PromptVariant(
        label="pc_kg_self_check",
        eval_type="action_sequencing",
        system_prompt=_ACTION_SELF_CHECK_SYSTEM,
        user_wrapper=_identity,
        requires_second_pass=True,
        critique_system_prompt=_ACTION_FORMAT_SYSTEM,
        critique_user_wrapper=_pc_kg_critique_user,
        description=(
            "Two-pass with precondition knowledge graph: verify draft against "
            "PC-KG, feed structured violations into critique, then fix."
        ),
    ),
    "sg_rag_pc_kg": PromptVariant(
        label="sg_rag_pc_kg",
        eval_type="action_sequencing",
        system_prompt=_ACTION_SELF_CHECK_SYSTEM,
        user_wrapper=_sg_rag_action_wrapper,
        requires_second_pass=True,
        critique_system_prompt=_ACTION_FORMAT_SYSTEM,
        critique_user_wrapper=_pc_kg_critique_user,
        description=(
            "Combined Knowledge-Grounded Recovery: SG-RAG injection in pass 1, "
            "PC-KG-driven critique in pass 2."
        ),
    ),
    "selective_recovery": PromptVariant(
        label="selective_recovery",
        eval_type="action_sequencing",
        system_prompt=_ACTION_PLAN_SYSTEM,
        user_wrapper=_selective_recovery_wrapper,
        requires_second_pass=True,
        critique_system_prompt=_ACTION_SELECTIVE_RECOVERY_SYSTEM,
        critique_user_wrapper=_selective_recovery_critique_user,
        description=(
            "Failure-aware selective recovery: plan-then-ground generation followed by "
            "deterministic local insertion of high-confidence missing prerequisites."
        ),
    ),
}


GOAL_VARIANTS: Dict[str, PromptVariant] = {
    "baseline": PromptVariant(
        label="baseline",
        eval_type="goal_interpretation",
        system_prompt=_GOAL_BASELINE_SYSTEM,
        user_wrapper=_identity,
        description="Direct goal interpretation with a JSON-only system prompt.",
    ),
    "schema_constrained": PromptVariant(
        label="schema_constrained",
        eval_type="goal_interpretation",
        system_prompt=_GOAL_SCHEMA_SYSTEM,
        user_wrapper=_identity,
        description="Strict schema for node, edge and action goals to lift F1.",
    ),
    "few_shot": PromptVariant(
        label="few_shot",
        eval_type="goal_interpretation",
        system_prompt=_GOAL_FEW_SHOT_SYSTEM,
        user_wrapper=_few_shot_goal_wrapper,
        description="Schema constraints plus a worked example from EAI documentation.",
    ),
    "decompose_then_merge": PromptVariant(
        label="decompose_then_merge",
        eval_type="goal_interpretation",
        system_prompt=_GOAL_DECOMPOSE_SYSTEM,
        user_wrapper=_decompose_goal_wrapper,
        description="Decompose into states, relations, actions, then merge before output.",
    ),
}


def list_variants(eval_type: str) -> List[str]:
    if eval_type == "action_sequencing":
        return list(ACTION_VARIANTS.keys())
    if eval_type == "goal_interpretation":
        return list(GOAL_VARIANTS.keys())
    raise ValueError(f"Unknown eval_type: {eval_type}")


def get_variant(eval_type: str, name: str) -> PromptVariant:
    if eval_type == "action_sequencing":
        if name not in ACTION_VARIANTS:
            raise KeyError(f"Unknown action variant '{name}'. Known: {list_variants(eval_type)}")
        return ACTION_VARIANTS[name]
    if eval_type == "goal_interpretation":
        if name not in GOAL_VARIANTS:
            raise KeyError(f"Unknown goal variant '{name}'. Known: {list_variants(eval_type)}")
        return GOAL_VARIANTS[name]
    raise ValueError(f"Unknown eval_type: {eval_type}")


__all__ = [
    "PromptVariant",
    "ACTION_VARIANTS",
    "GOAL_VARIANTS",
    "list_variants",
    "get_variant",
]
