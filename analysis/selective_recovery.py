#!/usr/bin/env python3
"""Conservative selective recovery for VirtualHome action sequences.

This module implements a deterministic, local post-processor for the thesis
improvement pipeline. It is intentionally more conservative than the failed
PC-KG rewrite loop: it never asks the model to rewrite a full program, and it
only inserts high-confidence prerequisite actions that are directly licensed by
existing steps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from precondition_kg import RULES, parse_action_sequence
except ModuleNotFoundError:  # pragma: no cover - package import path in unit tests
    from analysis.precondition_kg import RULES, parse_action_sequence


_INSERTABLE_ACTIONS = {"WALK", "GRAB", "OPEN"}
_REQUIRES_HELD_ARG1 = {"DRINK", "EAT", "READ", "PUTIN", "PUTBACK"}
_UNSAFE_TO_REPAIR = {"UNKNOWN_ACTION", "ARITY_MISMATCH", "UNKNOWN_ID", "NAME_ID_MISMATCH", "PARSE_ERROR"}


@dataclass(frozen=True)
class RecoveryReport:
    changed: bool
    inserted_actions: int
    skipped_reason: Optional[str] = None
    inserted_steps: Tuple[str, ...] = ()


def recover_sequence(
    action_sequence: str,
    scene_objects: Dict[int, Any],
    max_insertions: int = 4,
) -> Tuple[str, RecoveryReport]:
    """Return a locally repaired action sequence plus an audit report.

    The repair policy is deliberately narrow:
    - malformed / unknown-id / name-id-mismatch programs are returned unchanged;
    - missing WALK, GRAB, and OPEN prerequisites may be inserted;
    - no original step is deleted, reordered, or rewritten;
    - repair is abandoned if it would exceed ``max_insertions``.
    """
    steps, parse_violations = parse_action_sequence(action_sequence)
    if parse_violations:
        return action_sequence, RecoveryReport(False, 0, "parse_error")
    if not steps:
        return action_sequence, RecoveryReport(False, 0, "empty_sequence")

    unsafe_reason = _first_unsafe_reason(steps, scene_objects)
    if unsafe_reason:
        return action_sequence, RecoveryReport(False, 0, unsafe_reason)

    repaired: List[Tuple[str, List[str]]] = []
    inserted: List[Tuple[str, List[str]]] = []
    visited: set[int] = set()
    holding: set[int] = set()
    opened: set[int] = set()

    for action, args in steps:
        prerequisites = _required_insertions(action, args, scene_objects, visited, holding, opened)
        if len(inserted) + len(prerequisites) > max_insertions:
            return action_sequence, RecoveryReport(False, 0, "max_insertions_exceeded")
        for pre_action, pre_args in prerequisites:
            repaired.append((pre_action, pre_args))
            inserted.append((pre_action, pre_args))
            _apply_local_effect(pre_action, pre_args, visited, holding, opened)
        repaired.append((action, args))
        _apply_local_effect(action, args, visited, holding, opened)

    if not inserted:
        return action_sequence, RecoveryReport(False, 0, "no_repair_needed")
    return _format_steps(repaired), RecoveryReport(
        changed=True,
        inserted_actions=len(inserted),
        inserted_steps=tuple(_format_step(action, args) for action, args in inserted),
    )


def _first_unsafe_reason(steps: Sequence[Tuple[str, List[str]]], scene_objects: Dict[int, Any]) -> Optional[str]:
    for action, args in steps:
        rule = RULES.get(action)
        if rule is None:
            return "unsafe_unknown_action"
        if len(args) != rule.arity * 2:
            return "unsafe_arity_mismatch"
        for name, obj_id in _pairs(args):
            if obj_id is None or obj_id not in scene_objects:
                return "unsafe_unknown_id"
            class_name = _class_name(scene_objects[obj_id])
            if class_name and name and class_name != name:
                return "unsafe_name_id_mismatch"
    return None


def _required_insertions(
    action: str,
    args: List[str],
    scene_objects: Dict[int, Any],
    visited: set[int],
    holding: set[int],
    opened: set[int],
) -> List[Tuple[str, List[str]]]:
    rule = RULES[action]
    arg_pairs = _pairs(args)
    out: List[Tuple[str, List[str]]] = []

    def ensure_walk(name: str, obj_id: int) -> None:
        if obj_id not in visited:
            out.append(("WALK", [name, str(obj_id)]))
            visited.add(obj_id)

    def ensure_grab(name: str, obj_id: int) -> None:
        if obj_id not in holding and _has_property(scene_objects[obj_id], "GRABBABLE"):
            ensure_walk(name, obj_id)
            out.append(("GRAB", [name, str(obj_id)]))
            holding.add(obj_id)

    if rule.requires_walk_to_arg1 and arg_pairs:
        name, obj_id = arg_pairs[0]
        if obj_id is not None:
            ensure_walk(name, obj_id)
    if rule.requires_walk_to_arg2 and len(arg_pairs) >= 2:
        name, obj_id = arg_pairs[1]
        if obj_id is not None:
            ensure_walk(name, obj_id)
    if action in _REQUIRES_HELD_ARG1 and arg_pairs:
        name, obj_id = arg_pairs[0]
        if obj_id is not None:
            ensure_grab(name, obj_id)
    if rule.requires_open_arg2 and len(arg_pairs) >= 2:
        name, obj_id = arg_pairs[1]
        if obj_id is not None and obj_id not in opened and _has_property(scene_objects[obj_id], "CAN_OPEN"):
            ensure_walk(name, obj_id)
            out.append(("OPEN", [name, str(obj_id)]))
            opened.add(obj_id)
    return out


def _pairs(args: List[str]) -> List[Tuple[str, Optional[int]]]:
    out: List[Tuple[str, Optional[int]]] = []
    for index in range(0, len(args), 2):
        name = args[index]
        try:
            obj_id = int(args[index + 1])
        except (IndexError, TypeError, ValueError):
            obj_id = None
        out.append((name, obj_id))
    return out


def _class_name(node: Any) -> Optional[str]:
    if isinstance(node, dict):
        return node.get("class_name")
    return getattr(node, "class_name", None)


def _has_property(node: Any, prop: str) -> bool:
    props = node.get("properties", []) if isinstance(node, dict) else getattr(node, "properties", [])
    return prop in set(props or [])


def _apply_local_effect(
    action: str,
    args: List[str],
    visited: set[int],
    holding: set[int],
    opened: set[int],
) -> None:
    pairs = _pairs(args)
    first_id = pairs[0][1] if pairs else None
    second_id = pairs[1][1] if len(pairs) > 1 else None
    if action in {"WALK", "RUN", "TURNTO", "FIND"} and first_id is not None:
        visited.add(first_id)
    elif action == "GRAB" and first_id is not None:
        visited.add(first_id)
        holding.add(first_id)
    elif action in {"PUTBACK", "PUTIN"}:
        if first_id is not None:
            holding.discard(first_id)
        if second_id is not None:
            visited.add(second_id)
    elif action == "OPEN" and first_id is not None:
        visited.add(first_id)
        opened.add(first_id)
    elif action == "CLOSE" and first_id is not None:
        opened.discard(first_id)
    elif action in {"SWITCHON", "PLUGIN", "SIT", "LIE"} and first_id is not None:
        visited.add(first_id)


def _format_step(action: str, args: List[str]) -> str:
    return json.dumps({action: args}, separators=(",", ":"), ensure_ascii=False)


def _format_steps(steps: Sequence[Tuple[str, List[str]]]) -> str:
    return "".join(_format_step(action, args) for action, args in steps)


__all__ = ["RecoveryReport", "recover_sequence"]
