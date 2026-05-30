#!/usr/bin/env python3
"""Precondition Knowledge Graph (PC-KG) + symbolic verifier for VirtualHome.

Builds a small rule base describing the preconditions of each VirtualHome
action (derived from ``scripts.py`` in the EAI source tree, with manual
amendments for the walk/open/grab semantics documented in
``virtualhome.pddl``) and runs a miniature symbolic simulation over a
concatenated-JSON action sequence to emit structured ``Violation``s.

The verifier does **not** try to be a full VirtualHome simulator. It only
checks the set of preconditions that show up as dominant failure modes in
the thesis (``missing_step``, ``wrong_order``, ``hallucination`` /
``UNKNOWN_ID``, ``affordance_error``). Anything beyond that is left to
the real EAI evaluator.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


class ViolationCode(str, Enum):
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    ARITY_MISMATCH = "ARITY_MISMATCH"
    UNKNOWN_ID = "UNKNOWN_ID"
    NAME_ID_MISMATCH = "NAME_ID_MISMATCH"
    MISSING_WALK = "MISSING_WALK"
    MISSING_OPEN = "MISSING_OPEN"
    NOT_OPENABLE = "NOT_OPENABLE"
    ALREADY_OPEN = "ALREADY_OPEN"
    ALREADY_CLOSED = "ALREADY_CLOSED"
    NOT_GRABBABLE = "NOT_GRABBABLE"
    NOT_HELD = "NOT_HELD"
    HAND_FULL = "HAND_FULL"
    NO_SWITCH = "NO_SWITCH"
    NO_PLUG = "NO_PLUG"
    NOT_SITTABLE = "NOT_SITTABLE"
    NOT_LIEABLE = "NOT_LIEABLE"
    PARSE_ERROR = "PARSE_ERROR"


@dataclass(frozen=True)
class Violation:
    step_index: int
    action: str
    code: ViolationCode
    detail: str


@dataclass
class _WorldState:
    """Tiny symbolic world state driven by the action sequence."""

    visited: set = field(default_factory=set)       # object ids the char has walked to
    opened: set = field(default_factory=set)         # currently open containers
    holding: set = field(default_factory=set)        # object ids being held (max 2)
    plugged_in: set = field(default_factory=set)
    switched_on: set = field(default_factory=set)
    sitting_on: Optional[int] = None


@dataclass(frozen=True)
class ActionRule:
    name: str
    arity: int                         # expected (name,id) pairs: 1 or 2 (STANDUP = 0)
    required_props_arg1: Tuple[str, ...] = ()   # arg1 must have ALL of these
    required_props_arg2: Tuple[str, ...] = ()
    requires_walk_to_arg1: bool = True
    requires_walk_to_arg2: bool = False
    requires_hold_arg1: bool = False
    requires_open_arg2: bool = False


# ------------------------------------------------------------------ rule base
# Keeps the minimum set mentioned in doc.md §3.2.
RULES: Dict[str, ActionRule] = {
    "WALK": ActionRule(
        name="WALK", arity=1,
        requires_walk_to_arg1=False,
    ),
    "RUN": ActionRule(
        name="RUN", arity=1, requires_walk_to_arg1=False,
    ),
    "TURNTO": ActionRule(
        name="TURNTO", arity=1, requires_walk_to_arg1=False,
    ),
    "FIND": ActionRule(
        name="FIND", arity=1, requires_walk_to_arg1=False,
    ),
    "STANDUP": ActionRule(name="STANDUP", arity=0, requires_walk_to_arg1=False),
    "GRAB": ActionRule(
        name="GRAB", arity=1,
        required_props_arg1=("GRABBABLE",),
        requires_walk_to_arg1=True,
    ),
    "PUTBACK": ActionRule(
        name="PUTBACK", arity=2,
        requires_walk_to_arg1=False,
        requires_walk_to_arg2=True,
        requires_hold_arg1=True,
    ),
    "PUTIN": ActionRule(
        name="PUTIN", arity=2,
        requires_walk_to_arg1=False,
        requires_walk_to_arg2=True,
        requires_hold_arg1=True,
        requires_open_arg2=True,
    ),
    "OPEN": ActionRule(
        name="OPEN", arity=1,
        required_props_arg1=("CAN_OPEN",),
        requires_walk_to_arg1=True,
    ),
    "CLOSE": ActionRule(
        name="CLOSE", arity=1,
        required_props_arg1=("CAN_OPEN",),
        requires_walk_to_arg1=True,
    ),
    "SWITCHON": ActionRule(
        name="SWITCHON", arity=1,
        required_props_arg1=("HAS_SWITCH",),
        requires_walk_to_arg1=True,
    ),
    "SWITCHOFF": ActionRule(
        name="SWITCHOFF", arity=1,
        required_props_arg1=("HAS_SWITCH",),
        requires_walk_to_arg1=True,
    ),
    "PLUGIN": ActionRule(
        name="PLUGIN", arity=1,
        required_props_arg1=("HAS_PLUG",),
        requires_walk_to_arg1=True,
    ),
    "PLUGOUT": ActionRule(
        name="PLUGOUT", arity=1,
        required_props_arg1=("HAS_PLUG",),
        requires_walk_to_arg1=True,
    ),
    "SIT": ActionRule(
        name="SIT", arity=1,
        required_props_arg1=("SITTABLE",),
        requires_walk_to_arg1=True,
    ),
    "LIE": ActionRule(
        name="LIE", arity=1,
        required_props_arg1=("LIEABLE",),
        requires_walk_to_arg1=True,
    ),
    "DRINK": ActionRule(
        name="DRINK", arity=1,
        requires_hold_arg1=True,
        requires_walk_to_arg1=False,
    ),
    "EAT": ActionRule(
        name="EAT", arity=1,
        requires_hold_arg1=True,
        requires_walk_to_arg1=False,
    ),
    "READ": ActionRule(
        name="READ", arity=1,
        requires_hold_arg1=True,
        requires_walk_to_arg1=False,
    ),
    "WATCH": ActionRule(
        name="WATCH", arity=1,
        requires_walk_to_arg1=False,
    ),
    "WAKEUP": ActionRule(name="WAKEUP", arity=0, requires_walk_to_arg1=False),
    "SLEEP": ActionRule(name="SLEEP", arity=0, requires_walk_to_arg1=False),
}


# -------------------------------------------------------- concatenated-JSON parser
_ACTION_JSON = re.compile(r"\{\s*\"([A-Z_]+)\"\s*:\s*(\[[^\]]*\])\s*\}")


def parse_action_sequence(text: str) -> Tuple[List[Tuple[str, List[str]]], List[Violation]]:
    """Parse the compact ``{"WALK":["floor_lamp","1000"]}...`` format.

    Returns ``(steps, parse_violations)`` where ``parse_violations`` is
    non-empty when some snippet could not be decoded.
    """
    steps: List[Tuple[str, List[str]]] = []
    violations: List[Violation] = []
    if not text:
        return steps, violations
    matches = list(_ACTION_JSON.finditer(text))
    if not matches:
        violations.append(
            Violation(
                step_index=-1,
                action="",
                code=ViolationCode.PARSE_ERROR,
                detail="no {\"ACTION\":[...]} blocks found",
            )
        )
        return steps, violations
    for i, m in enumerate(matches):
        action = m.group(1).upper()
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError as exc:
            violations.append(
                Violation(
                    step_index=i,
                    action=action,
                    code=ViolationCode.PARSE_ERROR,
                    detail=f"invalid args json: {exc}",
                )
            )
            continue
        if not isinstance(args, list):
            violations.append(
                Violation(
                    step_index=i,
                    action=action,
                    code=ViolationCode.PARSE_ERROR,
                    detail="args not a list",
                )
            )
            continue
        steps.append((action, [str(a) for a in args]))
    return steps, violations


# --------------------------------------------------------------------- verifier
class PreconditionKG:
    """Tiny precondition KG + symbolic verifier for VirtualHome action sequences."""

    _SHARED: Optional["PreconditionKG"] = None

    def __init__(self, rules: Dict[str, ActionRule] = RULES) -> None:
        self.rules = dict(rules)

    @classmethod
    def shared(cls) -> "PreconditionKG":
        if cls._SHARED is None:
            import os
            backend = os.environ.get("KB_BACKEND", "default").lower()
            if backend == "persistent":
                try:
                    try:
                        from analysis.kb.persistent_kg import PersistentPreconditionKG
                    except ModuleNotFoundError:
                        from kb.persistent_kg import PersistentPreconditionKG  # type: ignore[no-redef]
                    cls._SHARED = PersistentPreconditionKG.shared()  # type: ignore[assignment]
                except Exception as exc:
                    log.warning(
                        "precondition_kg: persistent backend unavailable (%s); "
                        "falling back to in-code rules", exc,
                    )
                    cls._SHARED = cls()
            else:
                cls._SHARED = cls()
        return cls._SHARED

    # ----- core API
    def verify(
        self,
        action_sequence: str,
        scene_objects: Dict[int, Any],
    ) -> List[Violation]:
        """Return a list of :class:`Violation` for the draft sequence."""
        steps, parse_viols = parse_action_sequence(action_sequence)
        if parse_viols:
            return parse_viols
        state = _WorldState()
        out: List[Violation] = []

        for step_idx, (action, args) in enumerate(steps):
            rule = self.rules.get(action)
            if rule is None:
                out.append(
                    Violation(step_idx, action, ViolationCode.UNKNOWN_ACTION,
                              f"action {action} not in KG")
                )
                continue

            # arity check: args come in (name,id) pairs; STANDUP/SLEEP have 0.
            expected_len = rule.arity * 2
            if len(args) != expected_len:
                out.append(
                    Violation(step_idx, action, ViolationCode.ARITY_MISMATCH,
                              f"{action} expects {expected_len} args, got {len(args)}")
                )
                continue

            arg_ids = _extract_ids(args, rule.arity)
            # id existence + name-id consistency
            for pos, (arg_name, arg_id) in enumerate(arg_ids, start=1):
                if arg_id is None:
                    out.append(
                        Violation(step_idx, action, ViolationCode.UNKNOWN_ID,
                                  f"arg{pos} id unparseable: {arg_name}")
                    )
                    continue
                node = scene_objects.get(arg_id)
                if node is None:
                    out.append(
                        Violation(step_idx, action, ViolationCode.UNKNOWN_ID,
                                  f"arg{pos} id={arg_id} not in scene")
                    )
                    continue
                nc = getattr(node, "class_name", None) or (
                    node.get("class_name") if isinstance(node, dict) else None
                )
                if nc and arg_name and nc != arg_name:
                    out.append(
                        Violation(step_idx, action, ViolationCode.NAME_ID_MISMATCH,
                                  f"arg{pos} class {arg_name!r} != scene class {nc!r}")
                    )

            # property checks
            if rule.arity >= 1 and arg_ids:
                _check_props(out, step_idx, action, arg_ids[0], rule.required_props_arg1,
                             scene_objects, arg_slot=1)
            if rule.arity >= 2 and len(arg_ids) >= 2:
                _check_props(out, step_idx, action, arg_ids[1], rule.required_props_arg2,
                             scene_objects, arg_slot=2)

            # WALK-before-op
            if rule.requires_walk_to_arg1 and arg_ids:
                _id = arg_ids[0][1]
                if _id is not None and _id not in state.visited:
                    out.append(
                        Violation(step_idx, action, ViolationCode.MISSING_WALK,
                                  f"no WALK to {arg_ids[0][0]}({_id}) before {action}")
                    )
            if rule.requires_walk_to_arg2 and len(arg_ids) >= 2:
                _id = arg_ids[1][1]
                if _id is not None and _id not in state.visited:
                    out.append(
                        Violation(step_idx, action, ViolationCode.MISSING_WALK,
                                  f"no WALK to {arg_ids[1][0]}({_id}) before {action}")
                    )

            # container-open-before-putin
            if rule.requires_open_arg2 and len(arg_ids) >= 2:
                _id = arg_ids[1][1]
                if _id is not None and _id not in state.opened:
                    out.append(
                        Violation(step_idx, action, ViolationCode.MISSING_OPEN,
                                  f"{arg_ids[1][0]}({_id}) not opened before {action}")
                    )

            # hold-before-use
            if rule.requires_hold_arg1 and arg_ids:
                _id = arg_ids[0][1]
                if _id is not None and _id not in state.holding:
                    out.append(
                        Violation(step_idx, action, ViolationCode.NOT_HELD,
                                  f"{arg_ids[0][0]}({_id}) not held before {action}")
                    )

            # hand-full check for GRAB
            if action == "GRAB" and arg_ids and arg_ids[0][1] is not None:
                if len(state.holding) >= 2 and arg_ids[0][1] not in state.holding:
                    out.append(
                        Violation(step_idx, action, ViolationCode.HAND_FULL,
                                  "both hands already holding objects")
                    )

            # OPEN/CLOSE state checks
            if action == "OPEN" and arg_ids and arg_ids[0][1] in state.opened:
                out.append(
                    Violation(step_idx, action, ViolationCode.ALREADY_OPEN,
                              f"{arg_ids[0][0]}({arg_ids[0][1]}) already open")
                )
            if action == "CLOSE" and arg_ids and arg_ids[0][1] is not None \
                    and arg_ids[0][1] not in state.opened:
                out.append(
                    Violation(step_idx, action, ViolationCode.ALREADY_CLOSED,
                              f"{arg_ids[0][0]}({arg_ids[0][1]}) not open to close")
                )

            # apply effects
            _apply_effects(state, action, arg_ids)

        return out

    @staticmethod
    def summarise(violations: Sequence[Violation]) -> str:
        if not violations:
            return "[KG Verifier] No precondition violations detected."
        lines = [f"[KG Verifier] {len(violations)} precondition violation(s):"]
        for v in violations:
            lines.append(f"- step {v.step_index}: {v.action} -> {v.code.value} ({v.detail})")
        lines.append("Fix ONLY these violations; do not rewrite correct steps.")
        lines.append("[/KG Verifier]")
        return "\n".join(lines)

    @staticmethod
    def histogram(violations: Iterable[Violation]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in violations:
            out[v.code.value] = out.get(v.code.value, 0) + 1
        return out


# ------------------------------------------------------------------- helpers
def _extract_ids(args: List[str], arity: int) -> List[Tuple[str, Optional[int]]]:
    """Return list of (class_name, id) pairs of length ``arity``."""
    out: List[Tuple[str, Optional[int]]] = []
    for i in range(arity):
        name = args[2 * i] if 2 * i < len(args) else ""
        raw_id = args[2 * i + 1] if 2 * i + 1 < len(args) else ""
        try:
            obj_id = int(str(raw_id).strip())
        except (TypeError, ValueError):
            obj_id = None
        out.append((name, obj_id))
    return out


def _check_props(
    out: List[Violation],
    step_idx: int,
    action: str,
    arg: Tuple[str, Optional[int]],
    required: Tuple[str, ...],
    scene_objects: Dict[int, Any],
    arg_slot: int,
) -> None:
    if not required or arg[1] is None:
        return
    node = scene_objects.get(arg[1])
    if node is None:
        return
    props = getattr(node, "properties", None)
    if props is None and isinstance(node, dict):
        props = node.get("properties") or []
    props = set(props or [])
    for req in required:
        if req not in props:
            code_map = {
                "GRABBABLE": ViolationCode.NOT_GRABBABLE,
                "CAN_OPEN": ViolationCode.NOT_OPENABLE,
                "HAS_SWITCH": ViolationCode.NO_SWITCH,
                "HAS_PLUG": ViolationCode.NO_PLUG,
                "SITTABLE": ViolationCode.NOT_SITTABLE,
                "LIEABLE": ViolationCode.NOT_LIEABLE,
            }
            code = code_map.get(req, ViolationCode.NOT_OPENABLE)
            out.append(
                Violation(step_idx, action, code,
                          f"arg{arg_slot} {arg[0]}({arg[1]}) missing property {req}")
            )


def _apply_effects(
    state: _WorldState,
    action: str,
    arg_ids: List[Tuple[str, Optional[int]]],
) -> None:
    def _id(i: int) -> Optional[int]:
        return arg_ids[i][1] if i < len(arg_ids) else None

    if action in {"WALK", "RUN", "TURNTO", "FIND"} and _id(0) is not None:
        state.visited.add(_id(0))
    elif action == "GRAB" and _id(0) is not None:
        state.holding.add(_id(0))
        state.visited.add(_id(0))
    elif action in {"PUTBACK", "PUTIN"} and _id(0) is not None:
        state.holding.discard(_id(0))
        if _id(1) is not None:
            state.visited.add(_id(1))
    elif action == "OPEN" and _id(0) is not None:
        state.opened.add(_id(0))
        state.visited.add(_id(0))
    elif action == "CLOSE" and _id(0) is not None:
        state.opened.discard(_id(0))
    elif action == "SWITCHON" and _id(0) is not None:
        state.switched_on.add(_id(0))
        state.visited.add(_id(0))
    elif action == "SWITCHOFF" and _id(0) is not None:
        state.switched_on.discard(_id(0))
    elif action == "PLUGIN" and _id(0) is not None:
        state.plugged_in.add(_id(0))
        state.visited.add(_id(0))
    elif action == "PLUGOUT" and _id(0) is not None:
        state.plugged_in.discard(_id(0))
    elif action == "SIT" and _id(0) is not None:
        state.sitting_on = _id(0)
        state.visited.add(_id(0))
    elif action == "STANDUP":
        state.sitting_on = None


def verify_sequence(
    action_sequence: str,
    scene_objects: Dict[int, Any],
) -> List[Violation]:
    return PreconditionKG.shared().verify(action_sequence, scene_objects)


__all__ = [
    "Violation",
    "ViolationCode",
    "ActionRule",
    "PreconditionKG",
    "RULES",
    "parse_action_sequence",
    "verify_sequence",
]
