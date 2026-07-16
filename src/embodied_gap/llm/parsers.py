from __future__ import annotations

import ast
import json
import re
from typing import Any


def parse_action_list(text: str) -> tuple[str, ...]:
    payload = extract_structured_payload(text)
    if not payload:
        return ()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(payload)
    if not isinstance(parsed, list):
        raise ValueError("Planner output must be a list of action strings.")
    return tuple(normalize_action_item(item) for item in parsed)


def extract_structured_payload(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    fenced = re.search(r"```(?:json|python)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped

    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end > start:
        return stripped[start : end + 1].strip()

    return stripped


def normalize_action_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        if item and all(isinstance(part, str) for part in item):
            action = item[0]
            args = item[1:]
            if not action:
                raise ValueError("Action array must start with a non-empty action string.")
            return f"{action}({', '.join(args)})"
        if (
            len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], list)
            and all(isinstance(arg, str) for arg in item[1])
        ):
            action = item[0]
            if not action:
                raise ValueError("Action array must start with a non-empty action string.")
            return f"{action}({', '.join(item[1])})"
    if isinstance(item, dict):
        action = item.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError("Action dictionary must contain a non-empty action string.")
        if "args" in item:
            args = item["args"]
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise ValueError("Action dictionary args must be a list of strings.")
            return f"{action}({', '.join(args)})"
        if "object" in item:
            obj = item["object"]
            if not isinstance(obj, str):
                raise ValueError("Action dictionary object must be a string.")
            args = [part.strip() for part in obj.split(",") if part.strip()]
            return f"{action}({', '.join(args)})"
    raise ValueError("Planner output list items must be strings, action arrays, or action dictionaries.")
