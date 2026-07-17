from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


OFFICIAL_DATASETS = ("virtualhome", "behavior")
OFFICIAL_MODULES = (
    "goal_interpretation",
    "subgoal_decomposition",
    "action_sequencing",
    "transition_modeling",
)


def _issue(code: str, message: str, *, identifier: str | None = None) -> dict[str, str]:
    issue = {"code": code, "message": message}
    if identifier is not None:
        issue["identifier"] = identifier
    return issue


def _parse_llm_output(raw: Any, identifier: str) -> tuple[Any | None, list[dict[str, str]]]:
    if not isinstance(raw, str):
        return None, [_issue("llm_output_not_string", "llm_output must be a JSON string.", identifier=identifier)]
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        return json.loads(cleaned), []
    except json.JSONDecodeError as exc:
        return None, [
            _issue(
                "llm_output_invalid_json",
                f"llm_output is not valid JSON: {exc.msg}.",
                identifier=identifier,
            )
        ]


def validate_action_sequencing_records(
    records: Any,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Validate the input contract used by the pinned official EAI evaluator.

    VirtualHome validation intentionally follows the code at the pinned EAI
    commit: every object argument is represented by an object-name/object-id
    pair. This differs from prose in the generated prompt that says to omit IDs.
    """

    dataset = dataset.lower()
    if dataset not in OFFICIAL_DATASETS:
        raise ValueError(f"Unsupported EAI dataset: {dataset}")

    issues: list[dict[str, str]] = []
    identifiers: set[str] = set()
    parsed_count = 0
    nonempty_plan_count = 0

    if not isinstance(records, list):
        issues.append(_issue("root_not_list", "Official model output must be a JSON list."))
        records = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            issues.append(_issue("record_not_object", f"Record {index} must be an object."))
            continue
        identifier = record.get("identifier")
        if not isinstance(identifier, str) or not identifier:
            issues.append(_issue("invalid_identifier", f"Record {index} has no string identifier."))
            identifier = f"record-{index}"
        elif identifier in identifiers:
            issues.append(_issue("duplicate_identifier", "Identifier is duplicated.", identifier=identifier))
        identifiers.add(identifier)

        parsed, parse_issues = _parse_llm_output(record.get("llm_output"), identifier)
        issues.extend(parse_issues)
        if parse_issues:
            continue
        parsed_count += 1
        if not isinstance(parsed, list) or not parsed:
            issues.append(_issue("empty_or_nonlist_plan", "Action sequence must be a non-empty JSON list.", identifier=identifier))
            continue
        nonempty_plan_count += 1

        for step_index, step in enumerate(parsed):
            if not isinstance(step, dict):
                issues.append(_issue("step_not_object", f"Step {step_index} must be an object.", identifier=identifier))
                continue
            if dataset == "virtualhome":
                if len(step) != 1:
                    issues.append(_issue("virtualhome_step_key_count", f"Step {step_index} must contain exactly one action.", identifier=identifier))
                    continue
                action, arguments = next(iter(step.items()))
                if not isinstance(action, str) or not action:
                    issues.append(_issue("invalid_action", f"Step {step_index} has no action name.", identifier=identifier))
                if not isinstance(arguments, list) or len(arguments) % 2 != 0:
                    issues.append(
                        _issue(
                            "virtualhome_name_id_pairs_required",
                            f"Step {step_index} arguments must be [name, id] pairs at the pinned official commit.",
                            identifier=identifier,
                        )
                    )
            else:
                if set(step) != {"action", "object"}:
                    issues.append(
                        _issue(
                            "behavior_step_shape",
                            f"Step {step_index} must contain exactly action and object.",
                            identifier=identifier,
                        )
                    )
                elif not isinstance(step["action"], str) or not isinstance(step["object"], str):
                    issues.append(_issue("behavior_step_types", f"Step {step_index} action/object must be strings.", identifier=identifier))

    return {
        "dataset": dataset,
        "record_count": len(records),
        "parsed_count": parsed_count,
        "nonempty_plan_count": nonempty_plan_count,
        "valid": not issues,
        "issues": issues,
    }


def validate_action_sequencing_file(path: str | Path, *, dataset: str) -> dict[str, Any]:
    path = Path(path)
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "dataset": dataset,
            "path": str(path),
            "record_count": 0,
            "parsed_count": 0,
            "nonempty_plan_count": 0,
            "valid": False,
            "issues": [_issue("file_unreadable", str(exc))],
        }
    result = validate_action_sequencing_records(records, dataset=dataset)
    result["path"] = str(path)
    return result


def inspect_official_response_tree(
    response_root: str | Path,
    *,
    external_root: str | Path = "external/embodied-agent-interface",
) -> dict[str, Any]:
    response_root = Path(response_root)
    external_root = Path(external_root)
    slots: list[dict[str, Any]] = []
    for dataset in OFFICIAL_DATASETS:
        for module in OFFICIAL_MODULES:
            directory = response_root / dataset / module
            files = sorted(directory.glob("*_outputs.json")) if directory.is_dir() else []
            slot: dict[str, Any] = {
                "dataset": dataset,
                "module": module,
                "directory": str(directory),
                "files": [str(path) for path in files],
                "present": bool(files),
            }
            if module == "action_sequencing":
                slot["validations"] = [
                    validate_action_sequencing_file(path, dataset=dataset) for path in files
                ]
            slots.append(slot)

    present_count = sum(bool(slot["present"]) for slot in slots)
    validations = [validation for slot in slots for validation in slot.get("validations", [])]
    action_shapes_valid = bool(validations) and all(item["valid"] for item in validations)
    behavior_runtime_marker = external_root / "src" / "behavior_eval"
    virtualhome_runtime_marker = external_root / "src" / "virtualhome_eval"
    igibson_root = external_root.parent / "iGibson"
    igibson_dataset_root = Path(
        os.environ.get(
            "IGIBSON_DATASET_PATH",
            str(igibson_root / "igibson" / "data" / "ig_dataset"),
        )
    ).expanduser()
    igibson_source_present = (igibson_root / "igibson").is_dir()
    igibson_dataset_present = all(
        (igibson_dataset_root / component).is_dir()
        for component in ("scenes", "objects")
    )
    official_runtime_ready = (
        virtualhome_runtime_marker.is_dir()
        and behavior_runtime_marker.is_dir()
        and igibson_source_present
        and igibson_dataset_present
    )
    structurally_ready = present_count == len(slots) and action_shapes_valid
    return {
        "protocol": "official-eai-four-modules-two-datasets",
        "response_root": str(response_root),
        "external_root": str(external_root),
        "required_slot_count": len(OFFICIAL_DATASETS) * len(OFFICIAL_MODULES),
        "present_slot_count": present_count,
        "all_slots_present": present_count == len(slots),
        "action_sequencing_shapes_valid": action_shapes_valid,
        "structurally_ready": structurally_ready,
        "official_runtime_ready": official_runtime_ready,
        "submission_ready": structurally_ready and official_runtime_ready,
        "runtime_sources": {
            "virtualhome_present": virtualhome_runtime_marker.is_dir(),
            "behavior_present": behavior_runtime_marker.is_dir(),
            "igibson_source_present": igibson_source_present,
            "igibson_dataset_path": str(igibson_dataset_root),
            "igibson_dataset_present": igibson_dataset_present,
        },
        "slots": slots,
        "notes": [
            "This preflight validates file presence and action-sequencing shape; the official evaluator remains authoritative.",
            "A custom PDDL final-state score is not an official EAI score.",
            "The pinned VirtualHome evaluator requires object name/ID pairs despite conflicting prompt prose.",
            "Use a multi-task smoke set covering state, relation, and action goals to avoid zero-denominator failures.",
            "BEHAVIOR evaluation requires the separately licensed iGibson scenes and objects dataset.",
        ],
    }


def export_official_preflight(
    response_root: str | Path,
    output_path: str | Path,
    *,
    external_root: str | Path = "external/embodied-agent-interface",
) -> dict[str, Any]:
    report = inspect_official_response_tree(response_root, external_root=external_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
