from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from embodied_gap.core.action_schema import parse_call


OFFICIAL_DATASETS = ("virtualhome",)
OFFICIAL_MODULES = ("action_sequencing",)

# This mirrors valid_actions in the evaluator at the pinned EAI commit. It is
# deliberately not expanded from the prompt prose: PLUGIN and PLUGOUT are
# advertised there but absent from the executable evaluator dictionary.
VIRTUALHOME_OFFICIAL_ACTION_ARITY = {
    "CLOSE": 1,
    "CUT": 1,
    "DRINK": 1,
    "DROP": 1,
    "EAT": 1,
    "FIND": 1,
    "GRAB": 1,
    "GREET": 1,
    "LIE": 1,
    "LOOKAT": 1,
    "LOOKAT_LONG": 1,
    "LOOKAT_MEDIUM": 1,
    "LOOKAT_SHORT": 1,
    "MOVE": 1,
    "OPEN": 1,
    "POINTAT": 1,
    "POUR": 2,
    "PULL": 1,
    "PUSH": 1,
    "PUTBACK": 2,
    "PUTIN": 2,
    "PUTOBJBACK": 1,
    "PUTOFF": 1,
    "PUTON": 1,
    "READ": 1,
    "RINSE": 1,
    "RUN": 1,
    "SCRUB": 1,
    "SIT": 1,
    "SLEEP": 0,
    "SQUEEZE": 1,
    "STANDUP": 0,
    "SWITCHOFF": 1,
    "SWITCHON": 1,
    "TOUCH": 1,
    "TURNTO": 1,
    "TYPE": 1,
    "WAKEUP": 0,
    "WALK": 1,
    "WASH": 1,
    "WATCH": 1,
    "WIPE": 1,
}

# Source PDDL action -> (official action, indices of canonical PDDL arguments
# retained after removing the implicit character and other PDDL-only context).
VIRTUALHOME_PDDL_TO_OFFICIAL = {
    "close": ("CLOSE", (1,)),
    "cut": ("CUT", (1,)),
    "drink": ("DRINK", (1,)),
    "drop": ("DROP", (1,)),
    "eat": ("EAT", (1,)),
    "find": ("FIND", (1,)),
    "grab": ("GRAB", (1,)),
    "lie": ("LIE", (1,)),
    "look_at": ("LOOKAT", (1,)),
    "move": ("MOVE", (1,)),
    "open": ("OPEN", (1,)),
    "plug_in": ("PLUGIN", (1,)),
    "plug_out": ("PLUGOUT", (1,)),
    "pour": ("POUR", (1, 2)),
    "put_inside": ("PUTIN", (1, 2)),
    "put_on": ("PUTBACK", (1, 2)),
    "put_on_character": ("PUTON", (1,)),
    "read": ("READ", (1,)),
    "sit": ("SIT", (1,)),
    "sleep": ("SLEEP", ()),
    "squeeze": ("SQUEEZE", (1,)),
    "standup": ("STANDUP", ()),
    "switch_off": ("SWITCHOFF", (1,)),
    "switch_on": ("SWITCHON", (1,)),
    "touch": ("TOUCH", (1,)),
    "turn_to": ("TURNTO", (1,)),
    "type": ("TYPE", (1,)),
    "wake_up": ("WAKEUP", ()),
    "walk_into": ("WALK", (1,)),
    "walk_towards": ("WALK", (1,)),
    "wash": ("WASH", (1,)),
    "watch": ("WATCH", (1,)),
    # The official WIPE action has one target; the PDDL's second object is the
    # cleaning implement and is implicit in the official primitive.
    "wipe": ("WIPE", (1,)),
}

_VIRTUALHOME_OBJECT_LINE = re.compile(
    r"^\s*([^,\n]+),\s*id:\s*(\d+),\s*properties:", re.MULTILINE
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
    empty_plan_count = 0
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
        if not isinstance(parsed, list):
            issues.append(
                _issue(
                    "nonlist_plan",
                    "Action sequence must be a JSON list.",
                    identifier=identifier,
                )
            )
            continue
        if not parsed:
            # The pinned evaluator accepts [] and records it as a parsing failure.
            # Keeping the row is required to preserve the fixed task denominator.
            empty_plan_count += 1
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
                    continue
                if action not in VIRTUALHOME_OFFICIAL_ACTION_ARITY:
                    issues.append(
                        _issue(
                            "virtualhome_unsupported_action",
                            f"Step {step_index} action {action!r} is not supported by the pinned evaluator.",
                            identifier=identifier,
                        )
                    )
                    continue
                expected_pairs = VIRTUALHOME_OFFICIAL_ACTION_ARITY[action]
                if len(arguments) // 2 != expected_pairs:
                    issues.append(
                        _issue(
                            "virtualhome_action_arity",
                            f"Step {step_index} action {action} expects {expected_pairs} object pair(s).",
                            identifier=identifier,
                        )
                    )
                for pair_index in range(0, len(arguments), 2):
                    name = arguments[pair_index]
                    object_id = arguments[pair_index + 1]
                    if not isinstance(name, str) or not (
                        isinstance(object_id, int)
                        or (isinstance(object_id, str) and object_id.isdigit())
                    ):
                        issues.append(
                            _issue(
                                "virtualhome_name_id_types",
                                f"Step {step_index} requires string names and numeric IDs.",
                                identifier=identifier,
                            )
                        )
                        break

    return {
        "dataset": dataset,
        "record_count": len(records),
        "parsed_count": parsed_count,
        "empty_plan_count": empty_plan_count,
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
            "empty_plan_count": 0,
            "nonempty_plan_count": 0,
            "valid": False,
            "issues": [_issue("file_unreadable", str(exc))],
        }
    result = validate_action_sequencing_records(records, dataset=dataset)
    result["path"] = str(path)
    return result


def load_virtualhome_prompt_object_ids(
    prompt_path: str | Path,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Load task-specific object name/ID candidates from official prompts."""

    records = json.loads(Path(prompt_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Official VirtualHome prompt file must contain a list.")
    task_objects: dict[str, dict[str, tuple[int, ...]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        identifier = record.get("identifier")
        prompt = record.get("llm_prompt")
        if not isinstance(identifier, str) or not isinstance(prompt, str):
            continue
        candidates: dict[str, list[int]] = {}
        for name, raw_id in _VIRTUALHOME_OBJECT_LINE.findall(prompt):
            candidates.setdefault(name.strip(), []).append(int(raw_id))
        task_objects[identifier] = {
            name: tuple(dict.fromkeys(ids)) for name, ids in candidates.items()
        }
    return task_objects


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
            rows.append(row)
    return rows


def _atomic_write_json_value(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _translate_virtualhome_action(
    action_text: str,
    object_ids: dict[str, tuple[int, ...]],
) -> tuple[dict[str, list[Any]] | None, dict[str, str] | None]:
    call = parse_call(action_text)
    mapping = VIRTUALHOME_PDDL_TO_OFFICIAL.get(call.name)
    if mapping is None:
        return None, _issue(
            "source_action_unmapped",
            f"Canonical action {call.name!r} has no reviewed official mapping.",
        )
    official_action, argument_indices = mapping
    if official_action not in VIRTUALHOME_OFFICIAL_ACTION_ARITY:
        return None, _issue(
            "unsupported_at_pinned_commit",
            f"{call.name} maps to {official_action}, which the pinned evaluator does not execute.",
        )
    if any(index >= len(call.args) for index in argument_indices):
        return None, _issue(
            "source_action_arity",
            f"Canonical action {action_text!r} does not contain the expected arguments.",
        )

    arguments: list[Any] = []
    for index in argument_indices:
        name = call.args[index]
        candidates = object_ids.get(name, ())
        if not candidates:
            return None, _issue(
                "official_object_missing",
                f"Object {name!r} is absent from the task-specific official prompt.",
            )
        if len(candidates) != 1:
            return None, _issue(
                "official_object_ambiguous",
                f"Object {name!r} has multiple official IDs: {list(candidates)}.",
            )
        arguments.extend((name, candidates[0]))
    return {official_action: arguments}, None


def export_virtualhome_action_sequencing(
    *,
    runs_path: str | Path,
    tasks_path: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    planner_name: str,
    harness_mode: str,
    allow_partial: bool = False,
    include_failed_predictions: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export one exact project method to pinned official VirtualHome format.

    The exporter never guesses object IDs. A strict export writes the official
    output only when every expected VirtualHome task has one selected run and
    every action can be translated without ambiguity. For a prevalidated fixed
    cohort, ``include_failed_predictions`` preserves the common denominator by
    emitting ``[]`` for model outputs that cannot be translated; the pinned
    evaluator then counts those rows as failed predictions.
    """

    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Official export is non-overwriting; already exists: "
            + ", ".join(str(path) for path in existing)
        )

    tasks = _read_jsonl(tasks_path)
    runs = _read_jsonl(runs_path)
    prompt_objects = load_virtualhome_prompt_object_ids(prompts_path)
    virtualhome_tasks = [
        task
        for task in tasks
        if task.get("slots", {}).get("dataset") == "virtualhome"
        or task.get("metadata", {}).get("dataset") == "virtualhome"
    ]
    expected_task_ids = {str(task.get("id", "")) for task in virtualhome_tasks}
    selected: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if (
            str(run.get("task_id")) in expected_task_ids
            and run.get("planner_name") == planner_name
            and run.get("harness_mode") == harness_mode
        ):
            selected.setdefault(str(run.get("task_id")), []).append(run)

    outputs: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    failed_prediction_count = 0
    for task in virtualhome_tasks:
        task_id = str(task.get("id", ""))
        identifier = str(task.get("slots", {}).get("file_id", ""))
        task_runs = selected.get(task_id, [])
        if len(task_runs) != 1:
            code = "selected_run_missing" if not task_runs else "selected_run_duplicate"
            issues.append(
                _issue(
                    code,
                    f"Expected exactly one {planner_name}/{harness_mode} run; found {len(task_runs)}.",
                    identifier=identifier or task_id,
                )
            )
            if include_failed_predictions and identifier:
                outputs.append({"identifier": identifier, "llm_output": "[]"})
                failed_prediction_count += 1
            continue
        if not identifier or identifier not in prompt_objects:
            issues.append(
                _issue(
                    "official_prompt_missing",
                    "Task has no matching official VirtualHome action-sequencing prompt.",
                    identifier=identifier or task_id,
                )
            )
            continue
        actions = task_runs[0].get("final_plan", {}).get("actions")
        if not isinstance(actions, list) or not actions or actions == ["reject()"]:
            issues.append(
                _issue(
                    "final_plan_not_exportable",
                    "Selected run has no non-rejected final plan.",
                    identifier=identifier,
                )
            )
            if include_failed_predictions:
                outputs.append({"identifier": identifier, "llm_output": "[]"})
                failed_prediction_count += 1
            continue

        translated: list[dict[str, list[Any]]] = []
        task_failed = False
        for action_index, action_text in enumerate(actions):
            if not isinstance(action_text, str):
                issue = _issue(
                    "source_action_not_string",
                    f"Action {action_index} is not a string.",
                    identifier=identifier,
                )
                issues.append(issue)
                task_failed = True
                break
            step, issue = _translate_virtualhome_action(
                action_text,
                prompt_objects[identifier],
            )
            if issue is not None:
                issue["identifier"] = identifier
                issue["action_index"] = str(action_index)
                issues.append(issue)
                task_failed = True
                break
            assert step is not None
            translated.append(step)
        if not task_failed:
            outputs.append(
                {
                    "identifier": identifier,
                    "llm_output": json.dumps(translated, ensure_ascii=False, separators=(",", ":")),
                }
            )
        elif include_failed_predictions:
            outputs.append({"identifier": identifier, "llm_output": "[]"})
            failed_prediction_count += 1

    outputs.sort(key=lambda row: row["identifier"])
    complete = len(outputs) == len(virtualhome_tasks) and (
        include_failed_predictions or not issues
    )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset": "virtualhome",
        "module": "action_sequencing",
        "runs_path": str(runs_path),
        "tasks_path": str(tasks_path),
        "prompts_path": str(prompts_path),
        "output_path": str(output_path),
        "planner_name": planner_name,
        "harness_mode": harness_mode,
        "expected_task_count": len(virtualhome_tasks),
        "selected_run_count": sum(len(value) for value in selected.values()),
        "exported_task_count": len(outputs),
        "skipped_task_count": len(virtualhome_tasks) - len(outputs),
        "complete": complete,
        "partial_output_written": bool(allow_partial and outputs and not complete),
        "failed_prediction_count": failed_prediction_count,
        "common_denominator_preserved": bool(
            include_failed_predictions and len(outputs) == len(virtualhome_tasks)
        ),
        "issues": issues,
        "notes": [
            "Object IDs come only from each task's pinned official prompt; ambiguous IDs are never guessed.",
            "A successful export is an input artifact, not an official score; run the pinned evaluator separately.",
            "When enabled, failed predictions are emitted as [] so the official evaluator counts them as failures on the fixed cohort.",
        ],
    }
    _atomic_write_json_value(manifest_path, manifest)
    if complete or (allow_partial and outputs):
        _atomic_write_json_value(output_path, outputs)
    return manifest


def build_virtualhome_official_cohort(
    *,
    tasks_path: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze tasks whose gold plans are representable by the pinned evaluator.

    Cohort membership depends only on benchmark task resources and the pinned
    evaluator contract, never on any tested model or treatment output.
    """

    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Official cohort export is non-overwriting; already exists: "
            + ", ".join(str(path) for path in existing)
        )

    tasks = _read_jsonl(tasks_path)
    prompt_objects = load_virtualhome_prompt_object_ids(prompts_path)
    included: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for task in tasks:
        dataset = task.get("slots", {}).get("dataset") or task.get("metadata", {}).get("dataset")
        if dataset != "virtualhome":
            continue
        task_id = str(task.get("id", ""))
        identifier = str(task.get("slots", {}).get("file_id", ""))
        if not identifier or identifier not in prompt_objects:
            exclusions.append(
                _issue(
                    "official_prompt_missing",
                    "Task has no matching official VirtualHome action-sequencing prompt.",
                    identifier=identifier or task_id,
                )
            )
            continue
        gold_plan = task.get("gold_plan")
        if not isinstance(gold_plan, list) or not gold_plan:
            exclusions.append(
                _issue(
                    "gold_plan_missing",
                    "Task has no gold action sequence for evaluator-compatibility screening.",
                    identifier=identifier,
                )
            )
            continue
        compatible = True
        for action_index, action_text in enumerate(gold_plan):
            if not isinstance(action_text, str):
                issue = _issue(
                    "gold_action_not_string",
                    f"Gold action {action_index} is not a string.",
                    identifier=identifier,
                )
                exclusions.append(issue)
                compatible = False
                break
            _, issue = _translate_virtualhome_action(action_text, prompt_objects[identifier])
            if issue is not None:
                issue["identifier"] = identifier
                issue["action_index"] = str(action_index)
                exclusions.append(issue)
                compatible = False
                break
        if compatible:
            included.append(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for task in included:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    manifest = {
        "format_version": 1,
        "dataset": "virtualhome",
        "module": "action_sequencing",
        "source_tasks_path": str(tasks_path),
        "prompts_path": str(prompts_path),
        "output_path": str(output_path),
        "source_virtualhome_task_count": sum(
            1
            for task in tasks
            if task.get("slots", {}).get("dataset") == "virtualhome"
            or task.get("metadata", {}).get("dataset") == "virtualhome"
        ),
        "included_task_count": len(included),
        "excluded_task_count": len(exclusions),
        "selection_policy": (
            "Pre-outcome screening: include only tasks whose gold plan and task-specific "
            "object IDs are executable under the pinned official VirtualHome action-sequencing evaluator."
        ),
        "included_identifiers": sorted(
            str(task.get("slots", {}).get("file_id", "")) for task in included
        ),
        "exclusions": exclusions,
    }
    _atomic_write_json_value(manifest_path, manifest)
    return manifest


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
    virtualhome_runtime_marker = external_root / "src" / "virtualhome_eval"
    official_runtime_ready = virtualhome_runtime_marker.is_dir()
    structurally_ready = present_count == len(slots) and action_shapes_valid
    return {
        "protocol": "official-eai-virtualhome-action-sequencing",
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
        },
        "slots": slots,
        "notes": [
            "This preflight validates file presence and action-sequencing shape; the official evaluator remains authoritative.",
            "A custom PDDL final-state score is not an official EAI score.",
            "The pinned VirtualHome evaluator requires object name/ID pairs despite conflicting prompt prose.",
            "Use a multi-task smoke set covering state, relation, and action goals to avoid zero-denominator failures.",
            "This scoped preflight covers only the official VirtualHome action-sequencing evaluator.",
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
