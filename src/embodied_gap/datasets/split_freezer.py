from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from embodied_gap.core.task_schema import Task, load_tasks


def freeze_heldout_split(
    *,
    executable_path: str | Path,
    development_path: str | Path,
    output_dir: str | Path,
    name: str,
    expected_count: int,
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    """Freeze the executable-minus-development split and refuse silent changes."""

    executable_source = Path(executable_path)
    development_source = Path(development_path)
    executable = load_tasks(executable_source)
    development = load_tasks(development_source)
    executable_by_id = _unique_by_id(executable, "executable")
    development_by_id = _unique_by_id(development, "development")

    unknown_development = sorted(set(development_by_id) - set(executable_by_id))
    if unknown_development:
        raise ValueError(
            "Development tasks are missing from the executable inventory: "
            + ", ".join(unknown_development[:10])
        )

    heldout = [
        executable_by_id[task_id]
        for task_id in sorted(set(executable_by_id) - set(development_by_id))
    ]
    if len(heldout) != expected_count:
        raise ValueError(
            f"Held-out count changed: expected {expected_count}, found {len(heldout)}"
        )
    datasets = Counter(str(task.slots.get("dataset", "unknown")) for task in heldout)
    if expected_dataset and set(datasets) != {expected_dataset}:
        raise ValueError(
            f"Held-out dataset changed: expected only {expected_dataset}, found {dict(datasets)}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / f"{name}.jsonl"
    ids_path = output / f"{name}_ids.json"
    manifest_path = output / f"{name}_manifest.json"

    task_text = "".join(
        json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for task in heldout
    )
    task_ids = [task.id for task in heldout]
    ids_payload = {
        "name": name,
        "frozen": True,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_ids_sha256": _json_sha256(task_ids),
    }
    ids_text = json.dumps(ids_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest = {
        "schema_version": 1,
        "name": name,
        "frozen": True,
        "policy": "executable inventory minus development; never tune on held-out failures",
        "scope": "local held-out only; official hidden test remains the primary final benchmark",
        "inputs": {
            "executable": _file_fingerprint(executable_source),
            "development": _file_fingerprint(development_source),
        },
        "development_task_count": len(development),
        "heldout_task_count": len(heldout),
        "overlap_count": 0,
        "by_dataset": dict(sorted(datasets.items())),
        "by_difficulty": dict(
            sorted(
                Counter(
                    str(task.metadata.get("difficulty", {}).get("label", "unknown"))
                    for task in heldout
                ).items()
            )
        ),
        "task_ids_sha256": ids_payload["task_ids_sha256"],
        "tasks_sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
        "files": {
            "tasks": str(task_path),
            "ids": str(ids_path),
        },
        "limitations": [
            "All locally held-out tasks are VirtualHome.",
            "No untouched local BEHAVIOR final split remains.",
        ],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    _write_once_or_verify(task_path, task_text)
    _write_once_or_verify(ids_path, ids_text)
    _write_once_or_verify(manifest_path, manifest_text)
    return manifest


def _unique_by_id(tasks: list[Task], label: str) -> dict[str, Task]:
    rows: dict[str, Task] = {}
    for task in tasks:
        if task.id in rows:
            raise ValueError(f"Duplicate task ID in {label} split: {task.id}")
        rows[task.id] = task
    return rows


def _file_fingerprint(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_once_or_verify(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Frozen split artifact would change: {path}")
        return
    path.write_text(content, encoding="utf-8")
