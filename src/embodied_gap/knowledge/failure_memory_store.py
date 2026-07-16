from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from embodied_gap.core.action_schema import tokenize
from embodied_gap.core.state_schema import WorldState
from embodied_gap.core.task_schema import Task, load_tasks
from embodied_gap.core.violation_schema import Violation


@dataclass(frozen=True)
class FailureMemoryEntry:
    id: str
    source_task_id: str
    instruction: str
    dataset: str
    task_family: str
    tags: tuple[str, ...]
    error_type: str
    failed_plan: tuple[str, ...]
    repaired_plan: tuple[str, ...]
    repair_source: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FailureMemoryEntry":
        return cls(
            id=str(row["id"]),
            source_task_id=str(row["source_task_id"]),
            instruction=str(row["instruction"]),
            dataset=str(row.get("dataset", "")),
            task_family=str(row.get("task_family", "")),
            tags=tuple(str(value) for value in row.get("tags", [])),
            error_type=str(row.get("error_type", "unknown_failure")),
            failed_plan=tuple(str(value) for value in row.get("failed_plan", [])),
            repaired_plan=tuple(str(value) for value in row.get("repaired_plan", [])),
            repair_source=str(row.get("repair_source", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_task_id": self.source_task_id,
            "instruction": self.instruction,
            "dataset": self.dataset,
            "task_family": self.task_family,
            "tags": list(self.tags),
            "error_type": self.error_type,
            "failed_plan": list(self.failed_plan),
            "repaired_plan": list(self.repaired_plan),
            "repair_source": self.repair_source,
        }


@dataclass(frozen=True)
class RetrievedFailure:
    entry: FailureMemoryEntry
    score: float


class FrozenFailureMemory:
    """Read-only failure-to-repair examples created before final evaluation."""

    def __init__(
        self,
        entries: tuple[FailureMemoryEntry, ...] = (),
        *,
        source_path: str | None = None,
        sha256: str | None = None,
    ) -> None:
        self.entries = entries
        self.source_path = source_path
        self.sha256 = sha256

    @classmethod
    def empty(cls) -> "FrozenFailureMemory":
        return cls()

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "FrozenFailureMemory":
        source = Path(path)
        raw = source.read_bytes()
        entries = tuple(
            FailureMemoryEntry.from_dict(json.loads(line))
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        )
        return cls(
            entries,
            source_path=str(source),
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def retrieve(
        self,
        task: Task,
        violation: Violation | None,
        *,
        k: int = 1,
    ) -> list[RetrievedFailure]:
        query_tokens = tokenize(task.instruction)
        error_type = violation.type.value if violation else "unknown_failure"
        dataset = str(task.slots.get("dataset", ""))
        family = str(task.slots.get("task_family", ""))
        scored: list[RetrievedFailure] = []
        for entry in self.entries:
            if entry.source_task_id == task.id:
                continue
            entry_tokens = tokenize(entry.instruction)
            union = query_tokens | entry_tokens
            lexical = len(query_tokens & entry_tokens) / len(union) if union else 0.0
            score = lexical
            score += 1.0 if entry.error_type == error_type else 0.0
            score += 0.25 if dataset and entry.dataset == dataset else 0.0
            score += 0.25 if family and entry.task_family == family else 0.0
            tag_union = set(task.tags) | set(entry.tags)
            if tag_union:
                score += 0.25 * len(set(task.tags) & set(entry.tags)) / len(tag_union)
            scored.append(RetrievedFailure(entry, round(score, 6)))
        scored.sort(key=lambda item: (-item.score, item.entry.id))
        return scored[:k]

    def render_context(self, retrieved: list[RetrievedFailure]) -> str:
        if not retrieved:
            return "No frozen failure-repair example was retrieved."
        blocks = []
        for item in retrieved:
            entry = item.entry
            blocks.append(
                "\n".join(
                    [
                        f"Memory ID: {entry.id}",
                        f"Similar instruction: {entry.instruction}",
                        f"Error type: {entry.error_type}",
                        f"Failed plan: {list(entry.failed_plan)}",
                        f"Successful repair: {list(entry.repaired_plan)}",
                    ]
                )
            )
        return "\n\n".join(blocks)


def build_frozen_failure_memory(
    *,
    tasks_path: str | Path,
    runs_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    output_path: str | Path,
) -> dict[str, Any]:
    """Extract successful failure-to-repair pairs from development artifacts."""

    tasks = {task.id: task for task in load_tasks(tasks_path)}
    run_sources = (
        list(runs_path) if isinstance(runs_path, (list, tuple)) else [runs_path]
    )
    rows: list[dict[str, Any]] = []
    for run_source in run_sources:
        rows.extend(
            json.loads(line)
            for line in Path(run_source).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    entries: list[FailureMemoryEntry] = []
    seen: set[str] = set()
    for run in rows:
        task = tasks.get(str(run.get("task_id", "")))
        if task is None or task.split == "train":
            continue
        final_state = run.get("trace", {}).get("final_state", [])
        if not task.goal.is_satisfied(WorldState.from_facts(final_state)):
            continue
        for attempt in run.get("attempts", []):
            patch = attempt.get("patch")
            if not isinstance(patch, dict) or patch.get("before") == patch.get("after"):
                continue
            violation = attempt.get("trace", {}).get("violation") or {}
            error_type = str(violation.get("type", "unknown_failure"))
            identity = json.dumps(
                [task.id, error_type, patch.get("before", []), patch.get("after", [])],
                sort_keys=True,
            )
            entry_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            if entry_id in seen:
                continue
            seen.add(entry_id)
            entries.append(
                FailureMemoryEntry(
                    id=entry_id,
                    source_task_id=task.id,
                    instruction=task.instruction,
                    dataset=str(task.slots.get("dataset", "")),
                    task_family=str(task.slots.get("task_family", "")),
                    tags=task.tags,
                    error_type=error_type,
                    failed_plan=tuple(str(value) for value in patch.get("before", [])),
                    repaired_plan=tuple(str(value) for value in patch.get("after", [])),
                    repair_source=str(patch.get("source", "unknown")),
                )
            )

    entries.sort(key=lambda entry: entry.id)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for entry in entries
    )
    target.write_text(text, encoding="utf-8")
    repair_sources = Counter(entry.repair_source for entry in entries)
    error_types = Counter(entry.error_type for entry in entries)
    symbolic_sources = {"full_graph_replan", "symbolic_replan"}
    memory_teacher = (
        "symbolic_pddl"
        if repair_sources and set(repair_sources).issubset(symbolic_sources)
        else "mixed"
    )
    manifest = {
        "schema_version": 1,
        "tasks_path": str(tasks_path),
        "runs": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in run_sources
        ],
        "output_path": str(target),
        "entry_count": len(entries),
        "error_type_counts": dict(sorted(error_types.items())),
        "repair_source_counts": dict(sorted(repair_sources.items())),
        "memory_teacher": memory_teacher,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "policy": "development_successful_repairs_only; read_only_during_final_evaluation",
    }
    manifest_path = target.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
