from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from embodied_gap.core.task_schema import Task, dump_jsonl, load_tasks


@dataclass(frozen=True)
class TaskDifficulty:
    label: str
    plan_length: int
    goal_count: int
    object_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "plan_length": self.plan_length,
            "goal_count": self.goal_count,
            "object_count": self.object_count,
        }


class TaskSetBuilder:
    """Build thesis-ready task subsets from canonical clean task records."""

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = sorted(
            (task for task in tasks if task.slots.get("dataset") == "virtualhome"),
            key=lambda task: task.id,
        )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TaskSetBuilder":
        return cls(load_tasks(path))

    def export(self, out_dir: str | Path, per_family: int = 8) -> dict[str, Any]:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rag_train = [task for task in self.tasks if task.split == "train" and task.gold_plan]
        full_eval = [task for task in self.tasks if task.split != "train"]
        executable_eval = [
            task for task in full_eval if task.gold_plan and task.goal_facts
        ]
        balanced_eval = self._balanced_by_family(executable_eval, per_family=per_family)
        balanced_eval_20 = stratified_spread_sample(balanced_eval, limit=20)
        balanced_eval_50 = stratified_spread_sample(balanced_eval, limit=50)
        smoke_train = rag_train[:8]
        smoke_eval = smoke_train + executable_eval[:2]

        files = {
            "rag_train": output_dir / "rag_train.jsonl",
            "full_eval": output_dir / "full_eval.jsonl",
            "executable_eval": output_dir / "executable_eval.jsonl",
            "balanced_eval": output_dir / "balanced_eval.jsonl",
            "balanced_eval_20": output_dir / "balanced_eval_20.jsonl",
            "balanced_eval_50": output_dir / "balanced_eval_50.jsonl",
            "eai_smoke_eval": output_dir / "eai_smoke_eval.jsonl",
        }
        datasets = {
            "rag_train": rag_train,
            "full_eval": full_eval,
            "executable_eval": executable_eval,
            "balanced_eval": balanced_eval,
            "balanced_eval_20": balanced_eval_20,
            "balanced_eval_50": balanced_eval_50,
            "eai_smoke_eval": smoke_eval,
        }
        for name, path in files.items():
            dump_jsonl(path, [self._with_difficulty(task).to_dict() for task in datasets[name]])

        manifest = {
            "source_task_count": len(self.tasks),
            "per_family": per_family,
            "selection_policy": {
                "rag_train": "all train tasks with gold plans",
                "full_eval": "all non-train clean EAI tasks",
                "executable_eval": "non-train tasks with non-empty goals and gold plans",
                "balanced_eval": "deterministic length-spread sample per dataset/task_family",
                "balanced_eval_20": "stratified deterministic sample from balanced_eval by dataset/difficulty",
                "balanced_eval_50": "larger stratified deterministic sample from balanced_eval by dataset/difficulty",
                "eai_smoke_eval": "first 8 VirtualHome train demonstrations plus first 2 executable VirtualHome evaluation tasks",
            },
            "exclusions": {
                "missing_gold_plan": sum(not task.gold_plan for task in self.tasks),
                "empty_goal": sum(not task.goal_facts for task in self.tasks),
            },
            "files": {
                name: {
                    "path": str(path),
                    "rows": len(datasets[name]),
                    "summary": self._summarize(datasets[name]),
                }
                for name, path in files.items()
            },
        }
        manifest_path = output_dir / "tasksets_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def _balanced_by_family(self, tasks: list[Task], per_family: int) -> list[Task]:
        grouped: dict[tuple[str, str], list[Task]] = {}
        for task in tasks:
            key = (task.slots.get("dataset", "unknown"), task.slots.get("task_family", "unknown"))
            grouped.setdefault(key, []).append(task)

        selected: list[Task] = []
        for key in sorted(grouped):
            family_tasks = sorted(grouped[key], key=lambda task: (len(task.gold_plan), task.id))
            selected.extend(spread_sample(family_tasks, per_family))
        return sorted(selected, key=lambda task: task.id)

    def _with_difficulty(self, task: Task) -> Task:
        difficulty = classify_difficulty(task)
        metadata = dict(task.metadata)
        metadata["difficulty"] = difficulty.to_dict()
        return Task(
            id=task.id,
            instruction=task.instruction,
            initial_facts=task.initial_facts,
            goal=task.goal,
            allowed_actions=task.allowed_actions,
            action_model=task.action_model,
            split=task.split,
            tags=task.tags + (f"difficulty:{difficulty.label}",),
            slots=task.slots,
            gold_plan=task.gold_plan,
            safety_rules=task.safety_rules,
            source=task.source,
            metadata=metadata,
        )

    def _summarize(self, tasks: list[Task]) -> dict[str, Any]:
        by_dataset: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        plan_lengths: list[int] = []
        for task in tasks:
            by_dataset[task.slots.get("dataset", "unknown")] = (
                by_dataset.get(task.slots.get("dataset", "unknown"), 0) + 1
            )
            difficulty = classify_difficulty(task).label
            by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
            plan_lengths.append(len(task.gold_plan))
        return {
            "by_dataset": dict(sorted(by_dataset.items())),
            "by_difficulty": dict(sorted(by_difficulty.items())),
            "avg_gold_plan_length": (
                sum(plan_lengths) / len(plan_lengths) if plan_lengths else 0.0
            ),
            "max_gold_plan_length": max(plan_lengths) if plan_lengths else 0,
        }


def spread_sample(tasks: list[Task], limit: int) -> list[Task]:
    if limit <= 0 or len(tasks) <= limit:
        return list(tasks)
    if limit == 1:
        return [tasks[len(tasks) // 2]]
    last_index = len(tasks) - 1
    indices = sorted({round(index * last_index / (limit - 1)) for index in range(limit)})
    return [tasks[index] for index in indices]


def stratified_spread_sample(tasks: list[Task], limit: int) -> list[Task]:
    if limit <= 0 or len(tasks) <= limit:
        return sorted(tasks, key=lambda task: task.id)

    grouped: dict[tuple[str, str], list[Task]] = {}
    for task in tasks:
        key = (task.slots.get("dataset", "unknown"), classify_difficulty(task).label)
        grouped.setdefault(key, []).append(task)

    ordered_groups = {
        key: sorted(
            group,
            key=lambda task: (task.slots.get("task_family", ""), len(task.gold_plan), task.id),
        )
        for key, group in sorted(grouped.items())
    }
    total = len(tasks)
    allocations: dict[tuple[str, str], int] = {
        key: max(1, int(len(group) * limit / total))
        for key, group in ordered_groups.items()
    }

    while sum(allocations.values()) > limit:
        reducible = [key for key, count_value in allocations.items() if count_value > 1]
        if not reducible:
            break
        key_to_reduce = min(
            reducible,
            key=lambda key: (len(ordered_groups[key]) * limit / total) - allocations[key],
        )
        allocations[key_to_reduce] -= 1

    while sum(allocations.values()) < limit:
        expandable = [
            key for key, group in ordered_groups.items() if allocations[key] < len(group)
        ]
        if not expandable:
            break
        key_to_increase = max(
            expandable,
            key=lambda key: (
                len(ordered_groups[key]) * limit / total - allocations[key],
                len(ordered_groups[key]),
            ),
        )
        allocations[key_to_increase] += 1

    selected: list[Task] = []
    for key, group in ordered_groups.items():
        selected.extend(spread_sample(group, allocations[key]))

    selected_ids = {task.id for task in selected}
    if len(selected) < limit:
        remainder = [
            task for task in sorted(tasks, key=lambda task: task.id) if task.id not in selected_ids
        ]
        selected.extend(spread_sample(remainder, limit - len(selected)))

    return sorted(selected[:limit], key=lambda task: task.id)


def classify_difficulty(task: Task) -> TaskDifficulty:
    plan_length = len(task.gold_plan)
    goal_count = len(task.goal_facts)
    object_count = int(task.metadata.get("object_count", 0) or 0)
    if plan_length <= 4 and goal_count <= 2 and object_count <= 8:
        label = "easy"
    elif plan_length <= 12 and goal_count <= 6 and object_count <= 20:
        label = "medium"
    else:
        label = "hard"
    return TaskDifficulty(
        label=label,
        plan_length=plan_length,
        goal_count=goal_count,
        object_count=object_count,
    )
