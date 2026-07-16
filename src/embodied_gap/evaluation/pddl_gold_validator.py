from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task, dump_jsonl, load_tasks
from embodied_gap.execution.symbolic_executor import SymbolicExecutor


@dataclass(frozen=True)
class GoldPlanValidationRecord:
    task_id: str
    dataset: str
    task_family: str
    split: str
    plan_length: int
    executable: bool
    goal_success: bool
    success: bool
    trace_status: str
    violation_type: str | None
    violation_message: str | None
    engine: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "task_family": self.task_family,
            "split": self.split,
            "plan_length": self.plan_length,
            "executable": self.executable,
            "goal_success": self.goal_success,
            "success": self.success,
            "trace_status": self.trace_status,
            "violation_type": self.violation_type,
            "violation_message": self.violation_message,
            "engine": self.engine,
        }


class PDDLGoldPlanValidator:
    def __init__(self, executor: SymbolicExecutor | None = None) -> None:
        self.executor = executor or SymbolicExecutor()

    def validate_tasks(self, tasks: list[Task], limit: int | None = None) -> list[GoldPlanValidationRecord]:
        selected = tasks[:limit] if limit is not None else tasks
        records: list[GoldPlanValidationRecord] = []
        for task in selected:
            if not task.gold_plan:
                records.append(self._missing_plan_record(task))
                continue
            plan = PlanCandidate(
                planner_name="gold_pddl_plan",
                actions=task.gold_plan,
                raw_response=str(list(task.gold_plan)),
                metadata={"planner_family": "gold"},
            )
            trace = self.executor.execute(task, plan)
            goal_success = task.goal.is_satisfied(trace.final_state)
            violation = trace.violation
            records.append(
                GoldPlanValidationRecord(
                    task_id=task.id,
                    dataset=task.slots.get("dataset", "unknown"),
                    task_family=task.slots.get("task_family", "unknown"),
                    split=task.split,
                    plan_length=len(task.gold_plan),
                    executable=trace.executable,
                    goal_success=goal_success,
                    success=trace.executable and goal_success,
                    trace_status=trace.status,
                    violation_type=violation.type.value if violation else None,
                    violation_message=violation.message if violation else None,
                    engine=trace.metadata.get("engine"),
                )
            )
        return records

    def export(
        self,
        tasks_path: str | Path,
        out_dir: str | Path,
        limit: int | None = None,
    ) -> dict[str, Any]:
        tasks = load_tasks(tasks_path)
        records = self.validate_tasks(tasks, limit=limit)
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(output_dir / "gold_plan_validation.jsonl", [record.to_dict() for record in records])
        summary = summarize_validation(records)
        summary.update(
            {
                "tasks_path": str(tasks_path),
                "record_count": len(records),
                "limit": limit,
                "records_path": str(output_dir / "gold_plan_validation.jsonl"),
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def _missing_plan_record(self, task: Task) -> GoldPlanValidationRecord:
        return GoldPlanValidationRecord(
            task_id=task.id,
            dataset=task.slots.get("dataset", "unknown"),
            task_family=task.slots.get("task_family", "unknown"),
            split=task.split,
            plan_length=0,
            executable=False,
            goal_success=False,
            success=False,
            trace_status="missing_gold_plan",
            violation_type="no_plan",
            violation_message="Task has no gold PDDL plan.",
            engine=None,
        )


def summarize_validation(records: list[GoldPlanValidationRecord]) -> dict[str, Any]:
    by_dataset: dict[str, list[GoldPlanValidationRecord]] = {}
    failures: dict[str, int] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)
        if not record.success:
            key = record.violation_type or record.trace_status
            failures[key] = failures.get(key, 0) + 1

    return {
        "overall": summarize_rows(records),
        "by_dataset": {
            dataset: summarize_rows(rows) for dataset, rows in sorted(by_dataset.items())
        },
        "failure_counts": dict(sorted(failures.items())),
    }


def summarize_rows(records: list[GoldPlanValidationRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "n": 0,
            "executable_rate": 0.0,
            "goal_success_rate": 0.0,
            "success_rate": 0.0,
        }
    return {
        "n": total,
        "executable_rate": sum(record.executable for record in records) / total,
        "goal_success_rate": sum(record.goal_success for record in records) / total,
        "success_rate": sum(record.success for record in records) / total,
        "avg_plan_length": sum(record.plan_length for record in records) / total,
    }
