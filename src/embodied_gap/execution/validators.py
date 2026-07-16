from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation

from .symbolic_executor import ExecutionTrace, SymbolicExecutor


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    trace: ExecutionTrace
    violation: Violation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "trace": self.trace.to_dict(),
            "violation": self.violation.to_dict() if self.violation else None,
        }


class PlanValidator:
    def __init__(self, executor: SymbolicExecutor | None = None) -> None:
        self.executor = executor or SymbolicExecutor()

    def validate(self, task: Task, plan: PlanCandidate) -> ValidationReport:
        trace = self.executor.execute(task, plan, stop_on_safety=True)
        return ValidationReport(
            valid=trace.status in {"success", "rejected"} and not trace.risk,
            trace=trace,
            violation=trace.violation,
        )
