from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodied_gap.core.patch_schema import PlanPatch
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.violation_schema import Violation, ViolationType
from embodied_gap.execution.goal_checker import GoalChecker
from embodied_gap.execution.symbolic_executor import ExecutionTrace, SymbolicExecutor
from embodied_gap.execution.validators import PlanValidator
from embodied_gap.planners.base import InitialPlanner
from embodied_gap.repair.repair_router import RepairRouter
from embodied_gap.core.task_schema import Task

from .recovery_policy import HarnessMode
from .termination import RetryBudget


@dataclass(frozen=True)
class HarnessAttempt:
    retry_index: int
    plan: PlanCandidate
    trace: ExecutionTrace
    patch: PlanPatch | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retry_index": self.retry_index,
            "plan": self.plan.to_dict(),
            "trace": self.trace.to_dict(),
            "patch": self.patch.to_dict() if self.patch else None,
        }


@dataclass(frozen=True)
class HarnessRun:
    task_id: str
    planner_name: str
    harness_mode: HarnessMode
    initial_plan: PlanCandidate
    final_plan: PlanCandidate
    trace: ExecutionTrace
    attempts: tuple[HarnessAttempt, ...] = ()
    patches: tuple[PlanPatch, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def method_id(self) -> str:
        return f"{self.planner_name}__{self.harness_mode.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "planner_name": self.planner_name,
            "harness_mode": self.harness_mode.value,
            "method_id": self.method_id,
            "initial_plan": self.initial_plan.to_dict(),
            "final_plan": self.final_plan.to_dict(),
            "trace": self.trace.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "patches": [patch.to_dict() for patch in self.patches],
            "metadata": self.metadata,
        }


class HarnessController:
    """Runs an initial planner under an execution/recovery mode."""

    def __init__(
        self,
        executor: SymbolicExecutor | None = None,
        validator: PlanValidator | None = None,
        repair_router: RepairRouter | None = None,
        max_retries: int = 3,
    ) -> None:
        self.executor = executor or SymbolicExecutor()
        self.validator = validator or PlanValidator(self.executor)
        self.repair_router = repair_router or RepairRouter()
        self.goal_checker = GoalChecker()
        self.retry_budget = RetryBudget(max_retries)

    def run(
        self,
        task: Task,
        planner: InitialPlanner,
        mode: HarnessMode,
        initial_plan: PlanCandidate | None = None,
    ) -> HarnessRun:
        initial_plan = initial_plan or planner.plan(task)
        if mode == HarnessMode.H0_OPEN_LOOP:
            return self._open_loop(task, planner.name, initial_plan, mode)
        if mode == HarnessMode.H1_VERIFIER_GATED:
            return self._verifier_gated(task, planner.name, initial_plan, mode)
        return self._full_recovery(task, planner.name, initial_plan, mode)

    def _open_loop(
        self,
        task: Task,
        planner_name: str,
        initial_plan: PlanCandidate,
        mode: HarnessMode,
    ) -> HarnessRun:
        trace = self.executor.execute(task, initial_plan, stop_on_safety=False)
        attempt = HarnessAttempt(0, initial_plan, trace)
        return HarnessRun(
            task_id=task.id,
            planner_name=planner_name,
            harness_mode=mode,
            initial_plan=initial_plan,
            final_plan=initial_plan,
            trace=trace,
            attempts=(attempt,),
        )

    def _verifier_gated(
        self,
        task: Task,
        planner_name: str,
        initial_plan: PlanCandidate,
        mode: HarnessMode,
    ) -> HarnessRun:
        report = self.validator.validate(task, initial_plan)
        final_plan = initial_plan
        trace = report.trace
        if report.violation and report.violation.type == ViolationType.SAFETY:
            final_plan = PlanCandidate(
                planner_name=initial_plan.planner_name,
                actions=("reject()",),
                raw_response='["reject()"]',
                prompt=initial_plan.prompt,
                metadata={"gated_by": "safety_validator"},
            )
            trace = self.executor.execute(task, final_plan)
        elif not report.valid:
            violation = report.violation or Violation(
                type=ViolationType.BLOCKED,
                message="Verifier blocked the plan.",
            )
            trace = self.executor.blocked(task, initial_plan, violation)
        attempt = HarnessAttempt(0, initial_plan, trace)
        return HarnessRun(
            task_id=task.id,
            planner_name=planner_name,
            harness_mode=mode,
            initial_plan=initial_plan,
            final_plan=final_plan,
            trace=trace,
            attempts=(attempt,),
        )

    def _full_recovery(
        self,
        task: Task,
        planner_name: str,
        initial_plan: PlanCandidate,
        mode: HarnessMode,
    ) -> HarnessRun:
        plan = initial_plan
        attempts: list[HarnessAttempt] = []
        patches: list[PlanPatch] = []

        for retry_index in range(self.retry_budget.max_retries + 1):
            report = self.validator.validate(task, plan)
            trace = report.trace
            violation = report.violation
            if report.valid and not self.goal_checker.is_success(task, trace.final_state):
                violation = Violation(
                    type=ViolationType.GOAL_UNSATISFIED,
                    message="Plan executed but did not satisfy all goal facts.",
                )

            if report.valid and violation is None and self.goal_checker.is_success(task, trace.final_state):
                attempts.append(HarnessAttempt(retry_index, plan, trace))
                return HarnessRun(
                    task_id=task.id,
                    planner_name=planner_name,
                    harness_mode=mode,
                    initial_plan=initial_plan,
                    final_plan=plan,
                    trace=trace,
                    attempts=tuple(attempts),
                    patches=tuple(patches),
                )

            if report.valid and plan.rejected:
                attempts.append(HarnessAttempt(retry_index, plan, trace))
                return HarnessRun(
                    task_id=task.id,
                    planner_name=planner_name,
                    harness_mode=mode,
                    initial_plan=initial_plan,
                    final_plan=plan,
                    trace=trace,
                    attempts=tuple(attempts),
                    patches=tuple(patches),
                )

            if self.retry_budget.exhausted(retry_index):
                attempts.append(HarnessAttempt(retry_index, plan, trace))
                break

            patch = self.repair_router.repair(task, plan, violation)
            patches.append(patch)
            attempts.append(HarnessAttempt(retry_index, plan, trace, patch))
            if not patch.changed():
                break
            metadata = {key: value for key, value in plan.metadata.items() if key != "parse_error"}
            if "parse_error" in plan.metadata:
                metadata["repaired_from_parse_error"] = plan.metadata["parse_error"]
            metadata["last_patch"] = patch.to_dict()
            plan = PlanCandidate(
                planner_name=plan.planner_name,
                actions=patch.after,
                raw_response=str(list(patch.after)),
                prompt=plan.prompt,
                metadata=metadata,
            )

        final_trace = self.validator.validate(task, plan).trace
        return HarnessRun(
            task_id=task.id,
            planner_name=planner_name,
            harness_mode=mode,
            initial_plan=initial_plan,
            final_plan=plan,
            trace=final_trace,
            attempts=tuple(attempts),
            patches=tuple(patches),
            metadata={"termination": "retry_budget_or_no_patch"},
        )
