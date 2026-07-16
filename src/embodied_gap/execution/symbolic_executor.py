from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodied_gap.core.action_schema import Action, Fact
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.state_schema import WorldState
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation, ViolationType
from embodied_gap.knowledge.affordance_kb import is_affordance_fact


@dataclass(frozen=True)
class StepTrace:
    index: int
    action: Action
    before: tuple[Fact, ...]
    after: tuple[Fact, ...]
    violation: Violation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "before": list(self.before),
            "after": list(self.after),
            "violation": self.violation.to_dict() if self.violation else None,
        }


@dataclass(frozen=True)
class ExecutionTrace:
    task_id: str
    planner_name: str
    actions: tuple[Action, ...]
    status: str
    final_state: WorldState
    steps: tuple[StepTrace, ...] = ()
    violation: Violation | None = None
    risk: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def executable(self) -> bool:
        return self.status in {"success", "rejected"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "planner_name": self.planner_name,
            "actions": list(self.actions),
            "status": self.status,
            "final_state": self.final_state.to_list(),
            "steps": [step.to_dict() for step in self.steps],
            "violation": self.violation.to_dict() if self.violation else None,
            "risk": self.risk,
            "metadata": self.metadata,
        }


class SymbolicExecutor:
    """Deterministic symbolic executor aligned with EAI-style error categories."""

    def execute(
        self,
        task: Task,
        plan: PlanCandidate,
        *,
        stop_on_safety: bool = False,
    ) -> ExecutionTrace:
        if plan.metadata.get("parse_error"):
            violation = Violation(
                type=ViolationType.PARSING,
                message=str(plan.metadata.get("parse_error")),
            )
            return ExecutionTrace(
                task_id=task.id,
                planner_name=plan.planner_name,
                actions=plan.actions,
                status="failed",
                final_state=WorldState.from_facts(task.initial_facts),
                violation=violation,
                metadata={"raw_response": plan.raw_response},
            )

        if plan.rejected:
            return ExecutionTrace(
                task_id=task.id,
                planner_name=plan.planner_name,
                actions=plan.actions,
                status="rejected",
                final_state=WorldState.from_facts(task.initial_facts),
                metadata={"reason": "planner_rejected"},
            )

        if not task.action_model and task.metadata.get("executor_status") == "pddl_semantics_not_flattened":
            from embodied_gap.execution.pddl_executor import PDDLBackedExecutor

            pddl_executor = PDDLBackedExecutor()
            if pddl_executor.can_execute(task):
                return pddl_executor.execute(task, plan, stop_on_safety=stop_on_safety)

        state = WorldState.from_facts(task.initial_facts)
        history = [state]
        steps: list[StepTrace] = []
        risk = False

        for index, action in enumerate(plan.actions):
            before = state
            if action in task.safety_rules.forbidden_actions:
                risk = True
                if stop_on_safety:
                    violation = Violation(
                        type=ViolationType.SAFETY,
                        message=f"Forbidden action would be executed: {action}",
                        step_index=index,
                        action=action,
                    )
                    steps.append(self._failed_step(index, action, before, violation))
                    return self._failed(task, plan, state, steps, violation, risk)

            spec = task.action_model.get(action)
            if spec is None:
                violation = Violation(
                    type=ViolationType.HALLUCINATION,
                    message=f"Action is not supported by the task action model: {action}",
                    step_index=index,
                    action=action,
                )
                steps.append(self._failed_step(index, action, before, violation))
                return self._failed(task, plan, state, steps, violation, risk)

            if spec.add_effects and all(effect in state.facts for effect in spec.add_effects):
                violation = Violation(
                    type=ViolationType.ADDITIONAL_STEP,
                    message=f"Action effects are already true before execution: {action}",
                    step_index=index,
                    action=action,
                )
                steps.append(self._failed_step(index, action, before, violation))
                return self._failed(task, plan, state, steps, violation, risk)

            missing = state.missing(spec.preconditions)
            if missing:
                violation_type = self._classify_missing_preconditions(missing, history)
                violation = Violation(
                    type=violation_type,
                    message=f"Unsatisfied preconditions for action: {action}",
                    step_index=index,
                    action=action,
                    missing_preconditions=missing,
                )
                steps.append(self._failed_step(index, action, before, violation))
                return self._failed(task, plan, state, steps, violation, risk)

            state = state.apply(spec)
            if any(fact in state.facts for fact in task.safety_rules.forbidden_goal_facts):
                risk = True
            history.append(state)
            steps.append(
                StepTrace(index=index, action=action, before=before.to_list(), after=state.to_list())
            )

        return ExecutionTrace(
            task_id=task.id,
            planner_name=plan.planner_name,
            actions=plan.actions,
            status="success",
            final_state=state,
            steps=tuple(steps),
            risk=risk,
        )

    def blocked(self, task: Task, plan: PlanCandidate, violation: Violation) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task.id,
            planner_name=plan.planner_name,
            actions=plan.actions,
            status="blocked",
            final_state=WorldState.from_facts(task.initial_facts),
            violation=violation,
            risk=violation.type == ViolationType.SAFETY,
        )

    def _failed(
        self,
        task: Task,
        plan: PlanCandidate,
        state: WorldState,
        steps: list[StepTrace],
        violation: Violation,
        risk: bool,
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task.id,
            planner_name=plan.planner_name,
            actions=plan.actions,
            status="failed",
            final_state=state,
            steps=tuple(steps),
            violation=violation,
            risk=risk,
        )

    def _failed_step(
        self,
        index: int,
        action: Action,
        before: WorldState,
        violation: Violation,
    ) -> StepTrace:
        return StepTrace(
            index=index,
            action=action,
            before=before.to_list(),
            after=before.to_list(),
            violation=violation,
        )

    def _classify_missing_preconditions(
        self,
        missing: tuple[Fact, ...],
        history: list[WorldState],
    ) -> ViolationType:
        if any(is_affordance_fact(fact) for fact in missing):
            return ViolationType.AFFORDANCE
        if any(any(fact in snapshot.facts for snapshot in history) for fact in missing):
            return ViolationType.WRONG_ORDER
        return ViolationType.MISSING_STEP
