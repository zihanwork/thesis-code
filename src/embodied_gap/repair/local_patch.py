from __future__ import annotations

from embodied_gap.core.patch_schema import PatchType, PlanPatch
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.state_schema import WorldState
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation, ViolationType


class LocalPatchRepair:
    """Single-step patcher for missing preconditions."""

    name = "local_patch_repair"

    def repair(self, task: Task, plan: PlanCandidate, violation: Violation | None) -> PlanPatch:
        if not violation or violation.type not in {ViolationType.MISSING_STEP, ViolationType.WRONG_ORDER}:
            return self._none(plan, "Violation type is not locally patchable.")

        insertions: list[str] = []
        state = WorldState.from_facts(task.initial_facts)
        for missing in violation.missing_preconditions:
            if missing in state.facts:
                continue
            producer = self._find_immediate_producer(task, missing, state)
            if producer and producer not in insertions:
                insertions.append(producer)
                state = state.apply(task.action_model[producer])

        if not insertions:
            return self._none(plan, "No immediate producer action found.")

        index = violation.step_index or 0
        after = tuple(plan.actions[:index]) + tuple(insertions) + tuple(plan.actions[index:])
        return PlanPatch(
            patch_type=PatchType.INSERT,
            source=self.name,
            before=plan.actions,
            after=after,
            explanation="Inserted producer actions for missing preconditions.",
            metadata={"insertions": insertions, "index": index},
        )

    def _find_immediate_producer(self, task: Task, fact: str, state: WorldState) -> str | None:
        for action in task.allowed_actions:
            spec = task.action_model.get(action)
            if not spec or fact not in spec.add_effects:
                continue
            if state.contains_all(spec.preconditions):
                return action
        return None

    def _none(self, plan: PlanCandidate, explanation: str) -> PlanPatch:
        return PlanPatch(
            patch_type=PatchType.NONE,
            source=self.name,
            before=plan.actions,
            after=plan.actions,
            explanation=explanation,
        )
