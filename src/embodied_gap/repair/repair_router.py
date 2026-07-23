from __future__ import annotations

from embodied_gap.core.patch_schema import PatchType, PlanPatch
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation

class RepairRouter:
    """Apply the explicitly configured repair strategies in order."""

    def __init__(self, strategies: list[object]) -> None:
        self.strategies = strategies

    def repair(self, task: Task, plan: PlanCandidate, violation: Violation | None) -> PlanPatch:
        last_patch: PlanPatch | None = None
        for strategy in self.strategies:
            patch = strategy.repair(task, plan, violation)
            last_patch = patch
            if patch.patch_type != PatchType.NONE and patch.changed():
                return patch
        return last_patch or PlanPatch(
            patch_type=PatchType.NONE,
            source="repair_router",
            before=plan.actions,
            after=plan.actions,
            explanation="No repair strategy available.",
        )
