from __future__ import annotations

from embodied_gap.core.patch_schema import PatchType, PlanPatch
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation, ViolationType


class SafetyRuleRepair:
    name = "safety_rule_repair"

    def repair(self, task: Task, plan: PlanCandidate, violation: Violation | None) -> PlanPatch:
        if violation and violation.type == ViolationType.SAFETY:
            return PlanPatch(
                patch_type=PatchType.REJECT,
                source=self.name,
                before=plan.actions,
                after=("reject()",),
                explanation="Safety validator detected a forbidden action.",
            )
        if task.safety_rules.should_reject:
            instruction = task.instruction.lower()
            if any(keyword.lower() in instruction for keyword in task.safety_rules.hazard_keywords):
                return PlanPatch(
                    patch_type=PatchType.REJECT,
                    source=self.name,
                    before=plan.actions,
                    after=("reject()",),
                    explanation="Task matches hazard keywords and should be rejected.",
                )
        return PlanPatch(
            patch_type=PatchType.NONE,
            source=self.name,
            before=plan.actions,
            after=plan.actions,
            explanation="No safety rule patch applied.",
        )
