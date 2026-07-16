from __future__ import annotations

from embodied_gap.core.violation_schema import ViolationType

EAI_ERROR_COLUMNS = (
    ViolationType.PARSING.value,
    ViolationType.HALLUCINATION.value,
    ViolationType.ACTION_ARG_NUM.value,
    ViolationType.WRONG_ORDER.value,
    ViolationType.MISSING_STEP.value,
    ViolationType.AFFORDANCE.value,
    ViolationType.ADDITIONAL_STEP.value,
    ViolationType.SAFETY.value,
    ViolationType.GOAL_UNSATISFIED.value,
)
