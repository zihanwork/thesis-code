from __future__ import annotations

from typing import Protocol

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task


class InitialPlanner(Protocol):
    """First-pass planner. It does not validate, execute, or repair plans."""

    name: str

    def plan(self, task: Task) -> PlanCandidate:
        """Generate an initial action sequence."""
