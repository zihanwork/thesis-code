from __future__ import annotations

from embodied_gap.core.state_schema import WorldState
from embodied_gap.core.task_schema import Task


class GoalChecker:
    def partial_goal(self, task: Task, state: WorldState) -> float:
        return task.goal.satisfaction(state)

    def is_success(self, task: Task, state: WorldState) -> bool:
        return task.goal.is_satisfied(state)
