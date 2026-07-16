from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from embodied_gap.core.action_schema import Action, Fact
from embodied_gap.core.state_schema import WorldState
from embodied_gap.core.task_schema import Task


@dataclass(frozen=True)
class GraphSearchResult:
    actions: tuple[Action, ...]
    explored_states: int
    solved: bool
    reason: str = ""


class ActionKnowledgeGraph:
    """Grounded object-action-state graph with precondition/effect transitions."""

    def search(self, task: Task, max_depth: int | None = None) -> GraphSearchResult:
        if task.safety_rules.should_reject:
            return GraphSearchResult(("reject()",), 0, True, "safety_rule_requires_rejection")

        max_depth = max_depth or max(len(task.allowed_actions), 1)
        initial = WorldState.from_facts(task.initial_facts)
        goals = set(task.goal_facts)
        queue: deque[tuple[WorldState, tuple[Action, ...]]] = deque([(initial, ())])
        visited = {initial.facts}
        explored = 0

        while queue:
            state, prefix = queue.popleft()
            explored += 1
            if goals.issubset(state.facts):
                return GraphSearchResult(prefix, explored, True, "goal_satisfied")
            if len(prefix) >= max_depth:
                continue

            for action in task.allowed_actions:
                if self._would_repeat_without_progress(action, prefix):
                    continue
                if action in task.safety_rules.forbidden_actions:
                    continue
                spec = task.action_model.get(action)
                if spec is None or not state.contains_all(spec.preconditions):
                    continue
                next_state = state.apply(spec)
                if next_state.facts in visited:
                    continue
                visited.add(next_state.facts)
                queue.append((next_state, prefix + (action,)))

        return GraphSearchResult((), explored, False, "search_exhausted")

    def actions_that_add(self, task: Task, fact: Fact) -> tuple[Action, ...]:
        return tuple(
            action
            for action, spec in task.action_model.items()
            if fact in spec.add_effects and action in task.allowed_actions
        )

    def _would_repeat_without_progress(self, action: Action, prefix: tuple[Action, ...]) -> bool:
        return action in prefix
