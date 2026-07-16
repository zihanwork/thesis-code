from __future__ import annotations

import time

from embodied_gap.core.patch_schema import PatchType, PlanPatch
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation
from embodied_gap.knowledge.graph_store import ActionKnowledgeGraph
from embodied_gap.knowledge.pddl_grounded_search import PDDLGroundedSearch


class FullReplanRepair:
    name = "symbolic_replan"

    def __init__(
        self,
        graph: ActionKnowledgeGraph | None = None,
        pddl_search: PDDLGroundedSearch | None = None,
    ) -> None:
        self.graph = graph or ActionKnowledgeGraph()
        self.pddl_search = pddl_search or PDDLGroundedSearch()

    def repair(self, task: Task, plan: PlanCandidate, violation: Violation | None) -> PlanPatch:
        if self.pddl_search.can_search(task):
            started = time.perf_counter()
            result = self.pddl_search.search(task)
            search_seconds = time.perf_counter() - started
            if result.solved:
                return PlanPatch(
                    patch_type=PatchType.FULL_REPLAN,
                    source=self.name,
                    before=plan.actions,
                    after=result.actions,
                    explanation="Replaced the failed plan with a PDDL-grounded plan.",
                    metadata={
                        "engine": "pddl_grounded_search",
                        "explored_states": result.explored_states,
                        "search_seconds": round(search_seconds, 6),
                        "candidate_count": result.candidate_count,
                        "reason": result.reason,
                        "failure_memory_patterns": result.memory_patterns,
                        "trigger_violation": violation.type.value if violation else None,
                    },
                )

        started = time.perf_counter()
        result = self.graph.search(task)
        search_seconds = time.perf_counter() - started
        if not result.solved:
            return PlanPatch(
                patch_type=PatchType.NONE,
                source=self.name,
                before=plan.actions,
                after=plan.actions,
                explanation="Graph search could not find a valid repair.",
                metadata={
                    "explored_states": result.explored_states,
                    "search_seconds": round(search_seconds, 6),
                    "reason": result.reason,
                },
            )
        return PlanPatch(
            patch_type=PatchType.FULL_REPLAN,
            source=self.name,
            before=plan.actions,
            after=result.actions,
            explanation="Replaced the failed plan with a symbolic action-model plan.",
            metadata={
                "engine": "symbolic_action_model_search",
                "explored_states": result.explored_states,
                "search_seconds": round(search_seconds, 6),
                "reason": result.reason,
            },
        )
