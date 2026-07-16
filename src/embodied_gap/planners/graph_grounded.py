from __future__ import annotations

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.knowledge.graph_store import ActionKnowledgeGraph
from embodied_gap.knowledge.pddl_grounded_search import PDDLGroundedSearch
from embodied_gap.llm.prompts import PLANNING_PROMPT_VERSION, render_planning_prompt


class GraphGroundedPlanner:
    """P2: model-independent symbolic planning reference."""

    name = "P2_symbolic_pddl"

    def __init__(
        self,
        graph: ActionKnowledgeGraph | None = None,
        pddl_search: PDDLGroundedSearch | None = None,
    ) -> None:
        self.graph = graph or ActionKnowledgeGraph()
        self.pddl_search = pddl_search or PDDLGroundedSearch()

    def plan(self, task: Task) -> PlanCandidate:
        prompt = render_planning_prompt(task, strategy="symbolic_reference")
        if not task.action_model and self.pddl_search.can_search(task):
            result = self.pddl_search.search(task)
            return PlanCandidate(
                planner_name=self.name,
                actions=result.actions,
                raw_response=str(list(result.actions)),
                prompt=prompt,
                metadata={
                    "planner_family": "symbolic_pddl",
                    "prompt_version": "p2_symbolic_v2",
                    "prompt_template_version": PLANNING_PROMPT_VERSION,
                    "engine": "pddl_grounded_search",
                    "model_independent": True,
                    "solved": result.solved,
                    "explored_states": result.explored_states,
                    "candidate_count": result.candidate_count,
                    "reason": result.reason,
                    "failure_memory_patterns": result.memory_patterns,
                },
            )

        result = self.graph.search(task)
        return PlanCandidate(
            planner_name=self.name,
            actions=result.actions,
            raw_response=str(list(result.actions)),
            prompt=prompt,
            metadata={
                "planner_family": "symbolic_action_model",
                "prompt_version": "p2_symbolic_v2",
                "prompt_template_version": PLANNING_PROMPT_VERSION,
                "engine": "symbolic_action_model_search",
                "model_independent": True,
                "solved": result.solved,
                "explored_states": result.explored_states,
                "reason": result.reason,
            },
        )
