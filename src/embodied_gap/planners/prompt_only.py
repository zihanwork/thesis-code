from __future__ import annotations

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.llm.clients import LLMClient, last_call_metadata
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.prompts import PLANNING_PROMPT_VERSION, render_planning_prompt


class PromptOnlyPlanner:
    """P0: structured prompt-only planning baseline.

    The offline implementation is deterministic and intentionally lacks external
    grounding. Real LLM calls can be plugged in behind the same interface.
    """

    name = "P0_prompt_only"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def plan(self, task: Task) -> PlanCandidate:
        if self.llm_client:
            return self._llm_plan(task)

        obj = task.slots.get("object", "object")
        destination = task.slots.get("destination")
        source = task.slots.get("source")
        tool = task.slots.get("tool")
        container = task.slots.get("container")

        if "hazard" in task.tags:
            actions = self._unsafe_literal_plan(task)
        elif "move" in task.tags and destination:
            actions = (
                f"pickup({obj})",
                f"navigate({destination})",
                f"put({obj}, {destination})",
            )
        elif "clean" in task.tags and tool:
            actions = (
                f"pickup({obj})",
                f"navigate({tool})",
                f"clean({obj})",
            )
        elif source and destination:
            actions = (
                f"navigate({source})",
                f"pickup({obj})",
                f"navigate({destination})",
                f"put({obj}, {destination})",
            )
        else:
            actions = tuple(task.allowed_actions[:3])

        prompt = render_planning_prompt(task, strategy="structured_prompt_only")
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=str(list(actions)),
            prompt=prompt,
            metadata={
                "planner_family": "prompt_only",
                "prompt_version": "p0_v1",
                "prompt_template_version": PLANNING_PROMPT_VERSION,
            },
        )

    def _llm_plan(self, task: Task) -> PlanCandidate:
        prompt = render_planning_prompt(task, strategy="structured_prompt_only")
        raw_response = self.llm_client.generate(prompt)
        call_metadata = last_call_metadata(self.llm_client)
        try:
            actions = parse_action_list(raw_response)
            metadata = {
                "planner_family": "prompt_only",
                "prompt_version": "p0_v1",
                "prompt_template_version": PLANNING_PROMPT_VERSION,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "llm_call": call_metadata,
            }
        except Exception as exc:  # noqa: BLE001 - parse failures are experimental observations.
            actions = ()
            metadata = {
                "planner_family": "prompt_only",
                "prompt_version": "p0_v1",
                "prompt_template_version": PLANNING_PROMPT_VERSION,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "llm_call": call_metadata,
                "parse_error": str(exc),
            }
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=raw_response,
            prompt=prompt,
            metadata=metadata,
        )

    def _unsafe_literal_plan(self, task: Task) -> tuple[str, ...]:
        obj = task.slots.get("object", "object")
        container = task.slots.get("container", "container")
        return (
            "navigate(desk)",
            f"pickup({obj})",
            f"navigate({container})",
            f"open({container})",
            f"put({obj}, {container})",
            f"close({container})",
            f"turn_on({container})",
        )
