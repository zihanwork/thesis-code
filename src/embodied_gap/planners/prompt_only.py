from __future__ import annotations

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.llm.clients import LLMClient, last_call_metadata
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.prompts import PLANNING_PROMPT_VERSION, render_planning_prompt


class PromptOnlyPlanner:
    """P0: structured PDDL-informed prompt baseline.

    The offline implementation is deterministic and intentionally lacks external
    grounding. Real LLM calls can be plugged in behind the same interface.
    """

    name = "P0_structured_prompt"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        name: str | None = None,
        prompt_profile: str = "structured",
        prompt_version: str = "p0_structured_v2",
    ) -> None:
        self.llm_client = llm_client
        self.name = name or type(self).name
        self.prompt_profile = prompt_profile
        self.prompt_version = prompt_version

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

        prompt = self._render_prompt(task)
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=str(list(actions)),
            prompt=prompt,
            metadata={
                "planner_family": "structured_prompt",
                "prompt_version": self.prompt_version,
                "prompt_template_version": PLANNING_PROMPT_VERSION,
            },
        )

    def _llm_plan(self, task: Task) -> PlanCandidate:
        prompt = self._render_prompt(task)
        raw_response = self.llm_client.generate(prompt)
        call_metadata = last_call_metadata(self.llm_client)
        try:
            actions = parse_action_list(raw_response)
            metadata = {
                "planner_family": "structured_prompt",
                "prompt_version": self.prompt_version,
                "prompt_template_version": PLANNING_PROMPT_VERSION,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "llm_call": call_metadata,
            }
        except Exception as exc:  # noqa: BLE001 - parse failures are experimental observations.
            actions = ()
            metadata = {
                "planner_family": "structured_prompt",
                "prompt_version": self.prompt_version,
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

    def _render_prompt(self, task: Task) -> str:
        return render_planning_prompt(
            task,
            strategy=self.prompt_profile,
            profile=self.prompt_profile,
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


class MinimalPromptPlanner(PromptOnlyPlanner):
    """B0: minimal instruction/action-list prompt baseline."""

    name = "B0_minimal_prompt"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__(
            llm_client,
            name=self.name,
            prompt_profile="minimal",
            prompt_version="b0_minimal_v1",
        )


class EngineeredPromptPlanner(PromptOnlyPlanner):
    """P0-PE: structured prompt with an explicit constraint checklist."""

    name = "P0_engineered_prompt"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__(
            llm_client,
            name=self.name,
            prompt_profile="engineered",
            prompt_version="p0_engineered_v1",
        )
