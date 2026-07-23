from __future__ import annotations

import json

from embodied_gap.core.patch_schema import PatchType, PlanPatch
from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.core.violation_schema import Violation
from embodied_gap.knowledge.failure_memory_store import FrozenFailureMemory
from embodied_gap.llm.clients import LLMClient, last_call_metadata
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.prompts import REPAIR_PROMPT_VERSION, render_repair_prompt


class LLMFeedbackRepair:
    """Replace a failed plan by asking the original model to use explicit feedback."""

    name = "llm_feedback_replan"

    def __init__(
        self,
        llm_client: LLMClient | None,
        *,
        failure_memory: FrozenFailureMemory | None = None,
        name: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.failure_memory = failure_memory or FrozenFailureMemory.empty()
        self.name = name or type(self).name

    def repair(self, task: Task, plan: PlanCandidate, violation: Violation | None) -> PlanPatch:
        if self.llm_client is None:
            return self._none(plan, "The initial planner has no LLM client.")

        feedback = json.dumps(
            violation.to_dict()
            if violation
            else {"type": "unknown_failure", "message": "The plan did not satisfy the goal."},
            ensure_ascii=False,
            sort_keys=True,
        )
        retrieved = self.failure_memory.retrieve(task, violation, k=1)
        memory_context = self.failure_memory.render_context(retrieved) if retrieved else ""
        prompt = render_repair_prompt(
            task,
            plan.actions,
            feedback,
            memory_context=memory_context,
        )
        raw_response = self.llm_client.generate(prompt)
        call_metadata = last_call_metadata(self.llm_client)
        try:
            actions = parse_action_list(raw_response)
        except Exception as exc:  # noqa: BLE001 - parse failure is an experiment outcome.
            return PlanPatch(
                patch_type=PatchType.NONE,
                source=self.name,
                before=plan.actions,
                after=plan.actions,
                explanation="The LLM repair response could not be parsed.",
                metadata={
                    "repair_prompt_version": REPAIR_PROMPT_VERSION,
                    "parse_error": str(exc),
                    "llm_call": call_metadata,
                    "failure_memory_sha256": self.failure_memory.sha256,
                    "retrieved_failure_memory": [item.entry.id for item in retrieved],
                },
            )
        if not actions or actions == plan.actions:
            return PlanPatch(
                patch_type=PatchType.NONE,
                source=self.name,
                before=plan.actions,
                after=plan.actions,
                explanation="The LLM did not produce a changed replacement plan.",
                metadata={
                    "repair_prompt_version": REPAIR_PROMPT_VERSION,
                    "llm_call": call_metadata,
                    "failure_memory_sha256": self.failure_memory.sha256,
                    "retrieved_failure_memory": [item.entry.id for item in retrieved],
                },
            )
        return PlanPatch(
            patch_type=PatchType.FULL_REPLAN,
            source=self.name,
            before=plan.actions,
            after=actions,
            explanation="The original model generated a replacement plan from validator feedback.",
            metadata={
                "repair_prompt_version": REPAIR_PROMPT_VERSION,
                "trigger_violation": violation.type.value if violation else None,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "llm_call": call_metadata,
                "failure_memory_sha256": self.failure_memory.sha256,
                "retrieved_failure_memory": [item.entry.id for item in retrieved],
                "retrieval_scores": [item.score for item in retrieved],
            },
        )

    def _none(self, plan: PlanCandidate, explanation: str) -> PlanPatch:
        return PlanPatch(
            patch_type=PatchType.NONE,
            source=self.name,
            before=plan.actions,
            after=plan.actions,
            explanation=explanation,
            metadata={"repair_prompt_version": REPAIR_PROMPT_VERSION},
        )


class MemoryAugmentedLLMRepair(LLMFeedbackRepair):
    name = "memory_augmented_llm_replan"

    def __init__(
        self,
        llm_client: LLMClient | None,
        failure_memory: FrozenFailureMemory,
    ) -> None:
        super().__init__(llm_client, failure_memory=failure_memory, name=self.name)
