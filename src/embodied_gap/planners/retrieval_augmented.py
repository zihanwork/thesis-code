from __future__ import annotations

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.knowledge.retriever import ExampleRetriever, adapt_plan
from embodied_gap.llm.clients import LLMClient
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.prompts import render_planning_prompt

from .prompt_only import PromptOnlyPlanner


class RetrievalAugmentedPlanner:
    """P1: retrieval-augmented initial planner."""

    name = "P1_retrieval_augmented"

    def __init__(
        self,
        examples: list[Task],
        min_score: float = 0.2,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retriever = ExampleRetriever(examples)
        self.min_score = min_score
        self.fallback = PromptOnlyPlanner()
        self.llm_client = llm_client

    def plan(self, task: Task) -> PlanCandidate:
        retrieved = self.retriever.retrieve(task, k=1)
        context = self._render_retrieval_context(retrieved)
        prompt = render_planning_prompt(
            task,
            strategy="retrieval_augmented",
            extra_context=context,
        )
        if self.llm_client and retrieved and retrieved[0].score >= self.min_score:
            return self._llm_plan(prompt, retrieved)

        if not retrieved or retrieved[0].score < self.min_score:
            fallback = self.fallback.plan(task)
            return PlanCandidate(
                planner_name=self.name,
                actions=fallback.actions,
                raw_response=fallback.raw_response,
                prompt=prompt,
                metadata={
                    "planner_family": "retrieval_augmented",
                    "retrieved": None,
                    "fallback": fallback.planner_name,
                },
            )

        example = retrieved[0].task
        actions = adapt_plan(example, task)
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=str(list(actions)),
            prompt=prompt,
            metadata={
                "planner_family": "retrieval_augmented",
                "retrieved": example.id,
                "retrieval_score": retrieved[0].score,
                "retrieval_corpus": "train_gold_plans",
            },
        )

    def _llm_plan(self, prompt: str, retrieved: list[object]) -> PlanCandidate:
        raw_response = self.llm_client.generate(prompt)
        try:
            actions = parse_action_list(raw_response)
            metadata = {
                "planner_family": "retrieval_augmented",
                "retrieved": retrieved[0].task.id,
                "retrieval_score": retrieved[0].score,
                "retrieval_corpus": "train_gold_plans",
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
            }
        except Exception as exc:  # noqa: BLE001 - parse failures are experimental observations.
            actions = ()
            metadata = {
                "planner_family": "retrieval_augmented",
                "retrieved": retrieved[0].task.id,
                "retrieval_score": retrieved[0].score,
                "retrieval_corpus": "train_gold_plans",
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "parse_error": str(exc),
            }
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=raw_response,
            prompt=prompt,
            metadata=metadata,
        )

    def _render_retrieval_context(self, retrieved: list[object]) -> str:
        if not retrieved:
            return "No retrieved examples above threshold."
        example = retrieved[0].task
        return "\n".join(
            [
                "Retrieved demonstration:",
                f"Task ID: {example.id}",
                f"Instruction: {example.instruction}",
                f"Gold plan: {list(example.gold_plan)}",
            ]
        )
