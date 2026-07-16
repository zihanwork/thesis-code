from __future__ import annotations

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.knowledge.retriever import ExampleRetriever, RetrievedExample, adapt_plan
from embodied_gap.llm.clients import LLMClient, last_call_metadata
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.prompts import PLANNING_PROMPT_VERSION, render_planning_prompt

from .prompt_only import EngineeredPromptPlanner


class RetrievalAugmentedPlanner:
    """P1: retrieval-augmented initial planner."""

    name = "P1_rag"

    def __init__(
        self,
        examples: list[Task],
        min_score: float = 0.0,
        top_k: int = 1,
        retrieval_method: str = "lexical",
        field_profile: str = "instruction_state_goal_schema",
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retriever = ExampleRetriever(
            examples,
            method=retrieval_method,
            field_profile=field_profile,
        )
        self.min_score = min_score
        self.top_k = top_k
        self.retrieval_method = retrieval_method
        self.field_profile = field_profile
        self.fallback = EngineeredPromptPlanner()
        self.llm_client = llm_client

    def plan(self, task: Task) -> PlanCandidate:
        retrieved = [
            item
            for item in self.retriever.retrieve(task, k=self.top_k)
            if item.score >= self.min_score
        ]
        context = self._render_retrieval_context(retrieved)
        prompt = render_planning_prompt(
            task,
            strategy="retrieval_augmented",
            extra_context=context,
            profile="engineered",
        )
        if self.llm_client and retrieved:
            return self._llm_plan(prompt, retrieved)

        if not retrieved:
            fallback = self.fallback.plan(task)
            return PlanCandidate(
                planner_name=self.name,
                actions=fallback.actions,
                raw_response=fallback.raw_response,
                prompt=prompt,
                metadata={
                    "planner_family": "retrieval_augmented",
                    "prompt_version": "p1_rag_v2",
                    "prompt_template_version": PLANNING_PROMPT_VERSION,
                    "retrieved": None,
                    "fallback": fallback.planner_name,
                    "retrieval_method": self.retrieval_method,
                    "retrieval_field_profile": self.field_profile,
                    "retrieval_top_k": self.top_k,
                    "retrieval_min_score": self.min_score,
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
                "prompt_version": "p1_rag_v2",
                "prompt_template_version": PLANNING_PROMPT_VERSION,
                "retrieved": example.id,
                "retrieved_ids": [item.task.id for item in retrieved],
                "retrieval_score": retrieved[0].score,
                "retrieval_scores": [item.score for item in retrieved],
                "retrieval_components": [item.components for item in retrieved],
                "retrieval_corpus": "train_gold_plans",
                "retrieval_method": self.retrieval_method,
                "retrieval_field_profile": self.field_profile,
                "retrieval_top_k": self.top_k,
                "retrieval_min_score": self.min_score,
            },
        )

    def _llm_plan(self, prompt: str, retrieved: list[RetrievedExample]) -> PlanCandidate:
        raw_response = self.llm_client.generate(prompt)
        call_metadata = last_call_metadata(self.llm_client)
        try:
            actions = parse_action_list(raw_response)
            metadata = {
                "planner_family": "retrieval_augmented",
                "prompt_version": "p1_rag_v2",
                "prompt_template_version": PLANNING_PROMPT_VERSION,
                "retrieved": retrieved[0].task.id,
                "retrieved_ids": [item.task.id for item in retrieved],
                "retrieval_score": retrieved[0].score,
                "retrieval_scores": [item.score for item in retrieved],
                "retrieval_components": [item.components for item in retrieved],
                "retrieval_corpus": "train_gold_plans",
                "retrieval_method": self.retrieval_method,
                "retrieval_field_profile": self.field_profile,
                "retrieval_top_k": self.top_k,
                "retrieval_min_score": self.min_score,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "llm_call": call_metadata,
            }
        except Exception as exc:  # noqa: BLE001 - parse failures are experimental observations.
            actions = ()
            metadata = {
                "planner_family": "retrieval_augmented",
                "prompt_version": "p1_rag_v2",
                "prompt_template_version": PLANNING_PROMPT_VERSION,
                "retrieved": retrieved[0].task.id,
                "retrieved_ids": [item.task.id for item in retrieved],
                "retrieval_score": retrieved[0].score,
                "retrieval_scores": [item.score for item in retrieved],
                "retrieval_components": [item.components for item in retrieved],
                "retrieval_corpus": "train_gold_plans",
                "retrieval_method": self.retrieval_method,
                "retrieval_field_profile": self.field_profile,
                "retrieval_top_k": self.top_k,
                "retrieval_min_score": self.min_score,
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

    def _render_retrieval_context(self, retrieved: list[RetrievedExample]) -> str:
        if not retrieved:
            return "No retrieved examples above threshold."
        blocks = []
        for index, item in enumerate(retrieved, start=1):
            example = item.task
            blocks.append(
                "\n".join(
                    [
                        f"Retrieved demonstration {index}:",
                        f"Task ID: {example.id}",
                        f"Instruction: {example.instruction}",
                        f"Gold plan: {list(example.gold_plan)}",
                        f"Retrieval score: {item.score:.6f}",
                    ]
                )
            )
        return "\n\n".join(blocks)
