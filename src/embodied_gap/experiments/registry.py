from __future__ import annotations

from embodied_gap.core.task_schema import Task
from embodied_gap.harness.recovery_policy import HarnessMode
from embodied_gap.knowledge.graph_store import ActionKnowledgeGraph
from embodied_gap.llm.clients import OneAPIChatClient
from embodied_gap.planners.graph_grounded import GraphGroundedPlanner
from embodied_gap.planners.prompt_only import (
    EngineeredPromptPlanner,
    MinimalPromptPlanner,
    PromptOnlyPlanner,
)
from embodied_gap.planners.retrieval_augmented import RetrievalAugmentedPlanner


def build_planners(
    examples: list[Task],
    *,
    llm_backend: str = "deterministic",
    use_llm_for_planners: bool = False,
    llm_model: str | None = None,
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 2048,
    llm_timeout_seconds: int = 180,
    llm_max_attempts: int = 4,
    llm_backoff_seconds: float = 2.0,
    llm_input_cost_per_million: float | None = None,
    llm_output_cost_per_million: float | None = None,
    retrieval_method: str = "lexical",
    retrieval_field_profile: str = "instruction_state_goal_schema",
    retrieval_top_k: int = 1,
    retrieval_min_score: float = 0.0,
) -> dict[str, object]:
    graph = ActionKnowledgeGraph()
    llm_client = None
    if use_llm_for_planners:
        if llm_backend != "one_api":
            raise ValueError(f"Unsupported LLM backend for planners: {llm_backend}")
        llm_client = OneAPIChatClient.from_env(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout_seconds,
            max_attempts=llm_max_attempts,
            backoff_seconds=llm_backoff_seconds,
            input_cost_per_million=llm_input_cost_per_million,
            output_cost_per_million=llm_output_cost_per_million,
        )
    minimal = MinimalPromptPlanner(llm_client=llm_client)
    structured = PromptOnlyPlanner(llm_client=llm_client)
    engineered = EngineeredPromptPlanner(llm_client=llm_client)
    rag = RetrievalAugmentedPlanner(
        examples,
        min_score=retrieval_min_score,
        top_k=retrieval_top_k,
        retrieval_method=retrieval_method,
        field_profile=retrieval_field_profile,
        llm_client=llm_client,
    )
    symbolic = GraphGroundedPlanner(graph)
    return {
        "B0_minimal_prompt": minimal,
        "P0_structured_prompt": structured,
        "P0_engineered_prompt": engineered,
        "P1_rag": rag,
        "P2_symbolic_pddl": symbolic,
        # Legacy config aliases for historical pilot configuration files.
        "P0_prompt_only": structured,
        "P1_retrieval_augmented": rag,
        "P2_graph_grounded": symbolic,
    }


def parse_harness_modes(names: tuple[str, ...]) -> tuple[HarnessMode, ...]:
    return tuple(HarnessMode(name) for name in names)
