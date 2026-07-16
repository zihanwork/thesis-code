from __future__ import annotations

from embodied_gap.core.task_schema import Task
from embodied_gap.harness.recovery_policy import HarnessMode
from embodied_gap.knowledge.graph_store import ActionKnowledgeGraph
from embodied_gap.llm.clients import OneAPIChatClient
from embodied_gap.planners.graph_grounded import GraphGroundedPlanner
from embodied_gap.planners.prompt_only import PromptOnlyPlanner
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
    return {
        "P0_prompt_only": PromptOnlyPlanner(llm_client=llm_client),
        "P1_retrieval_augmented": RetrievalAugmentedPlanner(examples, llm_client=llm_client),
        "P2_graph_grounded": GraphGroundedPlanner(graph),
    }


def parse_harness_modes(names: tuple[str, ...]) -> tuple[HarnessMode, ...]:
    return tuple(HarnessMode(name) for name in names)
