from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_gap.core.task_schema import load_tasks
from embodied_gap.harness.recovery_policy import HarnessMode

from .model_matrix import ModelMatrixConfig


LLM_PLANNERS = {
    "B0_minimal_prompt",
    "P0_structured_prompt",
    "P0_engineered_prompt",
    "P1_rag",
    "P2_graph_rag",
    # Historical aliases remain inspectable for reproducibility.
    "P0_prompt_only",
    "P1_retrieval_augmented",
}

LLM_RECOVERY_MODES = {
    HarnessMode.H2_LLM_REFLECTION.value,
    HarnessMode.H2_ERROR_SPECIFIC.value,
    HarnessMode.H2_MEMORY.value,
    HarnessMode.H2_COMBINED.value,
    HarnessMode.H2_COMBINED_NO_LOCAL.value,
    HarnessMode.H2_COMBINED_NO_ERROR.value,
    HarnessMode.H2_COMBINED_NO_MEMORY.value,
}


def inspect_model_matrix(path: str | Path) -> dict[str, Any]:
    """Validate a model matrix and estimate its worst-case API-call budget."""

    config_path = Path(path)
    matrix = ModelMatrixConfig.from_json(config_path)
    base = matrix.base_experiment
    tasks_path = Path(base.tasks_path)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Task file does not exist: {tasks_path}")
    if base.retrieval_examples_path and not Path(base.retrieval_examples_path).exists():
        raise FileNotFoundError(
            f"Retrieval file does not exist: {base.retrieval_examples_path}"
        )
    if base.failure_memory_path and not Path(base.failure_memory_path).exists():
        raise FileNotFoundError(
            f"Failure-memory file does not exist: {base.failure_memory_path}"
        )
    if "P2_graph_rag" in base.planners and not Path(base.graph_path).exists():
        raise FileNotFoundError(f"GraphRAG edge file does not exist: {base.graph_path}")

    tasks = [task for task in load_tasks(tasks_path) if task.split != "train"]
    enabled_models = [model for model in matrix.models if model.enabled]
    llm_planners = [planner for planner in base.planners if planner in LLM_PLANNERS]
    llm_recovery_modes = [
        mode for mode in base.harness_modes if mode in LLM_RECOVERY_MODES
    ]

    initial_calls = (
        len(tasks) * len(llm_planners) * len(enabled_models)
        if base.use_llm_for_planners
        else 0
    )
    repair_calls = (
        len(tasks)
        * len(base.planners)
        * len(llm_recovery_modes)
        * base.max_retries
        * len(enabled_models)
        if base.use_llm_for_planners
        else 0
    )
    uses_frozen_heldout = "heldout" in str(tasks_path).lower()

    return {
        "config": str(config_path),
        "name": matrix.name,
        "task_file": str(tasks_path),
        "task_count": len(tasks),
        "model_count": len(enabled_models),
        "models": [model.model for model in enabled_models],
        "planner_count": len(base.planners),
        "planners": list(base.planners),
        "harness_count": len(base.harness_modes),
        "harness_modes": list(base.harness_modes),
        "run_record_count": (
            len(tasks)
            * len(base.planners)
            * len(base.harness_modes)
            * len(enabled_models)
        ),
        "initial_llm_call_count": initial_calls,
        "worst_case_repair_llm_call_count": repair_calls,
        "worst_case_total_llm_call_count": initial_calls + repair_calls,
        "uses_frozen_heldout": uses_frozen_heldout,
        "safe_for_development_selection": not uses_frozen_heldout,
        "pricing_configured_for_all_models": all(
            model.input_cost_per_million is not None
            and model.output_cost_per_million is not None
            for model in enabled_models
        ),
    }
