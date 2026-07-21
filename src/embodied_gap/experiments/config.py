from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    tasks_path: str
    output_dir: str
    retrieval_examples_path: str | None = None
    retrieval_method: str = "lexical"
    retrieval_field_profile: str = "instruction_state_goal_schema"
    retrieval_top_k: int = 1
    retrieval_min_score: float = 0.0
    graph_path: str = "data/knowledge/eai_train/kg_edges.jsonl"
    graph_top_k: int = 3
    failure_memory_path: str | None = None
    planners: tuple[str, ...] = (
        "P0_structured_prompt",
        "P0_engineered_prompt",
        "P1_rag",
    )
    harness_modes: tuple[str, ...] = (
        "H0_open_loop",
        "H2_local_recovery",
        "H2_llm_reflection",
        "H2_pddl_recovery",
    )
    seed: int = 13
    max_retries: int = 3
    llm_backend: str = "deterministic"
    use_llm_for_planners: bool = False
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 180
    llm_max_attempts: int = 4
    llm_backoff_seconds: float = 2.0
    llm_input_cost_per_million: float | None = None
    llm_output_cost_per_million: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            tasks_path=data["tasks_path"],
            output_dir=data["output_dir"],
            retrieval_examples_path=data.get("retrieval_examples_path"),
            retrieval_method=data.get("retrieval_method", "lexical"),
            retrieval_field_profile=data.get(
                "retrieval_field_profile", "instruction_state_goal_schema"
            ),
            retrieval_top_k=int(data.get("retrieval_top_k", 1)),
            retrieval_min_score=float(data.get("retrieval_min_score", 0.0)),
            graph_path=str(data.get("graph_path", "data/knowledge/eai_train/kg_edges.jsonl")),
            graph_top_k=int(data.get("graph_top_k", 3)),
            failure_memory_path=data.get("failure_memory_path"),
            planners=tuple(data.get("planners", cls.planners)),
            harness_modes=tuple(data.get("harness_modes", cls.harness_modes)),
            seed=int(data.get("seed", 13)),
            max_retries=int(data.get("max_retries", 3)),
            llm_backend=data.get("llm_backend", "deterministic"),
            use_llm_for_planners=bool(data.get("use_llm_for_planners", False)),
            llm_model=data.get("llm_model"),
            llm_temperature=float(data.get("llm_temperature", 0.0)),
            llm_max_tokens=int(data.get("llm_max_tokens", 2048)),
            llm_timeout_seconds=int(data.get("llm_timeout_seconds", 180)),
            llm_max_attempts=int(data.get("llm_max_attempts", 4)),
            llm_backoff_seconds=float(data.get("llm_backoff_seconds", 2.0)),
            llm_input_cost_per_million=_optional_float(
                data.get("llm_input_cost_per_million")
            ),
            llm_output_cost_per_million=_optional_float(
                data.get("llm_output_cost_per_million")
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tasks_path": self.tasks_path,
            "output_dir": self.output_dir,
            "retrieval_examples_path": self.retrieval_examples_path,
            "retrieval_method": self.retrieval_method,
            "retrieval_field_profile": self.retrieval_field_profile,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_min_score": self.retrieval_min_score,
            "graph_path": self.graph_path,
            "graph_top_k": self.graph_top_k,
            "failure_memory_path": self.failure_memory_path,
            "planners": list(self.planners),
            "harness_modes": list(self.harness_modes),
            "seed": self.seed,
            "max_retries": self.max_retries,
            "llm_backend": self.llm_backend,
            "use_llm_for_planners": self.use_llm_for_planners,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_max_attempts": self.llm_max_attempts,
            "llm_backoff_seconds": self.llm_backoff_seconds,
            "llm_input_cost_per_million": self.llm_input_cost_per_million,
            "llm_output_cost_per_million": self.llm_output_cost_per_million,
            "metadata": self.metadata,
        }


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None
