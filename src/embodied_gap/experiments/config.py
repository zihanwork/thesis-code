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
    planners: tuple[str, ...] = ("P0_prompt_only", "P1_retrieval_augmented", "P2_graph_grounded")
    harness_modes: tuple[str, ...] = ("H0_open_loop", "H1_verifier_gated", "H2_full_recovery")
    seed: int = 13
    max_retries: int = 3
    llm_backend: str = "deterministic"
    use_llm_for_planners: bool = False
    llm_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            tasks_path=data["tasks_path"],
            output_dir=data["output_dir"],
            retrieval_examples_path=data.get("retrieval_examples_path"),
            planners=tuple(data.get("planners", cls.planners)),
            harness_modes=tuple(data.get("harness_modes", cls.harness_modes)),
            seed=int(data.get("seed", 13)),
            max_retries=int(data.get("max_retries", 3)),
            llm_backend=data.get("llm_backend", "deterministic"),
            use_llm_for_planners=bool(data.get("use_llm_for_planners", False)),
            llm_model=data.get("llm_model"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tasks_path": self.tasks_path,
            "output_dir": self.output_dir,
            "retrieval_examples_path": self.retrieval_examples_path,
            "planners": list(self.planners),
            "harness_modes": list(self.harness_modes),
            "seed": self.seed,
            "max_retries": self.max_retries,
            "llm_backend": self.llm_backend,
            "use_llm_for_planners": self.use_llm_for_planners,
            "llm_model": self.llm_model,
            "metadata": self.metadata,
        }
