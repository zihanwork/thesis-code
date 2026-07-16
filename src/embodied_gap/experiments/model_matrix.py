from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from embodied_gap.analysis.make_tables import summary_to_markdown

from .config import ExperimentConfig
from .runner import ExperimentRunner


@dataclass(frozen=True)
class ModelSpec:
    id: str
    model: str
    provider: str = "one_api"
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            id=data["id"],
            model=data["model"],
            provider=data.get("provider", "one_api"),
            enabled=bool(data.get("enabled", True)),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "provider": self.provider,
            "enabled": self.enabled,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ModelMatrixConfig:
    name: str
    base_experiment: ExperimentConfig
    models: tuple[ModelSpec, ...]
    continue_on_error: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelMatrixConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        base_data = data["base_experiment"]
        return cls(
            name=data["name"],
            base_experiment=ExperimentConfig(
                name=base_data.get("name", data["name"]),
                tasks_path=base_data["tasks_path"],
                output_dir=base_data["output_dir"],
                retrieval_examples_path=base_data.get("retrieval_examples_path"),
                planners=tuple(base_data.get("planners", ExperimentConfig.planners)),
                harness_modes=tuple(
                    base_data.get("harness_modes", ExperimentConfig.harness_modes)
                ),
                seed=int(base_data.get("seed", 13)),
                max_retries=int(base_data.get("max_retries", 3)),
                llm_backend=base_data.get("llm_backend", "one_api"),
                use_llm_for_planners=bool(base_data.get("use_llm_for_planners", True)),
                llm_model=base_data.get("llm_model"),
                metadata=dict(base_data.get("metadata", {})),
            ),
            models=tuple(ModelSpec.from_dict(item) for item in data.get("models", [])),
            continue_on_error=bool(data.get("continue_on_error", True)),
        )


class MultiModelExperimentRunner:
    def __init__(self, config: ModelMatrixConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        enabled_models = [model for model in self.config.models if model.enabled]
        if not enabled_models:
            raise ValueError("Model matrix has no enabled models.")

        output_root = Path(self.config.base_experiment.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summaries: dict[str, Any] = {}

        for model in enabled_models:
            experiment_config = self._experiment_for_model(model, output_root)
            try:
                _, _, summary = ExperimentRunner(experiment_config).run()
            except Exception as exc:
                model_dir = Path(experiment_config.output_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                self._remove_stale_success_artifacts(model_dir)
                error_payload = {
                    "model": model.to_dict(),
                    "output_dir": experiment_config.output_dir,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                (model_dir / "error.json").write_text(
                    json.dumps(error_payload, indent=2, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                summaries[model.id] = error_payload
                if not self.config.continue_on_error:
                    raise
                continue
            else:
                model_dir = Path(experiment_config.output_dir)
                stale_error = model_dir / "error.json"
                if stale_error.exists():
                    stale_error.unlink()
                summaries[model.id] = {
                    "model": model.to_dict(),
                    "output_dir": experiment_config.output_dir,
                    "status": "succeeded",
                    "summary": summary,
                }
                (model_dir / "summary.md").write_text(
                    summary_to_markdown(summary),
                    encoding="utf-8",
                )

        matrix_summary = {
            "name": self.config.name,
            "model_count": len(enabled_models),
            "succeeded": sum(payload.get("status") == "succeeded" for payload in summaries.values()),
            "failed": sum(payload.get("status") == "failed" for payload in summaries.values()),
            "base_tasks_path": self.config.base_experiment.tasks_path,
            "models": summaries,
        }
        (output_root / "model_matrix_summary.json").write_text(
            json.dumps(matrix_summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return matrix_summary

    def _experiment_for_model(self, model: ModelSpec, output_root: Path) -> ExperimentConfig:
        base = self.config.base_experiment
        metadata = dict(base.metadata)
        metadata.update(
            {
                "model_matrix": self.config.name,
                "model_id": model.id,
                "model_name": model.model,
                "provider": model.provider,
            }
        )
        return ExperimentConfig(
            name=f"{base.name}_{model.id}",
            tasks_path=base.tasks_path,
            output_dir=str(output_root / model.id),
            retrieval_examples_path=base.retrieval_examples_path,
            planners=base.planners,
            harness_modes=base.harness_modes,
            seed=base.seed,
            max_retries=base.max_retries,
            llm_backend=base.llm_backend,
            use_llm_for_planners=base.use_llm_for_planners,
            llm_model=model.model,
            metadata=metadata,
        )

    def _remove_stale_success_artifacts(self, model_dir: Path) -> None:
        for filename in ("config.json", "runs.jsonl", "metrics.jsonl", "summary.json", "summary.md"):
            path = model_dir / filename
            if path.exists():
                path.unlink()
