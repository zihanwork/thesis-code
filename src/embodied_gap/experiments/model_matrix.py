from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from embodied_gap.analysis.make_tables import summary_to_markdown

from .config import ExperimentConfig
from .provenance import RunContext, atomic_write_json
from .runner import ExperimentRunner


@dataclass(frozen=True)
class ModelSpec:
    id: str
    model: str
    provider: str = "one_api"
    enabled: bool = True
    notes: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: int | None = None
    max_attempts: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            id=data["id"],
            model=data["model"],
            provider=data.get("provider", "one_api"),
            enabled=bool(data.get("enabled", True)),
            notes=data.get("notes", ""),
            temperature=_optional_float(data.get("temperature")),
            max_tokens=_optional_int(data.get("max_tokens")),
            timeout_seconds=_optional_int(data.get("timeout_seconds")),
            max_attempts=_optional_int(data.get("max_attempts")),
            input_cost_per_million=_optional_float(data.get("input_cost_per_million")),
            output_cost_per_million=_optional_float(data.get("output_cost_per_million")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "provider": self.provider,
            "enabled": self.enabled,
            "notes": self.notes,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
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
                retrieval_method=base_data.get("retrieval_method", "lexical"),
                retrieval_field_profile=base_data.get(
                    "retrieval_field_profile", "instruction_state_goal_schema"
                ),
                retrieval_top_k=int(base_data.get("retrieval_top_k", 1)),
                retrieval_min_score=float(base_data.get("retrieval_min_score", 0.0)),
                graph_path=str(base_data.get("graph_path", "data/knowledge/eai_train/kg_edges.jsonl")),
                graph_top_k=int(base_data.get("graph_top_k", 1)),
                failure_memory_path=base_data.get("failure_memory_path"),
                planners=tuple(base_data.get("planners", ExperimentConfig.planners)),
                harness_modes=tuple(
                    base_data.get("harness_modes", ExperimentConfig.harness_modes)
                ),
                seed=int(base_data.get("seed", 13)),
                max_retries=int(base_data.get("max_retries", 3)),
                llm_backend=base_data.get("llm_backend", "one_api"),
                use_llm_for_planners=bool(base_data.get("use_llm_for_planners", True)),
                llm_model=base_data.get("llm_model"),
                llm_temperature=float(base_data.get("llm_temperature", 0.0)),
                llm_max_tokens=int(base_data.get("llm_max_tokens", 2048)),
                llm_timeout_seconds=int(base_data.get("llm_timeout_seconds", 180)),
                llm_max_attempts=int(base_data.get("llm_max_attempts", 4)),
                llm_backoff_seconds=float(base_data.get("llm_backoff_seconds", 2.0)),
                llm_input_cost_per_million=_optional_float(
                    base_data.get("llm_input_cost_per_million")
                ),
                llm_output_cost_per_million=_optional_float(
                    base_data.get("llm_output_cost_per_million")
                ),
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

        context = RunContext.create(
            self.config.base_experiment.output_dir,
            name=self.config.name,
            config={
                "name": self.config.name,
                "continue_on_error": self.config.continue_on_error,
                "base_experiment": self.config.base_experiment.to_dict(),
                "models": [model.to_dict() for model in enabled_models],
            },
            tasks_path=self.config.base_experiment.tasks_path,
            retrieval_examples_path=self.config.base_experiment.retrieval_examples_path,
            failure_memory_path=self.config.base_experiment.failure_memory_path,
            models=[model.to_dict() for model in enabled_models],
        )
        output_root = context.output_dir
        summaries: dict[str, Any] = {}
        try:
            for model in enabled_models:
                experiment_config = self._experiment_for_model(model, output_root)
                model_dir = output_root / model.id
                runner = ExperimentRunner(
                    experiment_config,
                    output_dir=model_dir,
                    run_id=f"{context.run_id}__{model.id}",
                    parent_run_id=context.run_id,
                )
                try:
                    _, _, summary = runner.run()
                except Exception as exc:
                    error_payload = {
                        "model": model.to_dict(),
                        "output_dir": str(model_dir),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    atomic_write_json(model_dir / "error.json", error_payload)
                    summaries[model.id] = error_payload
                    context.update(model_results=summaries)
                    if not self.config.continue_on_error:
                        raise
                    continue
                summaries[model.id] = {
                    "model": model.to_dict(),
                    "output_dir": str(model_dir),
                    "status": "succeeded",
                    "summary": summary,
                }
                (model_dir / "summary.md").write_text(
                    summary_to_markdown(summary),
                    encoding="utf-8",
                )
                context.update(model_results=summaries)

            matrix_summary = {
                "name": self.config.name,
                "run_id": context.run_id,
                "output_dir": str(output_root),
                "model_count": len(enabled_models),
                "succeeded": sum(
                    payload.get("status") == "succeeded" for payload in summaries.values()
                ),
                "failed": sum(
                    payload.get("status") == "failed" for payload in summaries.values()
                ),
                "base_tasks_path": self.config.base_experiment.tasks_path,
                "models": summaries,
            }
            atomic_write_json(output_root / "model_matrix_summary.json", matrix_summary)
            final_status = "succeeded" if matrix_summary["failed"] == 0 else "partial"
            context.finalize(final_status, results=matrix_summary)
            return matrix_summary
        except Exception as exc:
            context.finalize("failed", results={"models": summaries}, error=exc)
            raise

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
            retrieval_method=base.retrieval_method,
            retrieval_field_profile=base.retrieval_field_profile,
            retrieval_top_k=base.retrieval_top_k,
            retrieval_min_score=base.retrieval_min_score,
            failure_memory_path=base.failure_memory_path,
            planners=base.planners,
            harness_modes=base.harness_modes,
            seed=base.seed,
            max_retries=base.max_retries,
            llm_backend=base.llm_backend,
            use_llm_for_planners=base.use_llm_for_planners,
            llm_model=model.model,
            llm_temperature=(
                model.temperature if model.temperature is not None else base.llm_temperature
            ),
            llm_max_tokens=(
                model.max_tokens if model.max_tokens is not None else base.llm_max_tokens
            ),
            llm_timeout_seconds=(
                model.timeout_seconds
                if model.timeout_seconds is not None
                else base.llm_timeout_seconds
            ),
            llm_max_attempts=(
                model.max_attempts
                if model.max_attempts is not None
                else base.llm_max_attempts
            ),
            llm_backoff_seconds=base.llm_backoff_seconds,
            llm_input_cost_per_million=(
                model.input_cost_per_million
                if model.input_cost_per_million is not None
                else base.llm_input_cost_per_million
            ),
            llm_output_cost_per_million=(
                model.output_cost_per_million
                if model.output_cost_per_million is not None
                else base.llm_output_cost_per_million
            ),
            metadata=metadata,
        )


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
