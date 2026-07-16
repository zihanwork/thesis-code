from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_gap.core.task_schema import load_tasks
from embodied_gap.evaluation.metrics import EvaluationRecord, aggregate_records, evaluate_run
from embodied_gap.harness.controller import HarnessController, HarnessRun
from embodied_gap.llm.clients import client_telemetry

from .config import ExperimentConfig
from .logger import ExperimentLogger
from .provenance import RunContext, make_run_id
from .registry import build_planners, parse_harness_modes


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self.config = config
        self._explicit_output_dir = Path(output_dir) if output_dir is not None else None
        self._requested_run_id = run_id
        self.parent_run_id = parent_run_id
        self.output_dir: Path | None = None
        self.run_id: str | None = None

    def run(self) -> tuple[list[HarnessRun], list[EvaluationRecord], dict[str, object]]:
        context = self._create_context()
        self.output_dir = context.output_dir
        self.run_id = context.run_id
        logger = ExperimentLogger(context.output_dir)
        logger.write_config(self.config)
        planner_registry: dict[str, object] = {}
        try:
            tasks = load_tasks(self.config.tasks_path)
            if self.config.retrieval_examples_path:
                examples = load_tasks(self.config.retrieval_examples_path)
            else:
                examples = [task for task in tasks if task.split == "train"]
            eval_tasks = [task for task in tasks if task.split != "train"]
            planner_registry = build_planners(
                examples,
                llm_backend=self.config.llm_backend,
                use_llm_for_planners=self.config.use_llm_for_planners,
                llm_model=self.config.llm_model,
                llm_temperature=self.config.llm_temperature,
                llm_max_tokens=self.config.llm_max_tokens,
                llm_timeout_seconds=self.config.llm_timeout_seconds,
                llm_max_attempts=self.config.llm_max_attempts,
                llm_backoff_seconds=self.config.llm_backoff_seconds,
                llm_input_cost_per_million=self.config.llm_input_cost_per_million,
                llm_output_cost_per_million=self.config.llm_output_cost_per_million,
            )
            harness_modes = parse_harness_modes(self.config.harness_modes)
            harness = HarnessController(max_retries=self.config.max_retries)

            runs: list[HarnessRun] = []
            records: list[EvaluationRecord] = []
            for task in eval_tasks:
                for planner_name in self.config.planners:
                    planner = planner_registry[planner_name]
                    initial_plan = planner.plan(task)
                    for mode in harness_modes:
                        run = harness.run(task, planner, mode, initial_plan=initial_plan)
                        runs.append(run)
                        records.append(evaluate_run(task, run))

            summary = aggregate_records(records)
            logger.write_runs(runs)
            logger.write_records(records)
            logger.write_summary(summary)
            context.finalize(
                "succeeded",
                results={
                    "evaluation_task_count": len(eval_tasks),
                    "run_record_count": len(runs),
                    "method_count": len(summary),
                    "summary": summary,
                    "artifacts": ["config.json", "runs.jsonl", "metrics.jsonl", "summary.json"],
                },
                telemetry=_planner_telemetry(planner_registry),
            )
            return runs, records, summary
        except Exception as exc:
            context.finalize(
                "failed",
                telemetry=_planner_telemetry(planner_registry),
                error=exc,
            )
            raise

    def _create_context(self) -> RunContext:
        config_payload = self.config.to_dict()
        if self._explicit_output_dir is not None:
            run_id = self._requested_run_id or make_run_id("explicit")
            return RunContext.create_at(
                self._explicit_output_dir,
                run_id=run_id,
                name=self.config.name,
                config=config_payload,
                tasks_path=self.config.tasks_path,
                retrieval_examples_path=self.config.retrieval_examples_path,
                models=self._model_manifest(),
                parent_run_id=self.parent_run_id,
            )
        return RunContext.create(
            self.config.output_dir,
            name=self.config.name,
            config=config_payload,
            tasks_path=self.config.tasks_path,
            retrieval_examples_path=self.config.retrieval_examples_path,
            models=self._model_manifest(),
        )

    def _model_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "provider": self.config.llm_backend,
                "model": self.config.llm_model,
                "use_llm_for_planners": self.config.use_llm_for_planners,
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens,
                "timeout_seconds": self.config.llm_timeout_seconds,
                "max_attempts": self.config.llm_max_attempts,
                "backoff_seconds": self.config.llm_backoff_seconds,
                "input_cost_per_million": self.config.llm_input_cost_per_million,
                "output_cost_per_million": self.config.llm_output_cost_per_million,
                "cost_currency": "USD",
            }
        ]


def _planner_telemetry(planners: dict[str, object]) -> dict[str, Any]:
    clients: dict[int, object] = {}
    for planner in planners.values():
        client = getattr(planner, "llm_client", None)
        if client is not None:
            clients[id(client)] = client
    payload: dict[str, Any] = {}
    for index, client in enumerate(clients.values(), start=1):
        telemetry = client_telemetry(client)
        parameters = telemetry.get("parameters", {})
        provider = parameters.get("provider", "unknown") if isinstance(parameters, dict) else "unknown"
        model = parameters.get("model", "unknown") if isinstance(parameters, dict) else "unknown"
        payload[f"{index}:{provider}:{model}"] = telemetry
    return payload
