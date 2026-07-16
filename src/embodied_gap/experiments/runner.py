from __future__ import annotations

from embodied_gap.core.task_schema import load_tasks
from embodied_gap.evaluation.metrics import EvaluationRecord, aggregate_records, evaluate_run
from embodied_gap.harness.controller import HarnessController, HarnessRun

from .config import ExperimentConfig
from .logger import ExperimentLogger
from .registry import build_planners, parse_harness_modes


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> tuple[list[HarnessRun], list[EvaluationRecord], dict[str, object]]:
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
        logger = ExperimentLogger(self.config.output_dir)
        logger.write_config(self.config)
        logger.write_runs(runs)
        logger.write_records(records)
        logger.write_summary(summary)
        return runs, records, summary
