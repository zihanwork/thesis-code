from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_gap.core.task_schema import dump_jsonl
from embodied_gap.evaluation.metrics import EvaluationRecord
from embodied_gap.harness.controller import HarnessRun

from .config import ExperimentConfig


class ExperimentLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: ExperimentConfig) -> None:
        self._write_json("config.json", config.to_dict())

    def write_runs(self, runs: list[HarnessRun]) -> None:
        dump_jsonl(self.output_dir / "runs.jsonl", [run.to_dict() for run in runs])

    def write_records(self, records: list[EvaluationRecord]) -> None:
        dump_jsonl(self.output_dir / "metrics.jsonl", [record.to_dict() for record in records])

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json("summary.json", summary)

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        (self.output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
