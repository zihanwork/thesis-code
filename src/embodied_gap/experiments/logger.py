from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_gap.evaluation.metrics import EvaluationRecord
from embodied_gap.harness.controller import HarnessRun

from .config import ExperimentConfig
from .provenance import atomic_write_json


class ExperimentLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: ExperimentConfig) -> None:
        self._write_json("config.json", config.to_dict())

    def write_runs(self, runs: list[HarnessRun]) -> None:
        self._write_jsonl("runs.jsonl", [run.to_dict() for run in runs])

    def write_records(self, records: list[EvaluationRecord]) -> None:
        self._write_jsonl("metrics.jsonl", [record.to_dict() for record in records])

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json("summary.json", summary)

    def write_analysis(self, analysis: dict[str, Any]) -> None:
        self._write_json("analysis.json", analysis)

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.output_dir / filename, payload)

    def _write_jsonl(self, filename: str, rows: list[dict[str, Any]]) -> None:
        target = self.output_dir / filename
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(target)
