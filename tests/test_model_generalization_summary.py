from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from embodied_gap.analysis.model_generalization import (
    export_model_generalization_summary,
)


class ModelGeneralizationSummaryTests(unittest.TestCase):
    def test_exports_auditable_model_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "matrix"
            model_dir = root / "glm"
            model_dir.mkdir(parents=True)
            _write_json(
                root / "model_matrix_summary.json",
                {
                    "model_count": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "models": {"glm": {"status": "succeeded"}},
                },
            )
            _write_json(
                root / "run_manifest.json",
                {
                    "run_id": "run-unit",
                    "status": "succeeded",
                    "code": {"commit": "abc123", "dirty": True},
                    "config": {"sha256": "config-hash"},
                    "data": {
                        "tasks": {
                            "path": "dev.jsonl",
                            "sha256": "tasks-hash",
                            "evaluation_task_ids_sha256": "ids-hash",
                            "evaluation_task_count": 20,
                        },
                        "retrieval_examples": {
                            "path": "train.jsonl",
                            "sha256": "train-hash",
                        },
                    },
                },
            )
            comparison = {
                "left": "P0_engineered_prompt__H0_open_loop",
                "right": "P1_rag__H0_open_loop",
                "mcnemar": {"exact_two_sided_p_value": 0.01},
            }
            _write_json(
                model_dir / "analysis_v2.json",
                {
                    "schema_version": 2,
                    "methods": {"P1_rag__H0_open_loop": {"task_success_rate": 0.7}},
                    "paired_comparisons": {"p0_vs_p1": comparison},
                    "stratified": {
                        "dataset": {"virtualhome": {}},
                        "difficulty": {"easy": {}, "medium": {}, "hard": {}},
                    },
                },
            )
            _write_json(
                model_dir / "run_manifest.json",
                {
                    "status": "succeeded",
                    "telemetry": {
                        "client": {
                            "call_count": 2,
                            "successful_calls": 2,
                            "failed_calls": 0,
                            "prompt_tokens": 100,
                            "completion_tokens": 40,
                            "total_tokens": 140,
                            "latency_seconds": 3.5,
                            "cost_status": "pricing_not_configured",
                            "estimated_cost_usd": None,
                            "parameters": {"model": "GLM-5-Turbo"},
                            "calls": [
                                {"finish_reason": "stop"},
                                {"finish_reason": "length"},
                            ],
                        }
                    },
                },
            )

            output = Path(tmpdir) / "summary.json"
            report = export_model_generalization_summary(root, output)

            self.assertTrue(output.exists())
            self.assertTrue(report["source"]["dirty_worktree"])
            self.assertEqual(report["models"]["glm"]["model"], "GLM-5-Turbo")
            self.assertEqual(
                report["models"]["glm"]["telemetry"]["length_truncated_calls"],
                1,
            )
            self.assertEqual(report["models"]["glm"]["p0_vs_p1"], comparison)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
