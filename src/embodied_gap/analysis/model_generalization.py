from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_gap.experiments.provenance import atomic_write_json


def export_model_generalization_summary(
    run_dir: str | Path,
    output_path: str | Path,
    *,
    analysis_filename: str = "analysis_v2.json",
) -> dict[str, Any]:
    """Export a compact, auditable summary from a completed model matrix."""

    root = Path(run_dir)
    matrix = _load_json(root / "model_matrix_summary.json")
    manifest = _load_json(root / "run_manifest.json")
    if matrix.get("failed"):
        raise ValueError("Model matrix contains failed models; summary is not complete.")

    models: dict[str, Any] = {}
    for model_id, model_result in sorted(matrix.get("models", {}).items()):
        if model_result.get("status") != "succeeded":
            raise ValueError(f"Model did not succeed: {model_id}")
        model_dir = root / model_id
        analysis_path = model_dir / analysis_filename
        if not analysis_path.exists():
            analysis_path = model_dir / "analysis.json"
        analysis = _load_json(analysis_path)
        child_manifest = _load_json(model_dir / "run_manifest.json")
        telemetry = _summarize_telemetry(child_manifest.get("telemetry", {}))
        models[model_id] = {
            "model": telemetry["model"],
            "run_status": child_manifest.get("status"),
            "analysis_schema_version": analysis.get("schema_version"),
            "methods": analysis.get("methods", {}),
            "p0_vs_p1": _p0_vs_p1(analysis.get("paired_comparisons", {})),
            "stratified": {
                "dataset": analysis.get("stratified", {}).get("dataset", {}),
                "difficulty": analysis.get("stratified", {}).get("difficulty", {}),
            },
            "telemetry": telemetry,
        }

    task_data = manifest.get("data", {}).get("tasks", {})
    retrieval_data = manifest.get("data", {}).get("retrieval_examples", {})
    report = {
        "schema_version": 1,
        "kind": "development_model_generalization_pilot",
        "claim_scope": (
            "Development-only evidence. Do not report as final held-out or official "
            "benchmark performance."
        ),
        "source": {
            "run_dir": str(root),
            "run_id": manifest.get("run_id"),
            "status": manifest.get("status"),
            "commit": manifest.get("code", {}).get("commit"),
            "dirty_worktree": manifest.get("code", {}).get("dirty"),
            "config_sha256": manifest.get("config", {}).get("sha256"),
            "tasks_path": task_data.get("path"),
            "tasks_sha256": task_data.get("sha256"),
            "evaluation_task_ids_sha256": task_data.get(
                "evaluation_task_ids_sha256"
            ),
            "evaluation_task_count": task_data.get("evaluation_task_count"),
            "retrieval_examples_path": retrieval_data.get("path"),
            "retrieval_examples_sha256": retrieval_data.get("sha256"),
        },
        "matrix": {
            "model_count": matrix.get("model_count"),
            "succeeded": matrix.get("succeeded"),
            "failed": matrix.get("failed"),
            "method_policy": "P0/H0 and P1/H0 only; no recovery calls.",
        },
        "models": models,
        "notes": {
            "confidence_intervals": "Wilson 95% intervals from analysis schema v2.",
            "paired_test": "Exact two-sided McNemar test on the same 20 tasks.",
            "cost": (
                "Token and latency telemetry are complete; monetary pricing was "
                "not configured."
            ),
            "truncation": "length_truncated_calls must be reported with success rates.",
        },
    }
    atomic_write_json(Path(output_path), report)
    return report


def _summarize_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    entries = [entry for entry in payload.values() if isinstance(entry, dict)]
    calls = [call for entry in entries for call in entry.get("calls", [])]
    parameters = entries[0].get("parameters", {}) if entries else {}
    return {
        "model": parameters.get("model", "unknown"),
        "call_count": sum(int(entry.get("call_count", 0)) for entry in entries),
        "successful_calls": sum(
            int(entry.get("successful_calls", 0)) for entry in entries
        ),
        "failed_calls": sum(int(entry.get("failed_calls", 0)) for entry in entries),
        "prompt_tokens": sum(int(entry.get("prompt_tokens", 0)) for entry in entries),
        "completion_tokens": sum(
            int(entry.get("completion_tokens", 0)) for entry in entries
        ),
        "total_tokens": sum(int(entry.get("total_tokens", 0)) for entry in entries),
        "latency_seconds": round(
            sum(float(entry.get("latency_seconds", 0.0)) for entry in entries), 6
        ),
        "length_truncated_calls": sum(
            call.get("finish_reason") == "length" for call in calls
        ),
        "cost_status": (
            entries[0].get("cost_status", "not_available")
            if entries
            else "not_available"
        ),
        "estimated_cost_usd": (
            sum(float(entry.get("estimated_cost_usd", 0.0)) for entry in entries)
            if entries
            and all(entry.get("estimated_cost_usd") is not None for entry in entries)
            else None
        ),
        "parameters": parameters,
    }


def _p0_vs_p1(comparisons: dict[str, Any]) -> dict[str, Any] | None:
    for comparison in comparisons.values():
        left = str(comparison.get("left", ""))
        right = str(comparison.get("right", ""))
        if left.startswith("P0_") and right.startswith("P1_"):
            return comparison
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
