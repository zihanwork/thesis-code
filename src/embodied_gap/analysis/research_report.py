from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import random
from typing import Any


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def exact_mcnemar(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    left_only = sum(left[key] and not right[key] for key in shared)
    right_only = sum(right[key] and not left[key] for key in shared)
    discordant = left_only + right_only
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "paired_task_count": len(shared),
        "left_only_success": left_only,
        "right_only_success": right_only,
        "discordant_count": discordant,
        "exact_two_sided_p_value": p_value,
    }


def paired_uplift_interval(
    left: dict[str, bool],
    right: dict[str, bool],
    *,
    samples: int = 2000,
    seed: int = 13,
) -> dict[str, float | int]:
    shared = sorted(set(left) & set(right))
    differences = [int(right[key]) - int(left[key]) for key in shared]
    if not differences:
        return {"paired_task_count": 0, "uplift": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    uplift = sum(differences) / len(differences)
    generator = random.Random(seed)
    bootstrapped = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    low_index = max(0, int(0.025 * samples) - 1)
    high_index = min(samples - 1, int(0.975 * samples))
    return {
        "paired_task_count": len(shared),
        "uplift": uplift,
        "ci95_low": bootstrapped[low_index],
        "ci95_high": bootstrapped[high_index],
    }


def build_research_analysis(
    metric_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_method[str(row["method_id"])].append(row)

    methods = {
        method: _summarize_metric_rows(rows)
        for method, rows in sorted(by_method.items())
    }
    outcomes = {
        method: {str(row["task_id"]): bool(row["task_success"]) for row in rows}
        for method, rows in by_method.items()
    }
    comparisons: dict[str, Any] = {}
    for left, right in combinations(sorted(outcomes), 2):
        comparison_id = f"{left}__vs__{right}"
        comparisons[comparison_id] = {
            "left": left,
            "right": right,
            "mcnemar": exact_mcnemar(outcomes[left], outcomes[right]),
            "paired_task_success_uplift": paired_uplift_interval(
                outcomes[left], outcomes[right]
            ),
        }

    return {
        "schema_version": 1,
        "record_count": len(metric_rows),
        "method_count": len(methods),
        "methods": methods,
        "stratified": {
            field: _stratify(by_method, field)
            for field in ("dataset", "difficulty", "task_family")
        },
        "paired_comparisons": comparisons,
        "cost_and_search": _summarize_run_costs(run_rows or [], outcomes),
        "notes": {
            "confidence_interval": "Wilson score interval for binary rates.",
            "paired_uplift_interval": "Task-paired percentile bootstrap with seed 13 and 2000 samples.",
            "mcnemar": "Exact two-sided binomial McNemar test.",
            "cost_scope": "Per-method standalone attribution; an initial planner call reused across harness rows is attributed to each method row.",
        },
    }


def export_research_analysis(
    *,
    metrics_path: str | Path,
    output_path: str | Path,
    runs_path: str | Path | None = None,
) -> dict[str, Any]:
    metric_rows = _load_jsonl(metrics_path)
    run_rows = _load_jsonl(runs_path) if runs_path else []
    report = build_research_analysis(metric_rows, run_rows)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success_count = sum(bool(row.get("task_success")) for row in rows)
    safe_count = sum(bool(row.get("safe_success")) for row in rows)
    execution_count = sum(bool(row.get("execution_success")) for row in rows)
    errors: dict[str, int] = defaultdict(int)
    for row in rows:
        for error_type, count in row.get("error_counts", {}).items():
            errors[str(error_type)] += int(count)
    success_ci = wilson_interval(success_count, total)
    safe_ci = wilson_interval(safe_count, total)
    execution_ci = wilson_interval(execution_count, total)
    return {
        "n": total,
        "task_success_count": success_count,
        "task_success_rate": success_count / total if total else 0.0,
        "task_success_ci95": list(success_ci),
        "safe_success_rate": safe_count / total if total else 0.0,
        "safe_success_ci95": list(safe_ci),
        "execution_success_rate": execution_count / total if total else 0.0,
        "execution_success_ci95": list(execution_ci),
        "risk_rate": sum(bool(row.get("risk")) for row in rows) / total if total else 0.0,
        "average_attempts": _average(rows, "attempts"),
        "average_repairs": _average(rows, "patch_count"),
        "failure_type_counts": dict(sorted(errors.items())),
    }


def _stratify(
    by_method: dict[str, list[dict[str, Any]]],
    field: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    values = sorted(
        {
            str(row.get("metadata", {}).get(field) or "unknown")
            for rows in by_method.values()
            for row in rows
        }
    )
    for value in values:
        result[value] = {
            method: _summarize_metric_rows(
                [
                    row
                    for row in rows
                    if str(row.get("metadata", {}).get(field) or "unknown") == value
                ]
            )
            for method, rows in sorted(by_method.items())
            if any(
                str(row.get("metadata", {}).get(field) or "unknown") == value
                for row in rows
            )
        }
    return result


def _summarize_run_costs(
    run_rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row.get("method_id", "unknown"))].append(row)
    result: dict[str, Any] = {}
    for method, rows in sorted(grouped.items()):
        calls = [call for row in rows for call in _llm_calls(row)]
        priced = [call.get("estimated_cost_usd") for call in calls]
        costs_available = bool(calls) and all(value is not None for value in priced)
        success_count = sum(
            outcomes.get(method, {}).get(str(row.get("task_id")), False) for row in rows
        )
        explored, search_seconds = _symbolic_search_metrics(rows)
        total_cost = sum(float(value) for value in priced if value is not None)
        result[method] = {
            "llm_call_count": len(calls),
            "prompt_tokens": sum(int(call.get("prompt_tokens", 0)) for call in calls),
            "completion_tokens": sum(int(call.get("completion_tokens", 0)) for call in calls),
            "total_tokens": sum(int(call.get("total_tokens", 0)) for call in calls),
            "latency_seconds": sum(float(call.get("latency_seconds", 0.0)) for call in calls),
            "estimated_cost_usd": total_cost if costs_available else None,
            "cost_status": (
                "estimated" if costs_available else "not_applicable" if not calls else "pricing_not_configured"
            ),
            "estimated_cost_per_success_usd": (
                total_cost / success_count if costs_available and success_count else None
            ),
            "average_repairs": (
                sum(len(row.get("patches", [])) for row in rows) / len(rows) if rows else 0.0
            ),
            "symbolic_explored_states": explored,
            "symbolic_search_seconds": search_seconds,
        }
    return result


def _llm_calls(run: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    initial = run.get("initial_plan", {}).get("metadata", {}).get("llm_call")
    if isinstance(initial, dict) and initial:
        calls.append(initial)
    for patch in run.get("patches", []):
        call = patch.get("metadata", {}).get("llm_call")
        if isinstance(call, dict) and call:
            calls.append(call)
    return calls


def _symbolic_search_metrics(rows: list[dict[str, Any]]) -> tuple[int, float]:
    explored = 0
    seconds = 0.0
    for row in rows:
        metadata_items = [row.get("initial_plan", {}).get("metadata", {})]
        metadata_items.extend(patch.get("metadata", {}) for patch in row.get("patches", []))
        for metadata in metadata_items:
            explored += int(metadata.get("explored_states", 0))
            seconds += float(metadata.get("search_seconds", 0.0))
    return explored, round(seconds, 6)


def _average(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / len(rows) if rows else 0.0


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
