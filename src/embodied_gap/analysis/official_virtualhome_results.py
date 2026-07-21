from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from embodied_gap.analysis.research_report import exact_mcnemar
from embodied_gap.analysis.thesis_validity import family_clustered_paired_analysis


_MODEL_LINE = re.compile(r"Model name is (.+)$")
_TASK_LINE = re.compile(r"Task is .+, file_id is (\S+)$")
_SUCCESS_LINE = re.compile(r"Goals all satisfied: all_pred_success=(True|False)$")


MODEL_IDS = ("deepseek_v4_flash", "gpt_5_5", "glm_5_turbo")
PLANNER_IDS = (
    "B0_minimal_prompt",
    "P0_structured_prompt",
    "P0_engineered_prompt",
    "P1_rag",
    "P2_graph_rag",
)
HARNESS_IDS = (
    "H0_open_loop",
    "H2_llm_reflection",
    "H2_memory",
    "H2_pddl_recovery",
)


def _build_key_contrasts() -> dict[str, tuple[str, str]]:
    contrasts: dict[str, tuple[str, str]] = {}
    for model in MODEL_IDS:
        for harness in HARNESS_IDS:
            for left_planner, right_planner in zip(PLANNER_IDS[:-1], PLANNER_IDS[1:], strict=True):
                contrast_id = f"{model}__{harness}__{left_planner}_to_{right_planner}"
                contrasts[contrast_id] = (
                    f"{model}__{left_planner}__{harness}",
                    f"{model}__{right_planner}__{harness}",
                )
        for planner in PLANNER_IDS:
            for recovery in HARNESS_IDS[1:]:
                contrast_id = f"{model}__{planner}__H0_to_{recovery}"
                contrasts[contrast_id] = (
                    f"{model}__{planner}__H0_open_loop",
                    f"{model}__{planner}__{recovery}",
                )
    return contrasts


KEY_CONTRASTS = _build_key_contrasts()


def parse_official_task_outcomes(
    log_path: str | Path,
    identifiers: list[str],
) -> dict[str, dict[str, bool]]:
    """Recover per-task official success from the pinned evaluator log."""

    expected = set(identifiers)
    outcomes: dict[str, dict[str, bool]] = {}
    current_model: str | None = None
    current_identifier: str | None = None
    for raw_line in Path(log_path).read_text(encoding="utf-8").splitlines():
        message = raw_line.rsplit(" - ", 1)[-1]
        if match := _MODEL_LINE.search(message):
            current_model = match.group(1)
            outcomes[current_model] = {identifier: False for identifier in identifiers}
            current_identifier = None
            continue
        if match := _TASK_LINE.search(message):
            current_identifier = match.group(1)
            continue
        if match := _SUCCESS_LINE.search(message):
            if current_model is not None and current_identifier in expected:
                outcomes[current_model][current_identifier] = match.group(1) == "True"
    return outcomes


def export_official_results_report(
    *,
    results_root: str | Path,
    cohort_path: str | Path,
    evaluator_log_path: str | Path,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
) -> dict[str, Any]:
    results_root = Path(results_root)
    cohort = [
        json.loads(line)
        for line in Path(cohort_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    family_by_identifier = {
        str(task["slots"]["file_id"]): str(task["slots"].get("task_family", "unknown"))
        for task in cohort
    }
    identifiers = sorted(family_by_identifier)
    outcomes = parse_official_task_outcomes(evaluator_log_path, identifiers)

    cells: dict[str, Any] = {}
    for summary_path in sorted(results_root.glob("*/summary.json")):
        cell_id = summary_path.parent.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task_outcomes = outcomes.get(cell_id, {})
        cells[cell_id] = {
            "official_summary_path": str(summary_path),
            "task_success_count": sum(task_outcomes.values()),
            "cohort_task_count": len(identifiers),
            "goal_evaluation": summary["goal_evaluation"],
            "trajectory_evaluation": summary["trajectory_evaluation"],
        }

    comparisons: dict[str, Any] = {}
    for comparison_id, (left_id, right_id) in KEY_CONTRASTS.items():
        if left_id not in outcomes or right_id not in outcomes:
            continue
        left = outcomes[left_id]
        right = outcomes[right_id]
        left_rows = [
            {
                "task_id": identifier,
                "task_success": left[identifier],
                "metadata": {"task_family": family_by_identifier[identifier]},
            }
            for identifier in identifiers
        ]
        right_rows = [
            {
                "task_id": identifier,
                "task_success": right[identifier],
                "metadata": {"task_family": family_by_identifier[identifier]},
            }
            for identifier in identifiers
        ]
        comparisons[comparison_id] = {
            "left": left_id,
            "right": right_id,
            "mcnemar": exact_mcnemar(left, right),
            "family_clustered": family_clustered_paired_analysis(
                left_rows,
                right_rows,
                samples=10_000,
                seed=13,
            ),
        }

    planner_ids = {cell_id.split("__")[1] for cell_id in cells}
    harness_ids = {cell_id.split("__")[2] for cell_id in cells}
    scope_claim = (
        "This report is a post-hoc replacement replication on the observed cohort; "
        "it supports same-cohort P1-to-P2 comparisons, not untouched confirmatory claims."
        if planner_ids == {"P1_rag", "P2_graph_rag"}
        else "The complete 5 x 4 planner-harness grid supports model-stratified planning, recovery, and interaction analyses."
    )
    grid_claim = (
        f"The reported grid contains {len(planner_ids)} planners and {len(harness_ids)} harnesses."
    )
    report = {
        "cohort": {

            "path": str(cohort_path),
            "task_count": len(identifiers),
            "task_family_count": len(set(family_by_identifier.values())),
            "selection": (
                "Pre-outcome compatibility screen using gold plans, official action support, "
                "and unambiguous task-specific object IDs."
            ),
        },
        "evaluator_log_path": str(evaluator_log_path),
        "cells": cells,
        "paired_comparisons": comparisons,
        "claim_boundaries": [
            "All reported outcome scores come from the pinned official evaluator.",
            "The local verifier is an intervention component, not an outcome authority.",
            "The 84-task cohort is not the complete official hidden challenge set.",
            scope_claim,
            grid_claim,
        ],
    }

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(output_markdown_path).write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Official VirtualHome Action Sequencing Results",
        "",
        "## Scope",
        "",
        f"The single outcome authority is the pinned official evaluator. The fixed cohort contains {report['cohort']['task_count']} tasks across {report['cohort']['task_family_count']} task families. Local execution checks are used only inside the verifier/recovery mechanism.",
        "",
        "## Official results",
        "",
        "| Cell | Task success | Total goal | Execution success | Parsing error |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell_id, cell in sorted(report["cells"].items()):
        goal = cell["goal_evaluation"]
        trajectory = cell["trajectory_evaluation"]
        lines.append(
            f"| `{cell_id}` | {goal['task_success_rate']:.3f}% | {goal['total_goal']:.3f}% | {trajectory['execution_success_rate']:.3f}% | {trajectory['grammar_error']['parsing']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Family-clustered paired comparisons",
            "",
            "| Contrast | Uplift | Family-clustered 95% CI | McNemar p |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison_id, comparison in report["paired_comparisons"].items():
        clustered = comparison["family_clustered"]
        low, high = clustered["task_weighted_family_clustered_ci95"]
        p_value = comparison["mcnemar"]["exact_two_sided_p_value"]
        lines.append(
            f"| `{comparison_id}` | {clustered['task_weighted_uplift'] * 100:.2f} pp | [{low * 100:.2f}, {high * 100:.2f}] pp | {p_value:.4f} |"
        )
    lines.extend(["", "## Claim boundaries", ""])
    lines.extend(f"- {item}" for item in report["claim_boundaries"])
    return "\n".join(lines) + "\n"
