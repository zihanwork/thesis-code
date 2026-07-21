from __future__ import annotations

from collections import defaultdict
from itertools import product
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_rag_overlap(
    training_tasks: list[dict[str, Any]],
    heldout_tasks: list[dict[str, Any]],
    retrieval_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure instance, family, instruction, plan, and selected-demo overlap.

    Task identifiers remain the unit for exact split disjointness. Instruction,
    family, and plan overlap are reported separately so ID-disjoint data is not
    accidentally described as unseen-family or unseen-plan generalization.
    """

    train_by_id = {str(task["id"]): task for task in training_tasks}
    heldout_by_id = {str(task["id"]): task for task in heldout_tasks}
    if len(train_by_id) != len(training_tasks):
        raise ValueError("RAG training tasks contain duplicate IDs.")
    if len(heldout_by_id) != len(heldout_tasks):
        raise ValueError("Held-out tasks contain duplicate IDs.")

    train_ids = set(train_by_id)
    heldout_ids = set(heldout_by_id)
    train_instructions = {_normalized_instruction(task) for task in training_tasks}
    train_families = {_task_family(task) for task in training_tasks}
    heldout_families = {_task_family(task) for task in heldout_tasks}
    train_plans = {_plan_key(task) for task in training_tasks if task.get("gold_plan")}

    instruction_seen = [
        task for task in heldout_tasks if _normalized_instruction(task) in train_instructions
    ]
    family_seen = [task for task in heldout_tasks if _task_family(task) in train_families]
    plan_seen = [task for task in heldout_tasks if _plan_key(task) in train_plans]

    report: dict[str, Any] = {
        "grain": "one held-out task instance",
        "training_task_count": len(training_tasks),
        "heldout_task_count": len(heldout_tasks),
        "task_id_overlap": _count_rate(len(train_ids & heldout_ids), len(heldout_tasks)),
        "normalized_instruction_seen": _count_rate(
            len(instruction_seen), len(heldout_tasks)
        ),
        "task_family_seen": _count_rate(len(family_seen), len(heldout_tasks)),
        "heldout_family_coverage": {
            "seen_family_count": len(train_families & heldout_families),
            "heldout_family_count": len(heldout_families),
            "rate": (
                len(train_families & heldout_families) / len(heldout_families)
                if heldout_families
                else 0.0
            ),
            "families": sorted(heldout_families),
        },
        "gold_plan_exactly_seen_anywhere": _count_rate(
            len(plan_seen), len(heldout_tasks)
        ),
        "claim_classification": {
            "supported": "seen-family, unseen-instance transfer",
            "not_supported": "unseen-task-family or unseen-plan generalization",
        },
    }

    if retrieval_runs is None:
        report["selected_demo_audit"] = None
        return report

    selected_rows: dict[str, dict[str, Any]] = {}
    for row in retrieval_runs:
        if row.get("planner_name") != "P1_rag":
            continue
        if row.get("harness_mode") != "H0_open_loop":
            continue
        task_id = str(row.get("task_id", ""))
        if task_id in selected_rows:
            raise ValueError(f"Duplicate P1/H0 retrieval row for task {task_id}.")
        selected_rows[task_id] = row

    missing = sorted(heldout_ids - set(selected_rows))
    if missing:
        raise ValueError(f"Retrieval run is missing {len(missing)} held-out tasks.")

    selected_same_family = 0
    selected_exact_plan = 0
    scores: list[float] = []
    per_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "selected_same_family": 0, "selected_exact_plan": 0}
    )
    for task_id in sorted(heldout_ids):
        task = heldout_by_id[task_id]
        row = selected_rows[task_id]
        metadata = row.get("initial_plan", {}).get("metadata", {})
        retrieved_id = str(metadata.get("retrieved", ""))
        if retrieved_id not in train_by_id:
            raise ValueError(
                f"Selected retrieval example {retrieved_id!r} is absent from training tasks."
            )
        retrieved = train_by_id[retrieved_id]
        family = _task_family(task)
        family_row = per_family[family]
        family_row["n"] += 1
        if family == _task_family(retrieved):
            selected_same_family += 1
            family_row["selected_same_family"] += 1
        if _plan_key(task) == _plan_key(retrieved):
            selected_exact_plan += 1
            family_row["selected_exact_plan"] += 1
        score = metadata.get("retrieval_score")
        if score is not None:
            scores.append(float(score))

    report["selected_demo_audit"] = {
        "source_row_count": len(selected_rows),
        "selected_same_family": _count_rate(
            selected_same_family, len(heldout_tasks)
        ),
        "selected_exact_gold_plan": _count_rate(
            selected_exact_plan, len(heldout_tasks)
        ),
        "retrieval_score": {
            "n": len(scores),
            "min": min(scores) if scores else None,
            "median": median(scores) if scores else None,
            "max": max(scores) if scores else None,
            "exactly_one": sum(score == 1.0 for score in scores),
        },
        "by_family": [
            {"task_family": family, **values}
            for family, values in sorted(per_family.items())
        ],
    }
    return report


def family_clustered_paired_analysis(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 13,
) -> dict[str, Any]:
    """Analyze paired binary outcomes with task-family cluster resampling.

    The task-weighted estimand preserves the original per-task success-rate
    contrast. The equal-family estimand gives each task family one vote. Both
    intervals resample whole families, preserving within-family dependence.
    """

    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")
    left = _unique_metric_rows(left_rows, "left")
    right = _unique_metric_rows(right_rows, "right")
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("Paired comparison has no shared task IDs.")
    if set(left) != set(right):
        raise ValueError("Paired comparison requires identical task ID sets.")

    clustered: dict[str, list[int]] = defaultdict(list)
    for task_id in shared:
        left_family = _metric_family(left[task_id])
        right_family = _metric_family(right[task_id])
        if left_family != right_family:
            raise ValueError(f"Task-family mismatch for {task_id}.")
        difference = int(bool(right[task_id].get("task_success"))) - int(
            bool(left[task_id].get("task_success"))
        )
        clustered[left_family].append(difference)

    families = sorted(clustered)
    family_means = {family: mean(clustered[family]) for family in families}
    all_differences = [value for family in families for value in clustered[family]]
    task_uplift = mean(all_differences)
    equal_family_uplift = mean(family_means.values())

    generator = random.Random(seed)
    task_weighted_bootstrap: list[float] = []
    equal_family_bootstrap: list[float] = []
    for _ in range(samples):
        sampled_families = [generator.choice(families) for _ in families]
        sampled_task_differences = [
            value for family in sampled_families for value in clustered[family]
        ]
        task_weighted_bootstrap.append(mean(sampled_task_differences))
        equal_family_bootstrap.append(
            mean(family_means[family] for family in sampled_families)
        )

    task_weighted_bootstrap.sort()
    equal_family_bootstrap.sort()
    return {
        "paired_task_count": len(shared),
        "task_family_count": len(families),
        "task_weighted_uplift": task_uplift,
        "task_weighted_family_clustered_ci95": [
            _percentile(task_weighted_bootstrap, 0.025),
            _percentile(task_weighted_bootstrap, 0.975),
        ],
        "equal_family_uplift": equal_family_uplift,
        "equal_family_clustered_ci95": [
            _percentile(equal_family_bootstrap, 0.025),
            _percentile(equal_family_bootstrap, 0.975),
        ],
        "exact_family_sign_flip_p_value": _exact_family_sign_flip(
            list(family_means.values())
        ),
        "bootstrap": {
            "unit": "task_family",
            "samples": samples,
            "seed": seed,
        },
        "by_family": [
            {
                "task_family": family,
                "n": len(clustered[family]),
                "uplift": family_means[family],
            }
            for family in families
        ],
    }


def audit_rq3_identifiability(method_ids: set[str]) -> dict[str, Any]:
    cells = {
        "P0_engineered_prompt__H0_open_loop": (
            "P0_engineered_prompt__H0_open_loop" in method_ids
        ),
        "P0_engineered_prompt__H2_llm_reflection": (
            "P0_engineered_prompt__H2_llm_reflection" in method_ids
        ),
        "P1_rag__H0_open_loop": "P1_rag__H0_open_loop" in method_ids,
        "P1_rag__H2_llm_reflection": (
            "P1_rag__H2_llm_reflection" in method_ids
        ),
    }
    missing = [cell for cell, present in cells.items() if not present]
    return {
        "required_2x2_cells": cells,
        "full_factorial_interaction_identifiable": not missing,
        "missing_cells": missing,
        "supported_estimands": [
            "RAG uplift over the engineered prompt under open-loop execution",
            "Reflection uplift over open loop conditional on the P1 RAG planner",
        ],
        "unsupported_estimand": (
            "The P0-versus-P1 difference in Reflection uplift (the factorial "
            "planning-by-recovery interaction)."
        ),
        "revised_rq3": (
            "After RAG-based initial planning, how many residual failures are repaired "
            "by validator-feedback reflection, and at what resource cost?"
        ),
        "claim_policy": (
            "Use conditional/sequential gain language. Do not describe the current "
            "final design as identifying complementarity or an interaction effect."
        ),
    }


def compare_local_and_official_metrics(
    final_results: dict[str, Any],
    official_results: dict[str, Any],
) -> dict[str, Any]:
    local = final_results["task_success"]["gpt-5.5"][
        "P1_rag__H2_llm_reflection"
    ]
    compatible = int(official_results["export"]["exported_task_count"])
    official_success = int(
        official_results["official_evaluator"]["task_success"]["count"]
    )
    local_all_success = int(local["success"]) == int(local["n"])
    if not local_all_success:
        raise ValueError(
            "Compatible-subset local success cannot be inferred unless all local tasks succeed."
        )
    official_rate = official_success / compatible if compatible else 0.0
    local_rate = 1.0 if compatible else 0.0
    action_goal = official_results["official_evaluator"]["goal_metrics"]["action"]
    heldout_total = int(official_results["source"]["heldout_task_count"])
    return {
        "method": "gpt-5.5 P1 RAG + LLM Reflection",
        "local_all_heldout": {
            "success": int(local["success"]),
            "n": int(local["n"]),
            "rate": float(local["rate"]),
            "metric": "custom PDDL final-state task success",
        },
        "compatible_subset": {
            "n": compatible,
            "heldout_total": heldout_total,
            "coverage": compatible / heldout_total if heldout_total else 0.0,
            "local_success_inferred": compatible,
            "local_rate_inferred": local_rate,
            "official_success": official_success,
            "official_rate": official_rate,
            "absolute_gap": local_rate - official_rate,
            "local_success_official_failure": compatible - official_success,
        },
        "official_action_goal": {
            "matched": int(action_goal["matched"]),
            "total": int(action_goal["total"]),
            "rate": (
                int(action_goal["matched"]) / int(action_goal["total"])
                if int(action_goal["total"])
                else 0.0
            ),
        },
        "interpretation": (
            "The local evaluator measures final-state reachability under the project PDDL "
            "semantics. The official evaluator additionally measures action/LTL objectives; "
            "the scores are separate estimands and must not be substituted for one another."
        ),
    }


def build_thesis_validity_audit(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    training_tasks = load_jsonl(inputs["rag_train"])
    heldout_tasks = load_jsonl(inputs["heldout"])
    retrieval_runs = load_jsonl(inputs["retrieval_runs"])

    metric_cache: dict[str, list[dict[str, Any]]] = {}

    def metrics(path: str) -> list[dict[str, Any]]:
        if path not in metric_cache:
            metric_cache[path] = load_jsonl(path)
        return metric_cache[path]

    clustered: dict[str, Any] = {}
    all_method_ids: set[str] = set()
    for item in config["method_sources"]:
        rows = metrics(item["path"])
        all_method_ids.update(str(row["method_id"]) for row in rows)
    for comparison in config["clustered_comparisons"]:
        left_rows = [
            row
            for row in metrics(comparison["left"]["path"])
            if row["method_id"] == comparison["left"]["method_id"]
        ]
        right_rows = [
            row
            for row in metrics(comparison["right"]["path"])
            if row["method_id"] == comparison["right"]["method_id"]
        ]
        result = family_clustered_paired_analysis(
            left_rows,
            right_rows,
            samples=int(config.get("bootstrap_samples", 10_000)),
            seed=int(config.get("seed", 13)),
        )
        clustered[comparison["id"]] = {
            "model": comparison["model"],
            "left_method": comparison["left"]["method_id"],
            "right_method": comparison["right"]["method_id"],
            **result,
        }

    final_results = json.loads(
        Path(inputs["final_results_evidence"]).read_text(encoding="utf-8")
    )
    official_results = json.loads(
        Path(inputs["official_results_evidence"]).read_text(encoding="utf-8")
    )
    source_paths = {
        inputs["rag_train"],
        inputs["heldout"],
        inputs["retrieval_runs"],
        inputs["final_results_evidence"],
        inputs["official_results_evidence"],
        *(item["path"] for item in config["method_sources"]),
    }
    return {
        "schema_version": 1,
        "audit_id": str(config.get("audit_id", "thesis_validity_audit_v1")),
        "status": "complete",
        "scope": (
            "Post-run validity analysis using frozen task files and immutable final-run "
            "artifacts. No model calls and no changes to the one-shot final protocol."
        ),
        "rag_overlap": audit_rag_overlap(
            training_tasks, heldout_tasks, retrieval_runs
        ),
        "rq3_identifiability": audit_rq3_identifiability(all_method_ids),
        "local_official_metric_gap": compare_local_and_official_metrics(
            final_results, official_results
        ),
        "family_clustered_comparisons": clustered,
        "source_sha256": {
            path: sha256_file(path) for path in sorted(source_paths)
        },
    }


def export_thesis_validity_audit(
    *,
    config_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    audit = build_thesis_validity_audit(config_path)
    output = Path(output_path)
    report = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.write_text(_render_markdown_report(audit), encoding="utf-8")
    return audit


def _render_markdown_report(audit: dict[str, Any]) -> str:
    rag = audit["rag_overlap"]
    selected = rag["selected_demo_audit"]
    rq3 = audit["rq3_identifiability"]
    metric_gap = audit["local_official_metric_gap"]
    compatible = metric_gap["compatible_subset"]
    lines = [
        "# Thesis validity amendment",
        "",
        "## Technical summary",
        "",
        (
            "The frozen final results remain reproducible, but the post-run validity "
            "audit narrows their interpretation. The held-out set is ID-disjoint but "
            "family- and plan-overlapping; RQ3 supports a conditional recovery gain, "
            "not a factorial interaction; the custom and official evaluators measure "
            "different estimands; and inferential summaries now resample whole task "
            "families."
        ),
        "",
        "## RAG evidence supports seen-family instance transfer",
        "",
        "| Check | Count | Rate |",
        "| --- | ---: | ---: |",
        _count_rate_row("Task-ID overlap", rag["task_id_overlap"]),
        _count_rate_row(
            "Held-out instruction seen in RAG training",
            rag["normalized_instruction_seen"],
        ),
        _count_rate_row("Held-out task family seen", rag["task_family_seen"]),
        _count_rate_row(
            "Held-out gold plan seen anywhere", rag["gold_plan_exactly_seen_anywhere"]
        ),
        _count_rate_row(
            "Selected demonstration from same family",
            selected["selected_same_family"],
        ),
        _count_rate_row(
            "Selected demonstration has exact gold plan",
            selected["selected_exact_gold_plan"],
        ),
        "",
        (
            "Accordingly, the final P1 result is reported as **seen-family, "
            "unseen-instance transfer**. It is not evidence of unseen-family or "
            "unseen-plan generalization."
        ),
        "",
        "## RQ3 is a conditional recovery question",
        "",
        f"Full 2x2 interaction identifiable: **{str(rq3['full_factorial_interaction_identifiable']).lower()}**.",
        "",
        f"Missing cell: `{', '.join(rq3['missing_cells'])}`.",
        "",
        f"Revised RQ3: **{rq3['revised_rq3']}**",
        "",
        "## Local and official success are separate estimands",
        "",
        "| Evaluation | Success | Rate |",
        "| --- | ---: | ---: |",
        (
            f"| Local PDDL, all held-out | {metric_gap['local_all_heldout']['success']}/"
            f"{metric_gap['local_all_heldout']['n']} | "
            f"{metric_gap['local_all_heldout']['rate']:.1%} |"
        ),
        (
            f"| Local PDDL, official-compatible subset (inferred) | "
            f"{compatible['local_success_inferred']}/{compatible['n']} | "
            f"{compatible['local_rate_inferred']:.1%} |"
        ),
        (
            f"| Pinned official evaluator, compatible subset | "
            f"{compatible['official_success']}/{compatible['n']} | "
            f"{compatible['official_rate']:.1%} |"
        ),
        "",
        (
            f"The compatible-subset gap is {compatible['absolute_gap']:.1%} "
            f"({compatible['local_success_official_failure']} tasks). This is a metric-definition "
            "difference, not a contradiction or a leaderboard result."
        ),
        "",
        "## Family-clustered robustness",
        "",
        "| Model and contrast | Task uplift | Clustered 95% CI | Equal-family uplift | Equal-family 95% CI | Sign-flip p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison in audit["family_clustered_comparisons"].values():
        task_ci = comparison["task_weighted_family_clustered_ci95"]
        family_ci = comparison["equal_family_clustered_ci95"]
        lines.append(
            f"| {comparison['model']}: {comparison['left_method']} → "
            f"{comparison['right_method']} | {comparison['task_weighted_uplift']:.1%} | "
            f"{task_ci[0]:.1%} to {task_ci[1]:.1%} | "
            f"{comparison['equal_family_uplift']:.1%} | "
            f"{family_ci[0]:.1%} to {family_ci[1]:.1%} | "
            f"{comparison['exact_family_sign_flip_p_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            (
                "These intervals preserve within-family dependence. They complement, "
                "rather than replace, the frozen task-level McNemar and Wilson results."
            ),
            "",
            "## Claim policy",
            "",
            "- Use instance-transfer language for the final RAG result.",
            "",
            "- Describe RQ3 as recovery gain conditional on P1, not complementarity.",
            "",
            "- Report local PDDL and official action/LTL outcomes side by side.",
            "",
            "- Use family-clustered intervals for the primary robustness claims.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m embodied_gap.cli audit-thesis-validity \\",
            "  --config configs/experiments/thesis_validity_audit_v1.json \\",
            "  --out docs/thesis_validity_evidence.json \\",
            "  --report docs/thesis_validity_report.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _unique_metric_rows(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in result:
            raise ValueError(f"Duplicate {label} metric row for task {task_id}.")
        result[task_id] = row
    return result


def _metric_family(row: dict[str, Any]) -> str:
    family = row.get("metadata", {}).get("task_family")
    if not family:
        raise ValueError(f"Metric row {row.get('task_id')} has no task_family.")
    return str(family)


def _normalized_instruction(task: dict[str, Any]) -> str:
    return " ".join(str(task.get("instruction", "")).lower().split())


def _task_family(task: dict[str, Any]) -> str:
    family = task.get("slots", {}).get("task_family")
    if not family:
        raise ValueError(f"Task {task.get('id')} has no task_family.")
    return str(family)


def _plan_key(task: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(action) for action in task.get("gold_plan", []))


def _count_rate(count: int, total: int) -> dict[str, float | int]:
    return {"count": count, "n": total, "rate": count / total if total else 0.0}


def _count_rate_row(label: str, value: dict[str, Any]) -> str:
    return f"| {label} | {value['count']}/{value['n']} | {value['rate']:.1%} |"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = quantile * (len(values) - 1)
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _exact_family_sign_flip(family_effects: list[float]) -> float:
    if not family_effects:
        return 1.0
    observed = abs(mean(family_effects))
    tolerance = 1e-12
    total = 0
    extreme = 0
    for signs in product((-1.0, 1.0), repeat=len(family_effects)):
        permuted = abs(
            mean(
                sign * effect
                for sign, effect in zip(signs, family_effects, strict=True)
            )
        )
        total += 1
        if permuted + tolerance >= observed:
            extreme += 1
    return extreme / total
