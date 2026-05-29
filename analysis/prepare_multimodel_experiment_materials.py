#!/usr/bin/env python3
"""Prepare multi-model experiment materials for the EAI/VirtualHome study.

The script is intentionally read-only with respect to evaluation inputs: it
collects existing ``summary.json`` files, deduplicates overlapping result roots,
and writes presentation-ready tables, notes, and lightweight SVG figures.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT_PRIORITY = {
    "output_norm_all": 0,
    "output_single_norm": 1,
    "output/improvement_run": 2,
    "output": 3,
}

ACTION_METRICS = [
    "goal_evaluation.task_success_rate",
    "trajectory_evaluation.execution_success_rate",
    "goal_evaluation.state_goal",
    "goal_evaluation.relation_goal",
    "goal_evaluation.action_goal",
    "goal_evaluation.total_goal",
    "trajectory_evaluation.grammar_error.parsing",
    "trajectory_evaluation.grammar_error.hallucination",
    "trajectory_evaluation.grammar_error.predicate_argument_number",
    "trajectory_evaluation.runtime_error.wrong_order",
    "trajectory_evaluation.runtime_error.missing_step",
    "trajectory_evaluation.runtime_error.affordance_error",
    "trajectory_evaluation.runtime_error.additional_step",
]

GOAL_METRICS = [
    "all_f1",
    "all_precision",
    "all_recall",
    "node_f1",
    "edge_f1",
    "action_f1",
]

RELATED_WORK = [
    {
        "topic": "Embodied Agent Interface",
        "citation": "Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making, NeurIPS Datasets and Benchmarks 2024",
        "url": "https://arxiv.org/abs/2410.07166",
        "use": "作为本实验的核心基准来源，说明 EAI 如何统一 goal interpretation、subgoal decomposition、action sequencing、transition modeling，并提供细粒度错误指标。",
    },
    {
        "topic": "VirtualHome",
        "citation": "VirtualHome: Simulating Household Activities via Programs",
        "url": "http://virtual-home.org/",
        "use": "作为家庭活动动作序列环境背景，用来解释对象、状态、关系、动作可执行性和长程 household task 的难点。",
    },
    {
        "topic": "Long-horizon planning",
        "citation": "Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks",
        "url": "https://arxiv.org/html/2503.09572",
        "use": "支撑“先规划再执行”和 dynamic replanning 的实验动机，强调复杂任务不能只依赖简单 prompt engineering。",
    },
    {
        "topic": "Embodied multimodal benchmarks",
        "citation": "EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents",
        "url": "https://arxiv.org/abs/2502.09560",
        "use": "补充具身智能评测背景，尤其是长程规划、低层执行和环境反馈对模型成功率的影响。",
    },
    {
        "topic": "Self-refinement for planning",
        "citation": "Self-Refine: Iterative Refinement with Self-Feedback",
        "url": "https://arxiv.org/abs/2303.17651",
        "use": "为生成后自检、修复和 self-consistency 提供方法依据；重点关注动作前置条件、状态更新和错误路径纠正。",
    },
    {
        "topic": "Verbal feedback for agents",
        "citation": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "url": "https://arxiv.org/abs/2303.11366",
        "use": "用于说明失败轨迹可转化为语言反馈，支持下一轮 action sequence 生成或 prompt 修复。",
    },
    {
        "topic": "Reasoning-action prompting",
        "citation": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "url": "https://arxiv.org/abs/2210.03629",
        "use": "可作为 plan-then-ground 或 thought/action 分离实验的背景，但最终输出仍需压回 EAI 的合法动作格式。",
    },
]

NEW_MODEL_MATRIX = [
    {
        "family": "OpenAI",
        "candidate_model": "gpt-4.1 or latest available GPT-4-class model",
        "role": "strong proprietary baseline",
        "priority": "high",
        "sample_scope": "full VirtualHome action_sequencing set after 10-sample smoke test",
        "generation_parameters": "temperature=0, max_tokens=2048, same EAI prompt",
        "generation_path": "OpenAI-compatible generator",
        "output_location": "output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json",
        "eval_command": "normalize_action_outputs.py, then run_action_sequencing_eval.sh",
        "notes": "与已有 gpt-4o / gpt-4-turbo 形成同家族纵向对比。",
    },
    {
        "family": "OpenAI",
        "candidate_model": "gpt-4o-mini or latest small model",
        "role": "cost-efficient proprietary baseline",
        "priority": "medium",
        "sample_scope": "full set; use for ablations if budget is constrained",
        "generation_parameters": "temperature=0, max_tokens=2048, same EAI prompt",
        "generation_path": "OpenAI-compatible generator",
        "output_location": "output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json",
        "eval_command": "normalize_action_outputs.py, then run_action_sequencing_eval.sh",
        "notes": "适合先跑全量或多轮消融，成本低。",
    },
    {
        "family": "Anthropic",
        "candidate_model": "Claude 3.5/3.7 Sonnet or latest Sonnet",
        "role": "reasoning and instruction-following comparison",
        "priority": "high",
        "sample_scope": "same identifiers as OpenAI run; smoke test first 10 rows",
        "generation_parameters": "temperature=0, max_tokens=2048, same system format contract",
        "generation_path": "Anthropic generator",
        "output_location": "output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json",
        "eval_command": "normalize_action_outputs.py, then run_action_sequencing_eval.sh",
        "notes": "与已有 Claude 3/3.5 结果比较格式稳定性和 planning order。",
    },
    {
        "family": "Google",
        "candidate_model": "Gemini 1.5/2.x Pro or Flash",
        "role": "long-context and fast model comparison",
        "priority": "high",
        "sample_scope": "same identifiers as OpenAI run; smoke test first 10 rows",
        "generation_parameters": "temperature=0, max_output_tokens=2048, same EAI prompt",
        "generation_path": "Gemini generator",
        "output_location": "output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json",
        "eval_command": "normalize_action_outputs.py, then run_action_sequencing_eval.sh",
        "notes": "重点观察 relation grounding 与 missing_step。",
    },
    {
        "family": "Open-weight",
        "candidate_model": "Llama 3.1/3.2 70B or Qwen2.5 72B Instruct",
        "role": "open-weight strong baseline",
        "priority": "medium",
        "sample_scope": "same identifiers; optionally start with representative 100-row subset",
        "generation_parameters": "temperature=0, max_tokens=2048, OpenAI-compatible chat format",
        "generation_path": "local or hosted OpenAI-compatible endpoint",
        "output_location": "output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json",
        "eval_command": "normalize_action_outputs.py, then run_action_sequencing_eval.sh",
        "notes": "用于说明开源模型与闭源模型的差距和可复现实验价值。",
    },
    {
        "family": "Mistral",
        "candidate_model": "Mistral Large latest",
        "role": "VirtualHome action sequencing comparison",
        "priority": "medium",
        "sample_scope": "same identifiers as OpenAI run; smoke test first 10 rows",
        "generation_parameters": "temperature=0, max_tokens=2048, same EAI prompt",
        "generation_path": "Mistral or hosted endpoint",
        "output_location": "output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json",
        "eval_command": "normalize_action_outputs.py, then run_action_sequencing_eval.sh",
        "notes": "EAI 官方结果中 Mistral Large 在 VirtualHome action sequencing 表现较强。",
    },
]

IMPROVEMENT_PLAN = [
    {
        "stage": "baseline",
        "prompt_or_method": "现有 EAI prompt + temperature=0",
        "hypothesis": "建立与已有结果可比的直接生成基线。",
        "measure": "task_success_rate, execution_success_rate, parsing, missing_step, wrong_order",
    },
    {
        "stage": "format_constraints",
        "prompt_or_method": "强化 JSON 拼接格式、动作大写、name/id 成对、禁止 Markdown",
        "hypothesis": "主要降低 parsing 和 predicate_argument_number 错误。",
        "measure": "grammar_error.parsing, grammar_error.predicate_argument_number",
    },
    {
        "stage": "few_shot_valid_actions",
        "prompt_or_method": "加入 1-3 个合法 VirtualHome action sequence 示例",
        "hypothesis": "改善动作选择和参数数量，减少 hallucination。",
        "measure": "hallucination, task_success_rate",
    },
    {
        "stage": "self_check_rewrite",
        "prompt_or_method": "生成后执行一次自检，检查动作合法性、对象 id、前置条件和顺序，再输出修正版",
        "hypothesis": "降低 wrong_order、missing_step 和 additional_step。",
        "measure": "runtime_error.wrong_order, runtime_error.missing_step, runtime_error.additional_step",
    },
    {
        "stage": "plan_then_ground",
        "prompt_or_method": "先生成 high-level plan，再映射到 VirtualHome name/id action sequence",
        "hypothesis": "改善长程动作顺序和 relation grounding。",
        "measure": "relation_goal, action_goal, execution_success_rate",
    },
    {
        "stage": "failure_driven_prompt",
        "prompt_or_method": "根据 diagnostics 中 relation_grounding、planning_order、format_or_parsing 的高频失败定向改 prompt",
        "hypothesis": "针对当前错误分布提升总体 task_success_rate。",
        "measure": "failure-type distribution before/after",
    },
]

FIGURE_PLAN = [
    {
        "figure": "fig_action_task_success.svg",
        "claim": "动作序列任务中不同模型的可执行成功率差异明显。",
        "data": "action_sequencing goal_evaluation.task_success_rate",
    },
    {
        "figure": "fig_goal_interpretation_f1.svg",
        "claim": "目标理解能力与动作执行能力不是同一个指标，应分开讨论。",
        "data": "goal_interpretation all_f1",
    },
    {
        "figure": "fig_action_execution_success.svg",
        "claim": "执行成功率能揭示格式合法但任务目标仍未完成的情况。",
        "data": "action_sequencing trajectory_evaluation.execution_success_rate",
    },
    {
        "figure": "fig_failure_profile.svg",
        "claim": "失败主要来自 missing_step、additional_step、hallucination 等细粒度错误，而不只是最终失败。",
        "data": "action_sequencing grammar_error and runtime_error metrics",
    },
    {
        "figure": "fig_family_average.svg",
        "claim": "模型家族层面的平均表现可辅助解释闭源、开源和推理模型差异。",
        "data": "family average of rank metrics",
    },
]


@dataclass
class ResultRow:
    dataset: str
    eval_type: str
    model: str
    family: str
    source_root: str
    summary_path: str
    rank_metric: str
    rank_value: float
    metrics: Dict[str, float]


def flatten_dict(data: Dict, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        elif isinstance(value, (int, float)):
            flat[full_key] = float(value)
    return flat


VARIANT_LABELS = (
    "format_constraints",
    "few_shot",
    "self_check",
    "plan_then_ground",
    "schema_constrained",
    "decompose_then_merge",
    # Knowledge-Grounded Recovery variants (SG-RAG + PC-KG)
    "sg_rag_pc_kg",
    "pc_kg_self_check",
    "pc_kg_triage",
    "sg_rag",
)


INTERVENTION_FAMILIES = {
    "baseline": "baseline",
    "format_constraints": "prompt",
    "few_shot": "prompt",
    "self_check": "prompt",
    "plan_then_ground": "prompt",
    "schema_constrained": "prompt",
    "decompose_then_merge": "prompt",
    "sg_rag": "sg_rag",
    "pc_kg_self_check": "pc_kg",
    "pc_kg_triage": "pc_kg",
    "sg_rag_pc_kg": "sg_rag_pc_kg",
}


def intervention_family_for(variant: str) -> str:
    return INTERVENTION_FAMILIES.get(variant, "prompt")


def parse_variant(model: str) -> Tuple[str, str]:
    """Split a directory name like ``gpt-4o-mini_self_check`` into base+variant."""

    for variant in VARIANT_LABELS:
        suffix = f"_{variant}"
        if model.endswith(suffix):
            return model[: -len(suffix)], variant
    return model, "baseline"


def infer_family(model: str) -> str:
    m = model.lower()
    if m == "gold_oracle":
        return "Oracle"
    if m.startswith(("gpt", "o1", "openai")):
        return "OpenAI"
    if m.startswith("claude"):
        return "Anthropic"
    if m.startswith("gemini"):
        return "Google"
    if "llama" in m:
        return "Meta"
    if "mistral" in m or "mixtral" in m:
        return "Mistral"
    if "cohere" in m or "command-r" in m:
        return "Cohere"
    return "Other"


def infer_rank_metric(eval_type: str, metrics: Dict[str, float]) -> Tuple[str, float]:
    if eval_type == "goal_interpretation":
        return "all_f1", metrics.get("all_f1", 0.0)
    candidates = (
        "goal_evaluation.task_success_rate",
        "trajectory_evaluation.execution_success_rate",
        "all_f1",
    )
    for key in candidates:
        if key in metrics:
            return key, metrics[key]
    return "na", 0.0


def collect_results(repo_root: Path) -> List[ResultRow]:
    import itertools
    selected: Dict[Tuple[str, str, str], Tuple[int, Path]] = {}
    summary_paths = itertools.chain(
        repo_root.glob("output*/virtualhome/evaluate_results/*/*/summary.json"),
        repo_root.glob("output*/improvement_run/virtualhome/evaluate_results/*/*/summary.json"),
    )
    for summary_path in summary_paths:
        rel_parts = summary_path.relative_to(repo_root).parts
        source_root = rel_parts[0] + ("/improvement_run" if "improvement_run" in rel_parts else "")
        eval_type = summary_path.parent.parent.name
        model = summary_path.parent.name
        key = ("virtualhome", eval_type, model)
        priority = ROOT_PRIORITY.get(source_root, 99)
        current = selected.get(key)
        if current is None or priority < current[0]:
            selected[key] = (priority, summary_path)

    rows: List[ResultRow] = []
    for (dataset, eval_type, model), (_, summary_path) in sorted(selected.items()):
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = flatten_dict(data if isinstance(data, dict) else {})
        rank_metric, rank_value = infer_rank_metric(eval_type, metrics)
        rel_parts = summary_path.relative_to(repo_root).parts
        source_root = rel_parts[0] + ("/improvement_run" if "improvement_run" in rel_parts else "")
        rows.append(
            ResultRow(
                dataset=dataset,
                eval_type=eval_type,
                model=model,
                family=infer_family(model),
                source_root=source_root,
                summary_path=str(summary_path.relative_to(repo_root)),
                rank_metric=rank_metric,
                rank_value=rank_value,
                metrics=metrics,
            )
        )
    return rows


def metric_keys(rows: Iterable[ResultRow]) -> List[str]:
    keys = set()
    for row in rows:
        keys.update(row.metrics.keys())
    return sorted(keys)


def write_result_csv(rows: Sequence[ResultRow], path: Path) -> None:
    keys = metric_keys(rows)
    fields = [
        "dataset",
        "eval_type",
        "model",
        "family",
        "source_root",
        "rank_metric",
        "rank_value",
        "summary_path",
    ] + keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "dataset": row.dataset,
                "eval_type": row.eval_type,
                "model": row.model,
                "family": row.family,
                "source_root": row.source_root,
                "rank_metric": row.rank_metric,
                "rank_value": f"{row.rank_value:.4f}",
                "summary_path": row.summary_path,
            }
            out.update({key: row.metrics.get(key, "") for key in keys})
            writer.writerow(out)


def write_rows_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def top_rows(rows: Sequence[ResultRow], eval_type: str, limit: int = 8) -> List[ResultRow]:
    selected = [r for r in rows if r.eval_type == eval_type]
    selected.sort(key=lambda r: r.rank_value, reverse=True)
    return selected[:limit]


def write_kg_verifier_report(repo_root: Path, out_path: Path) -> None:
    """Aggregate every ``*_pc_kg*_report.json`` / ``*_self_check_report.json``
    emitted by ``analysis/self_check_loop.py`` into a single diagnostic file.

    The aggregated report is used in Chapter 6 to quantify which precondition
    violations the PC-KG verifier catches and how often the LLM can fix them.
    """
    import glob as _glob

    reports: List[Dict[str, object]] = []
    totals: Dict[str, int] = {}
    rewrite_success: Dict[str, Dict[str, int]] = {}

    patterns = [
        "output/improvement_run/**/*_pc_kg*_report.json",
        "output/improvement_run/**/*_self_check_report.json",
    ]
    for pattern in patterns:
        for path_str in _glob.glob(str(repo_root / pattern), recursive=True):
            try:
                payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            reports.append(payload)
            for code, count in (payload.get("violations_histogram") or {}).items():
                totals[code] = totals.get(code, 0) + int(count)
            for code, stats in (payload.get("rewrite_success_by_violation_type") or {}).items():
                slot = rewrite_success.setdefault(code, {"attempts": 0, "successes": 0})
                slot["attempts"] += int(stats.get("attempts", 0))
                slot["successes"] += int(stats.get("successes", 0))

    aggregated = {
        "reports_found": len(reports),
        "violations_total_histogram": totals,
        "rewrite_success_by_violation_type": rewrite_success,
        "reports": reports,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_ablation_results(rows: Sequence[ResultRow]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    base_models_with_variants = set()
    for row in rows:
        base_model, variant = parse_variant(row.model)
        if variant != "baseline":
            base_models_with_variants.add(base_model)
    for row in rows:
        base_model, variant = parse_variant(row.model)
        if variant == "baseline" and base_model not in base_models_with_variants and row.model != "gold_oracle":
            continue
        out.append(
            {
                "eval_type": row.eval_type,
                "base_model": base_model,
                "variant": variant,
                "intervention_family": intervention_family_for(variant),
                "task_success_rate": row.metrics.get("goal_evaluation.task_success_rate"),
                "execution_success_rate": row.metrics.get(
                    "trajectory_evaluation.execution_success_rate"
                ),
                "all_f1": row.metrics.get("all_f1"),
                "node_f1": row.metrics.get("node_f1"),
                "edge_f1": row.metrics.get("edge_f1"),
                "action_f1": row.metrics.get("action_f1"),
                "missing_step": row.metrics.get(
                    "trajectory_evaluation.runtime_error.missing_step"
                ),
                "wrong_order": row.metrics.get(
                    "trajectory_evaluation.runtime_error.wrong_order"
                ),
                "additional_step": row.metrics.get(
                    "trajectory_evaluation.runtime_error.additional_step"
                ),
                "hallucination": row.metrics.get(
                    "trajectory_evaluation.grammar_error.hallucination"
                ),
                "parsing": row.metrics.get(
                    "trajectory_evaluation.grammar_error.parsing"
                ),
                "summary_path": row.summary_path,
            }
        )
    return out


def goal_to_action_pairs(rows: Sequence[ResultRow]) -> List[Dict[str, object]]:
    goal_by_model: Dict[str, float] = {}
    action_by_model: Dict[str, float] = {}
    for row in rows:
        base_model, variant = parse_variant(row.model)
        if variant != "baseline":
            continue
        if row.eval_type == "goal_interpretation":
            goal_by_model[base_model] = float(row.metrics.get("all_f1", 0.0))
        elif row.eval_type == "action_sequencing":
            action_by_model[base_model] = float(
                row.metrics.get("goal_evaluation.task_success_rate", 0.0)
            )

    pairs: List[Dict[str, object]] = []
    for model in sorted(set(goal_by_model) & set(action_by_model)):
        if model == "gold_oracle":
            continue
        pairs.append(
            {
                "model": model,
                "family": infer_family(model),
                "goal_all_f1": goal_by_model[model],
                "action_task_success": action_by_model[model],
            }
        )
    return pairs


def family_average(rows: Sequence[ResultRow]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        if row.family == "Oracle":
            continue
        grouped[(row.eval_type, row.family)].append(row.rank_value)

    out = []
    for (eval_type, family), values in sorted(grouped.items()):
        out.append(
            {
                "eval_type": eval_type,
                "family": family,
                "n_models": len(values),
                "average_rank_value": round(sum(values) / len(values), 4),
            }
        )
    return out


def failure_profile(rows: Sequence[ResultRow]) -> List[Dict[str, object]]:
    action_rows = [r for r in rows if r.eval_type == "action_sequencing"]
    out = []
    for row in action_rows:
        out.append(
            {
                "model": row.model,
                "family": row.family,
                "parsing": row.metrics.get("trajectory_evaluation.grammar_error.parsing", 0.0),
                "hallucination": row.metrics.get("trajectory_evaluation.grammar_error.hallucination", 0.0),
                "predicate_argument_number": row.metrics.get(
                    "trajectory_evaluation.grammar_error.predicate_argument_number", 0.0
                ),
                "wrong_order": row.metrics.get("trajectory_evaluation.runtime_error.wrong_order", 0.0),
                "missing_step": row.metrics.get("trajectory_evaluation.runtime_error.missing_step", 0.0),
                "affordance_error": row.metrics.get("trajectory_evaluation.runtime_error.affordance_error", 0.0),
                "additional_step": row.metrics.get("trajectory_evaluation.runtime_error.additional_step", 0.0),
            }
        )
    return out


def write_markdown_report(rows: Sequence[ResultRow], out_path: Path, repo_root: Path) -> None:
    action = top_rows(rows, "action_sequencing", 10)
    goal = top_rows(rows, "goal_interpretation", 10)
    families = family_average(rows)
    source_counts = defaultdict(int)
    for row in rows:
        source_counts[(row.eval_type, row.source_root)] += 1

    lines = [
        "# Multi-model Experiment Materials",
        "",
        "## Existing Model Inventory",
        "",
        f"- Total deduplicated result rows: {len(rows)}",
        "- Result selection prefers `output_norm_all` over `output_single_norm` over `output` when the same model/eval_type appears in multiple roots.",
        "- `gold_oracle` is retained as an upper-bound / pipeline sanity-check reference, not as a model prediction.",
        "",
        "### Source Coverage",
        "",
        "| Eval type | Source root | Models |",
        "| --- | --- | ---: |",
    ]
    for (eval_type, source_root), count in sorted(source_counts.items()):
        lines.append(f"| {eval_type} | `{source_root}` | {count} |")

    lines.extend(
        [
            "",
            "### Top Action Sequencing Results",
            "",
            "| Rank | Model | Family | Source | Task success | Execution success | Missing step | Hallucination |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(action, start=1):
        lines.append(
            "| {rank} | `{model}` | {family} | `{source}` | {task} | {exec_} | {missing} | {hall} |".format(
                rank=idx,
                model=row.model,
                family=row.family,
                source=row.source_root,
                task=fmt(row.metrics.get("goal_evaluation.task_success_rate")),
                exec_=fmt(row.metrics.get("trajectory_evaluation.execution_success_rate")),
                missing=fmt(row.metrics.get("trajectory_evaluation.runtime_error.missing_step")),
                hall=fmt(row.metrics.get("trajectory_evaluation.grammar_error.hallucination")),
            )
        )

    lines.extend(
        [
            "",
            "### Top Goal Interpretation Results",
            "",
            "| Rank | Model | Family | All F1 | Node F1 | Edge F1 | Action F1 |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(goal, start=1):
        lines.append(
            "| {rank} | `{model}` | {family} | {all_f1} | {node} | {edge} | {action_f1} |".format(
                rank=idx,
                model=row.model,
                family=row.family,
                all_f1=fmt(row.metrics.get("all_f1")),
                node=fmt(row.metrics.get("node_f1")),
                edge=fmt(row.metrics.get("edge_f1")),
                action_f1=fmt(row.metrics.get("action_f1")),
            )
        )

    lines.extend(
        [
            "",
            "## Related Work Notes",
            "",
            "| Topic | Citation | How to use in the write-up |",
            "| --- | --- | --- |",
        ]
    )
    for item in RELATED_WORK:
        lines.append(f"| {item['topic']} | [{item['citation']}]({item['url']}) | {item['use']} |")

    lines.extend(
        [
            "",
            "## New Model Evaluation Matrix",
            "",
            "| Family | Candidate model | Priority | Sample scope | Generation parameters | Output location | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in NEW_MODEL_MATRIX:
        lines.append(
            f"| {item['family']} | {item['candidate_model']} | {item['priority']} | {item['sample_scope']} | {item['generation_parameters']} | `{item['output_location']}` | {item['notes']} |"
        )

    lines.extend(
        [
            "",
            "每个新增模型建议使用同一条评测命令链，先规范化，再进入 EAI 评测：",
            "",
            "```bash",
            "python analysis/normalize_action_outputs.py \\",
            "  --input-dir output/<new_response_root>/helm_output/virtualhome/action_sequencing \\",
            "  --output-dir output/<new_response_root>_norm/helm_output/virtualhome/action_sequencing",
            "LLM_RESPONSE_PATH=\"$PWD/output/<new_response_root>_norm/helm_output\" NUM_WORKERS=1 \\",
            "  ./scripts/run_action_sequencing_eval.sh virtualhome eai-eval output",
            "```",
            "",
            "## Success-rate Improvement Ablation",
            "",
            "| Stage | Method | Hypothesis | Primary measures |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in IMPROVEMENT_PLAN:
        lines.append(
            f"| `{item['stage']}` | {item['prompt_or_method']} | {item['hypothesis']} | {item['measure']} |"
        )

    lines.extend(
        [
            "",
            "## Presentation Figure Plan",
            "",
            "| Figure | Main claim | Data source |",
            "| --- | --- | --- |",
        ]
    )
    for item in FIGURE_PLAN:
        lines.append(f"| `{item['figure']}` | {item['claim']} | {item['data']} |")

    lines.extend(
        [
            "",
            "## Suggested Slide Narrative",
            "",
            "1. 先说明为什么使用 EAI/VirtualHome：它不仅看最终成功率，还能拆出目标理解、动作序列和细粒度错误类型。",
            "2. 展示已有多模型覆盖：当前结果已经包含 OpenAI、Anthropic、Google、Meta、Mistral、Cohere 等模型家族。",
            "3. 分开讨论 goal interpretation 和 action sequencing：模型可能理解目标，但仍会在对象接地、动作顺序或执行约束上失败。",
            "4. 先展示失败分布，再引出改进方法：missing step、additional step、hallucinated action 和 relation grounding 是后续消融实验的动机。",
            "5. 最后给出下一轮实验矩阵：新增最新模型家族，并控制 prompt、self-check、planning decomposition 等变量。",
            "",
            "## Generated Artifacts",
            "",
            f"- `{out_path.relative_to(repo_root)}`",
            "- `output/diagnostics/multimodel_existing_inventory.csv`",
            "- `output/diagnostics/multimodel_family_averages.csv`",
            "- `output/diagnostics/multimodel_failure_profile.csv`",
            "- `output/diagnostics/multimodel_new_model_matrix.csv`",
            "- `output/diagnostics/multimodel_success_improvement_plan.csv`",
            "- `output/diagnostics/multimodel_prompt_templates.md`",
            "- `output/diagnostics/multimodel_ablation_summary.csv`",
            "- `output/diagnostics/multimodel_ablation_summary.md`",
            "- `output/diagnostics/multimodel_goal_vs_action.csv`",
            "- `output/diagnostics/progress_report.md`",
            "- `output/diagnostics/figures/*.svg`",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prompt_templates(path: Path) -> None:
    lines = [
        "# Multi-model Prompt and Ablation Templates",
        "",
        "这些模板用于下一轮成功率提升实验。所有模板都应保持相同样本、相同 temperature、相同 max token，并输出到独立 `<model>_<variant>_outputs.json`，方便和 baseline 做消融对比。",
        "",
        "## Variant 1: Baseline",
        "",
        "使用 EAI 原始 `llm_prompt`，仅加最小 system message，作为可比基线。",
        "",
        "```text",
        "You output ONLY the VirtualHome action sequence requested by the user.",
        "Do not include explanations, markdown, or extra text.",
        "```",
        "",
        "## Variant 2: Format Constraints",
        "",
        "目标是降低 `parsing` 和 `predicate_argument_number`。",
        "",
        "```text",
        "You output ONLY a compact JSON action sequence for VirtualHome.",
        "Format: concatenate one or more JSON objects with no separator.",
        "Example: {\"WALK\":[\"floor_lamp\",\"1000\"]}{\"SWITCHON\":[\"floor_lamp\",\"1000\"]}",
        "Rules:",
        "- Action names must be uppercase.",
        "- Each value must be a JSON array of strings.",
        "- Parameters must alternate object_name and numeric id.",
        "- One-object actions use 2 strings; two-object actions use 4 strings.",
        "- STANDUP uses an empty array [].",
        "- Do not wrap the answer in markdown.",
        "```",
        "",
        "## Variant 3: Few-shot Valid Actions",
        "",
        "目标是减少 hallucinated action 和参数数量错误。示例应来自合法轨迹或 oracle 转换结果，避免引入不可执行动作。",
        "",
        "```text",
        "Example 1:",
        "Task: Turn on light",
        "Answer: {\"WALK\":[\"floor_lamp\",\"1000\"]}{\"SWITCHON\":[\"floor_lamp\",\"1000\"]}",
        "",
        "Example 2:",
        "Task: Sit on chair",
        "Answer: {\"WALK\":[\"chair\",\"245\"]}{\"SIT\":[\"chair\",\"245\"]}",
        "",
        "Now solve the user's task with the same output-only format.",
        "```",
        "",
        "## Variant 4: Self-check Rewrite",
        "",
        "目标是降低 `missing_step`、`wrong_order` 和 `additional_step`。实现时建议两次调用：第一次生成草稿，第二次只输出修正版。",
        "",
        "```text",
        "Review the draft action sequence against these checks:",
        "1. Every action is a valid VirtualHome action from the prompt.",
        "2. Every object uses an object name and numeric id available in the prompt.",
        "3. Preconditions are satisfied before each action.",
        "4. The sequence contains necessary steps but no redundant repetitions.",
        "5. The final output still follows the compact JSON object concatenation format.",
        "",
        "Return ONLY the corrected final action sequence.",
        "```",
        "",
        "## Variant 5: Plan Then Ground",
        "",
        "目标是改善长程任务的动作顺序和 relation grounding。实现时第一步生成高层计划，第二步把计划压缩成 EAI 合法动作格式；只保存第二步输出供评测。",
        "",
        "```text",
        "First identify the high-level steps needed to satisfy the task.",
        "Then convert those steps into executable VirtualHome actions using only objects and ids from the prompt.",
        "Final answer must contain ONLY the compact JSON action sequence.",
        "Do not include the high-level plan in the final answer.",
        "```",
        "",
        "## Variant Naming",
        "",
        "- `<model>_baseline_outputs.json`",
        "- `<model>_format_constraints_outputs.json`",
        "- `<model>_few_shot_outputs.json`",
        "- `<model>_self_check_outputs.json`",
        "- `<model>_plan_then_ground_outputs.json`",
        "",
        "## Minimum Reporting Columns",
        "",
        "- model",
        "- variant",
        "- sample_scope",
        "- temperature",
        "- max_tokens",
        "- task_success_rate",
        "- execution_success_rate",
        "- parsing",
        "- hallucination",
        "- missing_step",
        "- wrong_order",
        "- additional_step",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def color(index: int) -> str:
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
    return palette[index % len(palette)]


def escape_xml(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_bar_svg(
    rows: Sequence[Tuple[str, float]],
    path: Path,
    title: str,
    x_label: str,
    width: int = 980,
    height: int = 540,
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    margin_left = 240
    margin_right = 40
    margin_top = 70
    row_height = 26
    height = max(height, margin_top + row_height * len(rows) + 60)
    plot_width = width - margin_left - margin_right
    max_value = max(max(v for _, v in rows), 1.0)
    tick_count = 5
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{escape_xml(title)}</text>',
        f'<text x="{margin_left + plot_width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12">{escape_xml(x_label)}</text>',
    ]
    for i in range(tick_count + 1):
        value = max_value * i / tick_count
        x = margin_left + plot_width * i / tick_count
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top - 10}" x2="{x:.1f}" y2="{height - 48}" stroke="#ddd"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 32}" text-anchor="middle" font-family="Arial" font-size="10">{value:.0f}</text>')
    for idx, (label, value) in enumerate(rows):
        y = margin_top + idx * row_height
        bar_width = plot_width * value / max_value if max_value else 0
        parts.append(f'<text x="{margin_left - 8}" y="{y + 16}" text-anchor="end" font-family="Arial" font-size="11">{escape_xml(label)}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="18" fill="{color(idx)}"/>')
        parts.append(f'<text x="{margin_left + bar_width + 5:.1f}" y="{y + 14}" font-family="Arial" font-size="11">{value:.1f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_failure_svg(profile_rows: Sequence[Dict[str, object]], path: Path) -> None:
    metric_names = [
        "parsing",
        "hallucination",
        "predicate_argument_number",
        "wrong_order",
        "missing_step",
        "affordance_error",
        "additional_step",
    ]
    rows = sorted(profile_rows, key=lambda r: sum(float(r[m]) for m in metric_names), reverse=True)[:12]
    width = 1100
    height = 560
    margin_left = 250
    margin_top = 70
    margin_right = 40
    row_height = 28
    plot_width = width - margin_left - margin_right
    max_total = max([sum(float(r[m]) for m in metric_names) for r in rows] + [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Action Sequencing Failure Profile</text>',
    ]
    legend_x = margin_left
    for idx, metric in enumerate(metric_names):
        lx = legend_x + (idx % 4) * 200
        ly = 50 + (idx // 4) * 18
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="10" height="10" fill="{color(idx)}"/>')
        parts.append(f'<text x="{lx + 14}" y="{ly}" font-family="Arial" font-size="10">{escape_xml(metric)}</text>')
    for ridx, row in enumerate(rows):
        y = margin_top + ridx * row_height
        parts.append(f'<text x="{margin_left - 8}" y="{y + 16}" text-anchor="end" font-family="Arial" font-size="11">{escape_xml(row["model"])}</text>')
        x = margin_left
        total = 0.0
        for midx, metric in enumerate(metric_names):
            value = float(row[metric])
            width_value = plot_width * value / max_total
            if width_value > 0:
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{width_value:.1f}" height="18" fill="{color(midx)}"/>')
            x += width_value
            total += value
        parts.append(f'<text x="{x + 5:.1f}" y="{y + 14}" font-family="Arial" font-size="11">{total:.1f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_figures(rows: Sequence[ResultRow], family_rows: Sequence[Dict[str, object]], profile_rows: Sequence[Dict[str, object]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    action_rows = [
        (r.model, r.metrics.get("goal_evaluation.task_success_rate", 0.0))
        for r in rows
        if r.eval_type == "action_sequencing" and r.model != "gold_oracle"
    ]
    action_rows.sort(key=lambda x: x[1], reverse=True)
    write_bar_svg(action_rows[:15], figures_dir / "fig_action_task_success.svg", "Action Sequencing Task Success", "task_success_rate (%)")

    exec_rows = [
        (r.model, r.metrics.get("trajectory_evaluation.execution_success_rate", 0.0))
        for r in rows
        if r.eval_type == "action_sequencing" and r.model != "gold_oracle"
    ]
    exec_rows.sort(key=lambda x: x[1], reverse=True)
    write_bar_svg(exec_rows[:15], figures_dir / "fig_action_execution_success.svg", "Action Sequencing Execution Success", "execution_success_rate (%)")

    goal_rows = [
        (r.model, r.metrics.get("all_f1", 0.0))
        for r in rows
        if r.eval_type == "goal_interpretation"
    ]
    goal_rows.sort(key=lambda x: x[1], reverse=True)
    write_bar_svg(goal_rows[:15], figures_dir / "fig_goal_interpretation_f1.svg", "Goal Interpretation F1", "all_f1 (%)")

    family_chart_rows = [
        (f"{r['family']} / {r['eval_type'].replace('_', ' ')}", float(r["average_rank_value"]))
        for r in family_rows
    ]
    family_chart_rows.sort(key=lambda x: x[1], reverse=True)
    write_bar_svg(family_chart_rows, figures_dir / "fig_family_average.svg", "Family-level Average Metric", "average rank metric (%)")
    write_failure_svg(profile_rows, figures_dir / "fig_failure_profile.svg")


def write_ablation_summary_md(rows: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "# Multi-model Ablation Summary",
        "",
        "Each row is a `<model>_<variant>_outputs.json` evaluation. `baseline` rows",
        "are kept only when an explicit baseline directory exists; same-name model",
        "directories are interpreted as the implicit baseline.",
        "",
    ]
    if not rows:
        lines.append("No ablation runs found yet. Run the improvement pipeline to populate this section.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["eval_type"])].append(row)

    for eval_type in sorted(grouped):
        lines.append(f"## {eval_type}")
        lines.append("")
        if eval_type == "action_sequencing":
            lines.extend(
                [
                    "| Base model | Variant | Task success | Execution success | Missing step | Wrong order | Hallucination |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in sorted(
                grouped[eval_type],
                key=lambda r: (
                    str(r.get("base_model")),
                    str(r.get("variant")),
                ),
            ):
                lines.append(
                    "| `{base}` | `{variant}` | {task} | {exec_} | {miss} | {wrong} | {hall} |".format(
                        base=row.get("base_model"),
                        variant=row.get("variant"),
                        task=fmt(row.get("task_success_rate")),
                        exec_=fmt(row.get("execution_success_rate")),
                        miss=fmt(row.get("missing_step")),
                        wrong=fmt(row.get("wrong_order")),
                        hall=fmt(row.get("hallucination")),
                    )
                )
        else:
            lines.extend(
                [
                    "| Base model | Variant | All F1 | Node F1 | Edge F1 | Action F1 |",
                    "| --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in sorted(
                grouped[eval_type],
                key=lambda r: (
                    str(r.get("base_model")),
                    str(r.get("variant")),
                ),
            ):
                lines.append(
                    "| `{base}` | `{variant}` | {all_f1} | {node} | {edge} | {action_f1} |".format(
                        base=row.get("base_model"),
                        variant=row.get("variant"),
                        all_f1=fmt(row.get("all_f1")),
                        node=fmt(row.get("node_f1")),
                        edge=fmt(row.get("edge_f1")),
                        action_f1=fmt(row.get("action_f1")),
                    )
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scatter_svg(
    points: Sequence[Dict[str, object]],
    path: Path,
    title: str,
) -> None:
    if not points:
        path.write_text("", encoding="utf-8")
        return
    width = 760
    height = 540
    margin_left = 90
    margin_bottom = 80
    margin_top = 70
    margin_right = 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{escape_xml(title)}</text>',
        f'<text x="{margin_left + plot_w / 2:.0f}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="12">Goal interpretation all_f1 (%)</text>',
        f'<text x="20" y="{margin_top + plot_h / 2:.0f}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(-90 20 {margin_top + plot_h / 2:.0f})">Action task_success_rate (%)</text>',
    ]
    for i in range(6):
        gx = margin_left + plot_w * i / 5
        gy = margin_top + plot_h * i / 5
        parts.append(
            f'<line x1="{gx:.1f}" y1="{margin_top}" x2="{gx:.1f}" y2="{margin_top + plot_h}" stroke="#eee"/>'
        )
        parts.append(
            f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{margin_left + plot_w}" y2="{gy:.1f}" stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{margin_top + plot_h + 16}" text-anchor="middle" font-family="Arial" font-size="10">{20 * i}</text>'
        )
        parts.append(
            f'<text x="{margin_left - 8}" y="{margin_top + plot_h - plot_h * i / 5 + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{20 * i}</text>'
        )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top}" stroke="#999" stroke-dasharray="4 4"/>'
    )
    families = sorted({str(p["family"]) for p in points})
    family_color = {fam: color(idx) for idx, fam in enumerate(families)}
    for idx, fam in enumerate(families):
        lx = margin_left + idx * 110
        parts.append(f'<rect x="{lx}" y="44" width="10" height="10" fill="{family_color[fam]}"/>')
        parts.append(
            f'<text x="{lx + 14}" y="54" font-family="Arial" font-size="11">{escape_xml(fam)}</text>'
        )
    for point in points:
        gx = margin_left + plot_w * float(point["goal_all_f1"]) / 100
        gy = margin_top + plot_h - plot_h * float(point["action_task_success"]) / 100
        parts.append(
            f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="6" fill="{family_color[str(point["family"])]}" stroke="white" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx + 8:.1f}" y="{gy + 4:.1f}" font-family="Arial" font-size="10">{escape_xml(point["model"])}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_empty_state_svg(path: Path, title: str, message: str) -> None:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300" viewBox="0 0 900 300">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="450" y="120" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape_xml(title)}</text>',
        f'<text x="450" y="170" text-anchor="middle" font-family="Arial" font-size="14" fill="#666666">{escape_xml(message)}</text>',
        "</svg>",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_ablation_svg(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        write_empty_state_svg(
            path,
            title="Action Sequencing Ablation (task_success_rate)",
            message="No ablation rows found yet. Run improvement experiments first.",
        )
        return
    bar_rows: List[Tuple[str, float]] = []
    for row in rows:
        if row["eval_type"] != "action_sequencing":
            continue
        value = row.get("task_success_rate")
        if value is None:
            continue
        bar_rows.append((f"{row['base_model']} / {row['variant']}", float(value)))
    if not bar_rows:
        write_empty_state_svg(
            path,
            title="Action Sequencing Ablation (task_success_rate)",
            message="No action-sequencing ablation rows found yet.",
        )
        return
    bar_rows.sort(key=lambda x: x[1], reverse=True)
    write_bar_svg(
        bar_rows,
        path,
        title="Action Sequencing Ablation (task_success_rate)",
        x_label="task_success_rate (%)",
    )


def write_goal_ablation_svg(rows: Sequence[Dict[str, object]], path: Path) -> None:
    bar_rows: List[Tuple[str, float]] = []
    for row in rows:
        if row["eval_type"] != "goal_interpretation":
            continue
        value = row.get("all_f1")
        if value is None:
            continue
        bar_rows.append((f"{row['base_model']} / {row['variant']}", float(value)))
    if not bar_rows:
        write_empty_state_svg(
            path,
            title="Goal Interpretation Ablation (all_f1)",
            message="No goal-interpretation ablation rows found yet.",
        )
        return
    bar_rows.sort(key=lambda x: x[1], reverse=True)
    write_bar_svg(
        bar_rows,
        path,
        title="Goal Interpretation Ablation (all_f1)",
        x_label="all_f1 (%)",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    diagnostics_dir = repo_root / "output" / "diagnostics"
    figures_dir = diagnostics_dir / "figures"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_results(repo_root)
    rows.sort(key=lambda r: (r.eval_type, -r.rank_value, r.model))

    inventory_csv = diagnostics_dir / "multimodel_existing_inventory.csv"
    write_result_csv(rows, inventory_csv)
    inventory_json = diagnostics_dir / "multimodel_existing_inventory.json"
    inventory_json.write_text(
        json.dumps(
            [
                {
                    "dataset": r.dataset,
                    "eval_type": r.eval_type,
                    "model": r.model,
                    "family": r.family,
                    "source_root": r.source_root,
                    "rank_metric": r.rank_metric,
                    "rank_value": r.rank_value,
                    "summary_path": r.summary_path,
                    "metrics": r.metrics,
                }
                for r in rows
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    family_rows = family_average(rows)
    write_rows_csv(
        [{k: str(v) for k, v in row.items()} for row in family_rows],
        diagnostics_dir / "multimodel_family_averages.csv",
    )

    profile_rows = failure_profile(rows)
    write_rows_csv(
        [{k: str(v) for k, v in row.items()} for row in profile_rows],
        diagnostics_dir / "multimodel_failure_profile.csv",
    )

    write_rows_csv(NEW_MODEL_MATRIX, diagnostics_dir / "multimodel_new_model_matrix.csv")
    write_rows_csv(IMPROVEMENT_PLAN, diagnostics_dir / "multimodel_success_improvement_plan.csv")
    write_rows_csv(RELATED_WORK, diagnostics_dir / "multimodel_related_work.csv")
    write_rows_csv(FIGURE_PLAN, diagnostics_dir / "multimodel_figure_plan.csv")
    write_prompt_templates(diagnostics_dir / "multimodel_prompt_templates.md")

    ablation_rows = collect_ablation_results(rows)
    write_rows_csv(
        [
            {k: ("" if v is None else str(v)) for k, v in row.items()}
            for row in ablation_rows
        ],
        diagnostics_dir / "multimodel_ablation_summary.csv",
    )
    write_ablation_summary_md(ablation_rows, diagnostics_dir / "multimodel_ablation_summary.md")

    kg_report_path = diagnostics_dir / "kg_verifier_report.json"
    write_kg_verifier_report(repo_root, kg_report_path)

    pair_rows = goal_to_action_pairs(rows)
    write_rows_csv(
        [{k: str(v) for k, v in row.items()} for row in pair_rows],
        diagnostics_dir / "multimodel_goal_vs_action.csv",
    )

    report_path = diagnostics_dir / "multimodel_experiment_materials.md"
    write_markdown_report(rows, report_path, repo_root)
    write_figures(rows, family_rows, profile_rows, figures_dir)
    write_scatter_svg(pair_rows, figures_dir / "fig_goal_vs_action.svg",
                      "Goal F1 vs Action Task Success")
    write_ablation_svg(ablation_rows, figures_dir / "fig_ablation_action.svg")
    write_goal_ablation_svg(ablation_rows, figures_dir / "fig_ablation_goal.svg")

    print(f"[DONE] wrote {inventory_csv.relative_to(repo_root)}")
    print(f"[DONE] wrote {inventory_json.relative_to(repo_root)}")
    print(f"[DONE] wrote {report_path.relative_to(repo_root)}")
    print(f"[DONE] wrote {figures_dir.relative_to(repo_root)}/*.svg")
    print(f"[DONE] wrote {(diagnostics_dir / 'multimodel_ablation_summary.csv').relative_to(repo_root)}")
    print(f"[DONE] wrote {(diagnostics_dir / 'multimodel_goal_vs_action.csv').relative_to(repo_root)}")
    print(f"[DONE] wrote {kg_report_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
