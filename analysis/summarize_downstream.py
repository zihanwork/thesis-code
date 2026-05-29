#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_summary_file(summary_path: Path) -> Dict:
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def flatten_dict(data: Dict, prefix: str = "") -> Dict:
    flat: Dict = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def collect_module_summaries(base_output: Path, dataset: str, eval_type: str) -> List[Dict]:
    module_dir = base_output / dataset / "evaluate_results" / eval_type
    rows: List[Dict] = []
    if not module_dir.exists():
        return rows

    for summary_path in sorted(module_dir.glob("*/summary.json")):
        model = summary_path.parent.name
        metrics = load_summary_file(summary_path)
        row = {
            "dataset": dataset,
            "eval_type": eval_type,
            "model": model,
            "summary_path": str(summary_path),
        }
        for key, value in flatten_dict(metrics).items():
            row[key] = value
        rows.append(row)
    return rows


def infer_rank_metric(eval_type: str, row: Dict) -> Tuple[str, float]:
    if eval_type == "goal_interpretation":
        return "all_f1", float(row.get("all_f1", 0.0) or 0.0)
    # Generic fallback for downstream modules
    for candidate in (
        "all_f1",
        "success_rate",
        "goal_success",
        "execution_success",
        "task_success",
        "goal_evaluation.task_success_rate",
        "trajectory_evaluation.execution_success_rate",
    ):
        if candidate in row:
            try:
                return candidate, float(row.get(candidate, 0.0) or 0.0)
            except (TypeError, ValueError):
                return candidate, 0.0
    return "na", 0.0


def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("dataset,eval_type,model\n", encoding="utf-8")
        return

    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered_keys = ["dataset", "eval_type", "model", "summary_path"] + sorted(
        k for k in keys if k not in {"dataset", "eval_type", "model", "summary_path"}
    )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Downstream Summary")
    lines.append("")

    if not rows:
        lines.append("No summary files found.")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["eval_type"], []).append(row)

    for eval_type in sorted(grouped.keys()):
        lines.append(f"## {eval_type}")
        lines.append("")
        module_rows = grouped[eval_type]
        ranked = []
        for row in module_rows:
            metric_name, metric_value = infer_rank_metric(eval_type, row)
            ranked.append((metric_value, metric_name, row))
        ranked.sort(key=lambda x: x[0], reverse=True)

        lines.append("| Rank | Model | Metric | Value |")
        lines.append("| --- | --- | --- | ---: |")
        for idx, (metric_value, metric_name, row) in enumerate(ranked, start=1):
            lines.append(
                f"| {idx} | {row['model']} | {metric_name} | {metric_value:.4f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize EAI evaluate_results across goal/action modules."
    )
    parser.add_argument("--dataset", default="virtualhome")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--diagnostics-dir", default="output/diagnostics")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    diagnostics_dir = Path(args.diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for eval_type in ("goal_interpretation", "action_sequencing"):
        all_rows.extend(collect_module_summaries(output_root, args.dataset, eval_type))

    csv_path = diagnostics_dir / "downstream_summary.csv"
    md_path = diagnostics_dir / "downstream_summary.md"
    json_path = diagnostics_dir / "downstream_summary.json"

    write_csv(all_rows, csv_path)
    write_markdown(all_rows, md_path)
    json_path.write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[DONE] Wrote: {csv_path}")
    print(f"[DONE] Wrote: {md_path}")
    print(f"[DONE] Wrote: {json_path}")


if __name__ == "__main__":
    main()
