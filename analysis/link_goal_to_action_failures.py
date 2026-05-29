#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CASE_RE = re.compile(r"Task is (?P<task>.*?), file_id is (?P<file_id>\S+)")
COUNT_RE = re.compile(r"TP_(?P<kind>\w+)_goals: (?P<tp>\d+), FP_(?P=kind)_goals: (?P<fp>\d+), FN_(?P=kind)_goals: (?P<fn>\d+)")
BOOL_RE = re.compile(r"(true|false|1|0)$", re.IGNORECASE)
MODEL_RE = re.compile(r"Model name is (?P<model>.+)$")


def safe_f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


def normalize_bool(value: str) -> Optional[bool]:
    v = value.strip().lower()
    if v in {"true", "1", "yes"}:
        return True
    if v in {"false", "0", "no"}:
        return False
    return None


def classify_failure(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ("name_id format", "parsing", "grammar", "no prediction")):
        return "format_or_parsing"
    if any(x in t for x in ("precondition", "before", "not open", "not close enough", "requires")):
        return "implicit_precondition"
    if any(x in t for x in ("inside", "on ", "relation", "ground", "object id", "target object")):
        return "relation_grounding"
    if any(x in t for x in ("order", "sequence", "first", "then", "step")):
        return "planning_order"
    if any(x in t for x in ("cannot", "infeasible", "invalid action", "unknown action", "not executable", "failed")):
        return "infeasible_or_execution"
    return "other"


def parse_goal_log(goal_log: Path) -> Dict[str, Dict]:
    cases: Dict[str, Dict] = {}
    current_key: Optional[str] = None
    current_model = "unknown"

    with goal_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            content = line.strip()
            model_match = MODEL_RE.search(content)
            if model_match:
                current_model = model_match.group("model").strip()
                continue

            case_match = CASE_RE.search(content)
            if case_match:
                task = case_match.group("task").strip()
                file_id = case_match.group("file_id").strip()
                current_key = f"{current_model}|{file_id}"
                cases[current_key] = {
                    "model": current_model,
                    "task": task,
                    "file_id": file_id,
                    "goal_counts": {},
                    "goal_f1": 0.0,
                }
                continue

            if current_key is None:
                continue

            count_match = COUNT_RE.search(content)
            if count_match:
                kind = count_match.group("kind")
                tp = int(count_match.group("tp"))
                fp = int(count_match.group("fp"))
                fn = int(count_match.group("fn"))
                cases[current_key]["goal_counts"][kind] = {"tp": tp, "fp": fp, "fn": fn}
                continue

    for case in cases.values():
        total_tp = total_fp = total_fn = 0
        for counts in case["goal_counts"].values():
            total_tp += counts["tp"]
            total_fp += counts["fp"]
            total_fn += counts["fn"]
        case["goal_f1"] = round(safe_f1(total_tp, total_fp, total_fn), 4)
    return cases


def parse_action_log(action_log: Path) -> Dict[str, Dict]:
    cases: Dict[str, Dict] = {}
    current_key: Optional[str] = None
    current_model = "unknown"

    with action_log.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            model_match = MODEL_RE.search(line)
            if model_match:
                current_model = model_match.group("model").strip()
                continue

            case_match = CASE_RE.search(line)
            if case_match:
                task = case_match.group("task").strip()
                file_id = case_match.group("file_id").strip()
                current_key = f"{current_model}|{file_id}"
                cases[current_key] = {
                    "model": current_model,
                    "task": task,
                    "file_id": file_id,
                    "task_success": None,
                    "goal_success": None,
                    "execution_success": None,
                    "raw_failure_text": "",
                }
                continue

            if current_key is None:
                continue

            low = line.lower()
            if "task success" in low:
                bool_match = BOOL_RE.search(low)
                if bool_match:
                    cases[current_key]["task_success"] = normalize_bool(bool_match.group(1))
            elif "goal success" in low:
                bool_match = BOOL_RE.search(low)
                if bool_match:
                    cases[current_key]["goal_success"] = normalize_bool(bool_match.group(1))
            elif "execution success" in low:
                bool_match = BOOL_RE.search(low)
                if bool_match:
                    cases[current_key]["execution_success"] = normalize_bool(bool_match.group(1))

            if any(
                w in low
                for w in (
                    "error",
                    "fail",
                    "invalid",
                    "cannot",
                    "precondition",
                    "infeasible",
                    "did not pass",
                    "no prediction",
                    "name_id format",
                )
            ):
                if cases[current_key]["raw_failure_text"]:
                    cases[current_key]["raw_failure_text"] += " | "
                cases[current_key]["raw_failure_text"] += line

    for case in cases.values():
        if case["task_success"] is None:
            exec_success = case["execution_success"]
            goal_success = case["goal_success"]
            if exec_success is not None:
                case["task_success"] = exec_success
            elif goal_success is not None:
                case["task_success"] = goal_success

        failed = (
            (case["task_success"] is False)
            or (case["execution_success"] is False)
            or (case["raw_failure_text"] != "")
        )
        case["is_failed"] = bool(failed)
        case["failure_type"] = classify_failure(case["raw_failure_text"]) if failed else "none"
    return cases


def write_csv(rows: List[Dict], path: Path) -> None:
    fields = [
        "model",
        "task",
        "file_id",
        "goal_f1",
        "task_success",
        "goal_success",
        "execution_success",
        "failure_type",
        "raw_failure_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


FAILURE_TYPE_EXPLANATION = {
    "format_or_parsing": "Output is not a valid concatenated JSON action sequence.",
    "implicit_precondition": "An implicit precondition (e.g. OPEN before PUTIN) was not satisfied.",
    "relation_grounding": "Wrong object id or spatial relation prevented execution.",
    "planning_order": "Required actions appeared in the wrong order or a step was skipped.",
    "infeasible_or_execution": "Action was rejected as infeasible or non-executable.",
    "other": "Failure pattern not yet categorised.",
    "none": "No failure recorded.",
}


def _stringify_actions(actions: object) -> str:
    if isinstance(actions, list):
        rendered = []
        for item in actions:
            if isinstance(item, dict):
                rendered.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            else:
                rendered.append(str(item))
        return "\n".join(rendered) if rendered else "(empty)"
    if isinstance(actions, str):
        return actions
    return json.dumps(actions, ensure_ascii=False, indent=2)


def _truncate(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def write_case_study_md(
    rows: List[Dict],
    case_study_path: Path,
    case_study_top_n: int,
    error_info: Optional[Dict[str, Dict]] = None,
    gold_actions: Optional[Dict[str, Dict]] = None,
) -> None:
    error_info = error_info or {}
    gold_actions = gold_actions or {}
    candidates = [
        r
        for r in rows
        if (r["goal_f1"] or 0.0) > 0.0
        and (
            r.get("task_success") is False
            or r.get("execution_success") is False
            or str(r.get("raw_failure_text", "")).strip() != ""
        )
    ]
    candidates.sort(key=lambda r: (r["goal_f1"] or 0.0), reverse=True)
    cases = candidates[:case_study_top_n]

    lines = [
        "# Failure Case Studies",
        "",
        f"- selected cases: {len(cases)} (top-{case_study_top_n} by goal F1)",
        "- joined cases scanned: " + str(len(rows)),
        "",
        "These cases highlight failures where the model appeared to understand the",
        "goal (high goal F1) but its action sequence still failed. Each block lists",
        "the task, the failure category, the predicted action sequence taken from",
        "`error_info.json`, and (when available) the gold action sequence from the",
        "VirtualHome programs.",
        "",
    ]
    if not cases:
        lines.append("No qualifying cases found yet.")
        case_study_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for idx, case in enumerate(cases, start=1):
        file_id = case.get("file_id") or ""
        info = error_info.get(file_id, {})
        gold = gold_actions.get(file_id, {})
        predicted_text = _stringify_actions(info.get("actions")) if info else ""
        gold_text = _stringify_actions(gold.get("actions")) if gold else ""
        failure_type = case.get("failure_type", "other")
        lines.extend(
            [
                f"## Case {idx}: {case.get('task', '')} ({file_id})",
                "",
                f"- model: `{case.get('model', '')}`",
                f"- goal F1: {case.get('goal_f1', 0.0):.4f}",
                f"- task success: {case.get('task_success')}",
                f"- execution success: {case.get('execution_success')}",
                f"- failure type: `{failure_type}` — {FAILURE_TYPE_EXPLANATION.get(failure_type, '')}",
                f"- evaluator hint: {info.get('error_type') or 'n/a'}",
                "",
                "Predicted action sequence:",
                "",
                "```text",
                _truncate(predicted_text or "(unavailable)"),
                "```",
                "",
                "Ground-truth action sequence (if available):",
                "",
                "```text",
                _truncate(gold_text or "(unavailable)"),
                "```",
                "",
                "Evaluator log excerpt:",
                "",
                "```text",
                _truncate(str(case.get("raw_failure_text", ""))[:600]),
                "```",
                "",
            ]
        )
    case_study_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json_if_exists(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Link goal interpretation cases with action sequencing failures."
    )
    parser.add_argument("--goal-log", required=True)
    parser.add_argument("--action-log", required=True)
    parser.add_argument("--output-dir", default="output/diagnostics")
    parser.add_argument("--goal-threshold", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--case-study-top-n", type=int, default=5,
                        help="How many cases to write into failure_case_studies.md")
    parser.add_argument("--error-info-json", default=None,
                        help="Optional path to error_info.json for predicted action traces")
    parser.add_argument("--gold-actions-json", default=None,
                        help="Optional path to a JSON file mapping file_id -> {actions: [...]}")
    args = parser.parse_args()

    goal_log = Path(args.goal_log)
    action_log = Path(args.action_log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not goal_log.exists():
        raise FileNotFoundError(f"Goal log not found: {goal_log}")
    if not action_log.exists():
        raise FileNotFoundError(f"Action log not found: {action_log}")

    goal_cases = parse_goal_log(goal_log)
    action_cases = parse_action_log(action_log)

    goal_by_file_id: Dict[str, Dict] = {}
    for gc in goal_cases.values():
        fid = gc.get("file_id", "")
        if not fid:
            continue
        if fid not in goal_by_file_id or gc.get("goal_f1", 0.0) > goal_by_file_id[fid].get("goal_f1", 0.0):
            goal_by_file_id[fid] = gc

    joined_rows: List[Dict] = []
    for case_key, goal_case in goal_cases.items():
        action_case = action_cases.get(case_key)
        if not action_case:
            action_case = next(
                (ac for ac in action_cases.values() if ac.get("file_id") == goal_case.get("file_id")),
                None,
            )
        if not action_case:
            continue
        goal_model = goal_case.get("model", "unknown") or "unknown"
        action_model = action_case.get("model", "unknown") or "unknown"
        chosen_model = action_model if goal_model == "unknown" else goal_model
        row = {
            "model": chosen_model,
            "task": goal_case.get("task", ""),
            "file_id": goal_case.get("file_id", ""),
            "goal_f1": goal_case.get("goal_f1", 0.0),
            "task_success": action_case.get("task_success"),
            "goal_success": action_case.get("goal_success"),
            "execution_success": action_case.get("execution_success"),
            "failure_type": action_case.get("failure_type", "other"),
            "raw_failure_text": action_case.get("raw_failure_text", ""),
        }
        joined_rows.append(row)

    if not joined_rows:
        for ac in action_cases.values():
            fid = ac.get("file_id", "")
            gc = goal_by_file_id.get(fid)
            if not gc:
                continue
            chosen_model = ac.get("model") or gc.get("model") or "unknown"
            joined_rows.append(
                {
                    "model": chosen_model,
                    "task": ac.get("task", gc.get("task", "")),
                    "file_id": fid,
                    "goal_f1": gc.get("goal_f1", 0.0),
                    "task_success": ac.get("task_success"),
                    "goal_success": ac.get("goal_success"),
                    "execution_success": ac.get("execution_success"),
                    "failure_type": ac.get("failure_type", "other"),
                    "raw_failure_text": ac.get("raw_failure_text", ""),
                }
            )

    joined_rows.sort(key=lambda r: r["goal_f1"], reverse=True)
    all_joined_path = output_dir / "goal_action_joined_cases.csv"
    write_csv(joined_rows, all_joined_path)

    diagnostic_rows = [
        r
        for r in joined_rows
        if (
            r["goal_f1"] >= args.goal_threshold
            and (
                r["task_success"] is False
                or r["execution_success"] is False
                or str(r.get("raw_failure_text", "")).strip() != ""
            )
        )
    ]
    diagnostic_rows = diagnostic_rows[: args.top_n]

    top_cases_path = output_dir / "goal_correct_but_action_fail_top_cases.json"
    top_cases_path.write_text(
        json.dumps(diagnostic_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pattern_counter = Counter(
        r["failure_type"]
        for r in joined_rows
        if (
            r["task_success"] is False
            or r["execution_success"] is False
            or str(r.get("raw_failure_text", "")).strip() != ""
        )
    )
    pattern_path = output_dir / "failure_pattern_counts.json"
    pattern_path.write_text(
        json.dumps(pattern_counter, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Goal-to-Action Diagnostic Cases",
        "",
        f"- goal threshold: {args.goal_threshold}",
        f"- top_n: {args.top_n}",
        f"- joined cases: {len(joined_rows)}",
        f"- selected diagnostic cases: {len(diagnostic_rows)}",
        "",
        "## Cases",
        "",
    ]
    if diagnostic_rows:
        for idx, case in enumerate(diagnostic_rows, start=1):
            md_lines.extend(
                [
                    f"### Case {idx}",
                    f"- Task: {case['task']}",
                    f"- file_id: {case['file_id']}",
                    f"- Goal F1: {case['goal_f1']}",
                    f"- Task success: {case['task_success']}",
                    f"- Execution success: {case['execution_success']}",
                    f"- Failure type: {case['failure_type']}",
                    f"- Evidence: {case['raw_failure_text'][:300]}",
                    "",
                ]
            )
    else:
        md_lines.append("No qualifying cases found. Check action log format or lower threshold.")
        md_lines.append("")

    md_lines.append("## Failure Pattern Counts")
    md_lines.append("")
    if pattern_counter:
        for key, value in pattern_counter.most_common():
            md_lines.append(f"- {key}: {value}")
    else:
        md_lines.append("- none")

    md_path = output_dir / "goal_action_diagnostics.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    case_study_path = output_dir / "failure_case_studies.md"
    write_case_study_md(
        joined_rows,
        case_study_path,
        case_study_top_n=args.case_study_top_n,
        error_info=_load_json_if_exists(args.error_info_json),
        gold_actions=_load_json_if_exists(args.gold_actions_json),
    )

    print(f"[DONE] Wrote: {all_joined_path}")
    print(f"[DONE] Wrote: {top_cases_path}")
    print(f"[DONE] Wrote: {pattern_path}")
    print(f"[DONE] Wrote: {md_path}")
    print(f"[DONE] Wrote: {case_study_path}")


if __name__ == "__main__":
    main()
