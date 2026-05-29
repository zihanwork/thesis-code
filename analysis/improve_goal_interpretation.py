#!/usr/bin/env python3
"""Run goal-interpretation prompt variants and validate JSON output.

This is a thin wrapper around :mod:`generate_outputs` that:

* picks one of the goal-interpretation variants from
  :mod:`prompt_variants`,
* generates outputs into the EAI ``goal_interpretation`` directory,
* sanity-checks the JSON shape so that EAI's evaluator can score
  ``all_f1`` / ``node_f1`` / ``edge_f1`` / ``action_f1``.

Usage example::

    python analysis/improve_goal_interpretation.py \
        --provider openai --api-model gpt-4o-mini --model-name gpt-4o-mini \
        --variant schema_constrained \
        --helm-prompt output/virtualhome/generate_prompts/goal_interpretation/helm_prompt.json \
        --out-dir output/api_helm/helm_output/virtualhome/goal_interpretation \
        --max-prompts 0
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Tuple

THIS_DIR = osp.dirname(osp.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from generate_outputs import (  # type: ignore  # noqa: E402
    PROVIDERS,
    _generate_one,
    _outputs_filename,
)
from prompt_variants import GOAL_VARIANTS, get_variant  # type: ignore  # noqa: E402

REQUIRED_KEYS = ("node goals", "edge goals", "action goals")
NODE_STATES = {
    "CLOSED",
    "OPEN",
    "ON",
    "OFF",
    "SITTING",
    "DIRTY",
    "CLEAN",
    "LYING",
    "PLUGGED_IN",
    "PLUGGED_OUT",
}
EDGE_RELATIONS = {"ON", "INSIDE", "BETWEEN", "CLOSE", "FACING", "HOLDS_RH", "HOLDS_LH"}


def parse_goal_json(text: str) -> Tuple[Optional[Dict], List[str]]:
    issues: List[str] = []
    if not text:
        return None, ["empty_output"]
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, [f"json_decode:{exc.msg}"]
    if not isinstance(obj, dict):
        return None, ["not_a_dict"]
    for key in REQUIRED_KEYS:
        if key not in obj:
            issues.append(f"missing_key:{key}")
            obj[key] = []
        elif not isinstance(obj[key], list):
            issues.append(f"not_a_list:{key}")
            obj[key] = []
    for goal in obj.get("node goals", []):
        if not isinstance(goal, dict) or "state" not in goal or "name" not in goal:
            issues.append("node_goal_shape")
        elif goal["state"] not in NODE_STATES:
            issues.append(f"node_state:{goal['state']}")
    for goal in obj.get("edge goals", []):
        if not isinstance(goal, dict) or not {"from_name", "relation", "to_name"} <= set(goal):
            issues.append("edge_goal_shape")
        elif goal["relation"] not in EDGE_RELATIONS:
            issues.append(f"edge_relation:{goal['relation']}")
    for goal in obj.get("action goals", []):
        if not isinstance(goal, dict) or "action" not in goal:
            issues.append("action_goal_shape")
    return obj, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS.keys()))
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(GOAL_VARIANTS.keys()))
    parser.add_argument("--helm-prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--validation-report", default=None,
                        help="Optional JSON path for per-row validation issues")
    args = parser.parse_args()

    variant = get_variant("goal_interpretation", args.variant)
    options: Dict = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "api_key": os.environ.get(args.api_key_env) if args.api_key_env else None,
        "base_url": args.base_url,
    }

    with open(args.helm_prompt, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = osp.join(args.out_dir, _outputs_filename(args.model_name, variant.label))

    out_rows: List[Dict] = []
    issues_per_row: List[Dict] = []
    issue_counts: Dict[str, int] = {}

    limit = args.max_prompts if args.max_prompts and args.max_prompts > 0 else len(prompts)
    for i, row in enumerate(prompts):
        if i >= limit:
            break
        identifier = str(row.get("identifier"))
        try:
            text = _generate_one(
                args.provider,
                args.api_model,
                variant,
                row.get("llm_prompt", ""),
                options,
                "goal_interpretation",
            )
        except Exception as exc:
            text = ""
            print(f"[WARN] {identifier} failed: {exc}", file=sys.stderr)
        parsed, issues = parse_goal_json(text)
        if parsed is not None:
            text = json.dumps(parsed, ensure_ascii=False)
        out_rows.append({"identifier": identifier, "llm_output": text})
        issues_per_row.append({"identifier": identifier, "issues": issues})
        for issue in issues:
            key = issue.split(":", 1)[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    if args.validation_report:
        report_path = args.validation_report
        os.makedirs(osp.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "variant": variant.label,
                    "model": args.model_name,
                    "issue_counts": issue_counts,
                    "rows": issues_per_row,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[DONE] validation -> {report_path}")
    print(f"[DONE] wrote {len(out_rows)} rows -> {out_path}")
    if issue_counts:
        print("[INFO] issue summary:", json.dumps(issue_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
