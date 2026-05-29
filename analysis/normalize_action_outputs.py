#!/usr/bin/env python3
"""Normalize VirtualHome action-sequencing outputs into EAI name_id format.

This script rewrites each `<model>_outputs.json` row so that `llm_output`
follows the expected action format for `evaluate_results.py`:
`{"ACTION":["name","id"]}{"ACTION2":["name","id","name2","id2"]}...`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def prepend_src(repo_root: Path) -> None:
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def clean_output_text(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_actions(raw_text: str) -> List[Dict[str, List[str]]]:
    """Parse possibly-invalid JSON while preserving repeated action keys."""
    s = clean_output_text(raw_text)
    matches = re.findall(r'"(\w+)"\s*:\s*(\[[^\]]*\])', s)
    parsed: List[Dict[str, List[str]]] = []
    for key, arr_text in matches:
        try:
            arr = json.loads(arr_text)
            if isinstance(arr, list):
                parsed.append({key.upper(): [str(x) for x in arr]})
        except json.JSONDecodeError:
            continue
    if parsed:
        return parsed

    # Fallback for normal JSON object (without repeated keys)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    parsed.append({str(key).upper(): [str(x) for x in value]})
    except Exception:
        pass
    return parsed


def pick_symbolic_id(name: str, relevant_name_to_id: Dict[str, int]) -> str:
    prefix = f"{name}_"
    cands: List[Tuple[int, str]] = []
    for key in relevant_name_to_id.keys():
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix.isdigit():
                cands.append((int(suffix), suffix))
    if not cands:
        return "0"
    cands.sort(key=lambda x: x[0])
    return cands[0][1]


def normalize_params(params: List[str], relevant_name_to_id: Dict[str, int]) -> List[str]:
    if len(params) == 0:
        return []

    # Already name_id, keep as-is.
    if len(params) % 2 == 0 and all(str(params[i]).isdigit() for i in range(1, len(params), 2)):
        return [str(x) for x in params]

    # Otherwise treat entries as object names and inject symbolic ids.
    names: List[str] = [str(x).strip().lower().replace(" ", "_") for x in params if str(x).strip()]
    out: List[str] = []
    for name in names[:2]:
        out.extend([name, pick_symbolic_id(name, relevant_name_to_id)])
    return out


def actions_to_string(action_dicts: List[Dict[str, List[str]]]) -> str:
    parts: List[str] = []
    for d in action_dicts:
        for k, v in d.items():
            parts.append(json.dumps({k: v}, separators=(",", ":")))
    return "".join(parts)


def load_vh_context(repo_root: Path):
    prepend_src(repo_root)
    import virtualhome_eval.simulation.evolving_graph.utils as utils
    from virtualhome_eval.simulation.evolving_graph.eval_utils import construct_planner

    resource_root = repo_root / "src" / "virtualhome_eval" / "resources" / "virtualhome"
    data_dir = (
        repo_root
        / "src"
        / "virtualhome_eval"
        / "dataset"
        / "programs_processed_precond_nograb_morepreconds"
    )
    task_dict = json.loads((resource_root / "task_state_LTL_formula_accurate.json").read_text())
    id2task = json.loads((resource_root / "id2task.json").read_text())

    props = utils.load_properties_data()
    placing = utils.load_object_placing()
    name_eq = utils.load_name_equivalence()
    return construct_planner, task_dict["scene_1"], id2task, props, placing, name_eq, str(data_dir)


def build_relevant_name_to_id(
    file_id: str,
    construct_planner,
    task_dict_scene1: Dict,
    id2task: Dict,
    props,
    placing,
    name_eq,
    data_dir: str,
) -> Dict[str, int]:
    task = id2task[file_id]
    program_dict = task_dict_scene1[task][file_id]
    goals = program_dict["vh_goal"]
    gold_action_goals = list(set(goals["actions"]))

    gold_node_goals = []
    gold_edge_goals = []
    for g in goals["goal"]:
        if "id" in g and "class_name" in g and "state" in g:
            gold_node_goals.append(g)
        elif "from_id" in g and "to_id" in g and "relation_type" in g:
            gold_edge_goals.append(g)

    planner, _, _, _, _ = construct_planner(
        name_eq,
        props,
        placing,
        scenegraph_id=1,
        script_id=file_id,
        dataset_root=data_dir,
    )
    _, _, _, _, _, relevant_name_to_id = planner.get_symbolic_goal_nl(
        gold_node_goals, gold_edge_goals, action_goals=gold_action_goals
    )
    return relevant_name_to_id


def process_model_file(
    model_file: Path,
    out_file: Path,
    context_cache: Dict[str, Dict[str, int]],
    vh_context,
) -> Tuple[int, int]:
    (
        construct_planner,
        task_dict_scene1,
        id2task,
        props,
        placing,
        name_eq,
        data_dir,
    ) = vh_context

    rows = json.loads(model_file.read_text(encoding="utf-8"))
    total = 0
    converted = 0
    out_rows = []
    for row in rows:
        total += 1
        file_id = str(row.get("identifier", ""))
        if file_id not in context_cache:
            context_cache[file_id] = build_relevant_name_to_id(
                file_id,
                construct_planner,
                task_dict_scene1,
                id2task,
                props,
                placing,
                name_eq,
                data_dir,
            )
        relevant_name_to_id = context_cache[file_id]

        actions = parse_actions(str(row.get("llm_output", "")))
        normalized_actions: List[Dict[str, List[str]]] = []
        for action_dict in actions:
            for action, params in action_dict.items():
                normalized_actions.append(
                    {
                        action.upper(): normalize_params(params, relevant_name_to_id),
                    }
                )

        if normalized_actions:
            converted += 1
            llm_output = actions_to_string(normalized_actions)
        else:
            llm_output = str(row.get("llm_output", ""))

        out_rows.append({"identifier": file_id, "llm_output": llm_output})

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return total, converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("embodied-agent-interface-main"))
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_outputs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write normalized *_outputs.json",
    )
    parser.add_argument("--model", default="", help="Optional single model name without suffix")
    args = parser.parse_args()

    vh_context = load_vh_context(args.repo_root.resolve())
    cache: Dict[str, Dict[str, int]] = {}

    model_files = sorted(args.input_dir.glob("*_outputs.json"))
    if args.model:
        model_files = [args.input_dir / f"{args.model}_outputs.json"]

    if not model_files:
        raise FileNotFoundError(f"No *_outputs.json found in {args.input_dir}")

    for mf in model_files:
        out = args.output_dir / mf.name
        total, converted = process_model_file(mf, out, cache, vh_context)
        print(f"[DONE] {mf.name}: converted {converted}/{total} rows -> {out}")


if __name__ == "__main__":
    main()
