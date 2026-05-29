#!/usr/bin/env python3
"""
从 VirtualHome 官方脚本轨迹构造「合法」action_sequencing 的 llm_output，
写入 helm_output/virtualhome/action_sequencing/<model>_outputs.json，
供 eai-eval evaluate_results 使用（用于验证管线 / 论文中的 oracle 上界对照）。

说明：这不是「模型预测」，而是 gold program 的格式对齐版本；汇报中应明确标注为
gold_oracle 或 upper-bound baseline。
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys


def _prepend_src(repo_root: str) -> None:
    src = osp.join(repo_root, "src")
    if osp.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def gd_line_to_action_dict(line: str) -> dict | None:
    line = line.strip()
    m = re.match(r"^\[(\w+)\]\s*(.*)$", line)
    if not m:
        return None
    action, rest = m.group(1).upper(), m.group(2).strip()
    pairs = re.findall(r"<([^>]+)>\s*\((\d+)\)", rest)
    if not pairs:
        return {action: []}
    params: list[str] = []
    for name, oid in pairs:
        params.extend([name, str(int(oid))])
    return {action: params}


def actions_to_llm_output_string(action_dicts: list[dict]) -> str:
    """与 virtualhome_eval.eval_utils.load_json_preserving_order 兼容的拼接串。"""
    parts = []
    for d in action_dicts:
        for k, v in d.items():
            parts.append(json.dumps({k: v}, separators=(",", ":")))
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=osp.join(osp.dirname(__file__), "..", "embodied-agent-interface-main"),
        help="embodied-agent-interface 源码根目录（含 src/virtualhome_eval）",
    )
    parser.add_argument(
        "--out-helm-root",
        default=osp.join(osp.dirname(__file__), "..", "output", "legal_helm_output"),
        help="将写入 <out>/helm_output/virtualhome/action_sequencing/",
    )
    parser.add_argument("--model-name", default="gold_oracle")
    parser.add_argument("--scene-id", type=int, default=1)
    args = parser.parse_args()

    repo_root = osp.abspath(args.repo_root)
    _prepend_src(repo_root)

    import virtualhome_eval.simulation.evolving_graph.utils as utils
    from virtualhome_eval.simulation.evolving_graph.eval_utils import (
        construct_planner,
        remove_duplicate_dicts,
        scene_evaluate_wID,
    )

    resource_root = osp.join(repo_root, "src", "virtualhome_eval", "resources", "virtualhome")
    data_dir = osp.join(
        repo_root,
        "src",
        "virtualhome_eval",
        "dataset",
        "programs_processed_precond_nograb_morepreconds",
    )
    task_dict_path = osp.join(resource_root, "task_state_LTL_formula_accurate.json")
    id2task_path = osp.join(resource_root, "id2task.json")

    task_dicts_all = json.load(open(task_dict_path, "r", encoding="utf-8"))
    id2task = json.load(open(id2task_path, "r", encoding="utf-8"))
    scene_id = f"scene_{args.scene_id}"
    task_dict = task_dicts_all[scene_id]

    properties_data = utils.load_properties_data()
    object_placing = utils.load_object_placing()
    name_equivalence = utils.load_name_equivalence()

    outputs: list[dict] = []

    for task_name, task_dicts in task_dict.items():
        for file_id, program_dict in task_dicts.items():
            goals = program_dict["vh_goal"]
            gold_action_goals = goals["actions"]
            scene_goals = goals["goal"]
            gold_node_goals = []
            gold_edge_goals = []
            for scene_goal in scene_goals:
                if "id" in scene_goal and "class_name" in scene_goal and "state" in scene_goal:
                    gold_node_goals.append(scene_goal)
                elif "from_id" in scene_goal and "to_id" in scene_goal and "relation_type" in scene_goal:
                    gold_edge_goals.append(scene_goal)
                else:
                    raise ValueError("Scene goal is not in correct format")

            gold_node_goals = remove_duplicate_dicts(gold_node_goals)
            gold_edge_goals = remove_duplicate_dicts(gold_edge_goals)
            gold_action_goals = list(set(gold_action_goals))

            motion_planner, relevant_id, gd_actions, _tn, _td = construct_planner(
                name_equivalence,
                properties_data,
                object_placing,
                scenegraph_id=args.scene_id,
                script_id=file_id,
                dataset_root=data_dir,
            )
            _, _, _, all_success, _, _, _ = scene_evaluate_wID(
                motion_planner.final_state_dict,
                gold_node_goals,
                gold_edge_goals,
                motion_planner.acting_char_id,
            )
            if not all_success:
                continue

            action_dicts: list[dict] = []
            for cmd in gd_actions:
                d = gd_line_to_action_dict(cmd)
                if d:
                    action_dicts.append(d)

            llm_out = actions_to_llm_output_string(action_dicts)
            outputs.append({"identifier": file_id, "llm_output": llm_out})

    out_dir = osp.join(args.out_helm_root, "helm_output", "virtualhome", "action_sequencing")
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"{args.model_name}_outputs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Wrote {len(outputs)} programs to {out_path}")
    print(f"[HINT] Evaluate with:")
    print(
        f'  LLM_RESPONSE_PATH="{osp.abspath(args.out_helm_root)}/helm_output" '
        f"conda run -n eai-eval eai-eval --dataset virtualhome --eval-type action_sequencing "
        f'--mode evaluate_results --output-dir output --num-workers 1'
    )


if __name__ == "__main__":
    main()
