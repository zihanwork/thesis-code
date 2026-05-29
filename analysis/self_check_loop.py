#!/usr/bin/env python3
"""Failure-driven critique-rewrite loop for action sequencing outputs.

Given a baseline ``<model>_outputs.json`` and the matching evaluator
``error_info.json``, this script targets the failing rows, asks the
model to revise them, and writes a new ``<model>_self_check_outputs.json``
file.

The script is intentionally conservative: rows that already passed
under the baseline are copied over as-is, so the new file is a strict
superset of the baseline.  This keeps the EAI evaluator's success rate
monotonically non-decreasing as long as the rewrite does not introduce
new errors.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional

THIS_DIR = osp.dirname(osp.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from generate_outputs import PROVIDERS, _strip_code_fences  # type: ignore  # noqa: E402
from precondition_kg import PreconditionKG, Violation  # type: ignore  # noqa: E402
from scene_graph_rag import SceneGraphRetriever  # type: ignore  # noqa: E402

CRITIQUE_SYSTEM = (
    "You output ONLY a compact JSON action sequence for VirtualHome. "
    "No markdown, no explanations. Action names must be uppercase. "
    "Each value is a JSON array of strings: pairs of object_name and id. "
    "STANDUP uses []. WALK to an object before any action that uses it."
)

CRITIQUE_USER_TEMPLATE = (
    "You will revise a draft VirtualHome action sequence.\n\n"
    "Original task and constraints:\n{prompt}\n\n"
    "Draft action sequence:\n{draft}\n\n"
    "Evaluator feedback for the draft:\n{feedback}\n\n"
    "Apply these checks:\n"
    "1. Add missing WALK steps before any action that requires being NEAR the object.\n"
    "2. Reorder actions so preconditions are satisfied (e.g. OPEN before PUTIN, GRAB before PUTBACK).\n"
    "3. Replace hallucinated objects/ids with names and ids from the prompt.\n"
    "4. Remove redundant or repeated steps.\n"
    "5. Keep necessary action goals from the original task.\n\n"
    "Return ONLY the corrected compact JSON action sequence."
)


def _flatten_error_info(error_info: Dict) -> Dict[str, Dict]:
    flat: Dict[str, Dict] = {}
    for file_id, payload in error_info.items():
        if not isinstance(payload, dict):
            continue
        flat[str(file_id)] = payload
    return flat


def _format_feedback(payload: Dict) -> str:
    parts = []
    if payload.get("error_type"):
        parts.append(f"error_type: {payload['error_type']}")
    if payload.get("error_action"):
        parts.append(f"error_action: {payload['error_action']}")
    if payload.get("executable") is False:
        parts.append("executable: false")
    if payload.get("error_message"):
        parts.append(f"error_message: {payload['error_message']}")
    if not parts:
        parts.append("Unknown failure; please double-check action ordering and grounding.")
    return "; ".join(parts)


def _row_should_rewrite(payload: Dict) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("executable") is False:
        return True
    if payload.get("error_type"):
        return True
    return False


def _pc_kg_violations(
    identifier: str,
    draft: str,
    retriever: SceneGraphRetriever,
    kg: PreconditionKG,
) -> List[Violation]:
    scene_objects = retriever.load_scene_objects(identifier)
    return kg.verify(draft, scene_objects)


def _merge_feedback(evaluator_fb: str, kg_fb: str) -> str:
    if not evaluator_fb:
        return kg_fb
    if not kg_fb:
        return evaluator_fb
    return evaluator_fb + "\n" + kg_fb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS.keys()))
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--baseline-outputs", required=True,
                        help="Path to baseline <model>_outputs.json")
    parser.add_argument("--error-info", default=None,
                        help="Path to evaluator error_info.json; required unless --verifier pc_kg")
    parser.add_argument("--helm-prompt", required=True,
                        help="Path to EAI helm_prompt.json (for original prompt text)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant-label", default="self_check")
    parser.add_argument("--verifier", choices=("evaluator", "pc_kg", "both"),
                        default="evaluator",
                        help="Source of rewrite feedback: evaluator error_info, "
                             "precondition KG, or their union.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--max-rewrites", type=int, default=0,
                        help="0 means rewrite every failing row")
    parser.add_argument("--report", default=None,
                        help="Optional JSON path summarising rewrite outcomes")
    args = parser.parse_args()

    if args.verifier in ("evaluator", "both") and not args.error_info:
        parser.error("--error-info is required for verifier={evaluator,both}")

    options: Dict = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "api_key": os.environ.get(args.api_key_env) if args.api_key_env else None,
        "base_url": args.base_url,
    }

    with open(args.baseline_outputs, "r", encoding="utf-8") as f:
        baseline_rows = json.load(f)
    errors: Dict[str, Dict] = {}
    if args.error_info:
        with open(args.error_info, "r", encoding="utf-8") as f:
            errors = _flatten_error_info(json.load(f))
    with open(args.helm_prompt, "r", encoding="utf-8") as f:
        prompts = {str(row["identifier"]): row["llm_prompt"] for row in json.load(f)}

    provider_fn = PROVIDERS[args.provider]
    options["eval_type"] = "action_sequencing"

    retriever: Optional[SceneGraphRetriever] = None
    kg: Optional[PreconditionKG] = None
    if args.verifier in ("pc_kg", "both"):
        retriever = SceneGraphRetriever.shared()
        kg = PreconditionKG.shared()

    out_rows: List[Dict] = []
    rewrites = 0
    successes = 0
    skipped = 0
    violation_counts: Dict[str, int] = {}
    rewrite_success_by_code: Dict[str, Dict[str, int]] = {}

    for row in baseline_rows:
        identifier = str(row.get("identifier"))
        draft = row.get("llm_output", "")
        payload = errors.get(identifier, {})

        kg_violations: List[Violation] = []
        if retriever is not None and kg is not None:
            kg_violations = _pc_kg_violations(identifier, draft, retriever, kg)
            for v in kg_violations:
                violation_counts[v.code.value] = violation_counts.get(v.code.value, 0) + 1

        should_rewrite = False
        if args.verifier == "evaluator":
            should_rewrite = _row_should_rewrite(payload)
        elif args.verifier == "pc_kg":
            should_rewrite = bool(kg_violations)
        else:  # both
            should_rewrite = _row_should_rewrite(payload) or bool(kg_violations)

        if not should_rewrite:
            out_rows.append({"identifier": identifier, "llm_output": draft})
            skipped += 1
            continue

        if args.max_rewrites and rewrites >= args.max_rewrites:
            out_rows.append({"identifier": identifier, "llm_output": draft})
            continue

        evaluator_fb = _format_feedback(payload) if args.verifier != "pc_kg" and payload else ""
        kg_fb = PreconditionKG.summarise(kg_violations) if kg_violations else ""
        feedback = _merge_feedback(evaluator_fb, kg_fb) or (
            "Unknown failure; double-check action ordering and grounding."
        )
        prompt_text = prompts.get(identifier, "")
        user = CRITIQUE_USER_TEMPLATE.format(
            prompt=prompt_text[: 80000],
            draft=draft,
            feedback=feedback,
        )
        try:
            result = provider_fn(args.api_model, CRITIQUE_SYSTEM, user, options)
            new_text = _strip_code_fences(result.text)
        except Exception as exc:
            new_text = ""
            print(f"[WARN] {identifier} rewrite failed: {exc}", file=sys.stderr)
        rewrite_ok = False
        if new_text:
            out_rows.append({"identifier": identifier, "llm_output": new_text})
            successes += 1
            rewrite_ok = True
        else:
            out_rows.append({"identifier": identifier, "llm_output": draft})

        # track rewrite success per violation code
        for v in kg_violations:
            slot = rewrite_success_by_code.setdefault(
                v.code.value, {"attempts": 0, "successes": 0}
            )
            slot["attempts"] += 1
            if rewrite_ok:
                slot["successes"] += 1

        rewrites += 1
        if args.sleep:
            import time

            time.sleep(args.sleep)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = osp.join(
        args.out_dir,
        f"{args.model_name}_{args.variant_label}_outputs.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)
    print(
        f"[DONE] rewrote {successes}/{rewrites} rows; "
        f"left {skipped} passing rows untouched -> {out_path}"
    )

    if args.report:
        os.makedirs(osp.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model_name,
                    "variant": args.variant_label,
                    "verifier": args.verifier,
                    "total_rows": len(baseline_rows),
                    "skipped_passing": skipped,
                    "rewrite_attempts": rewrites,
                    "successful_rewrites": successes,
                    "violations_histogram": violation_counts,
                    "rewrite_success_by_violation_type": rewrite_success_by_code,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[DONE] report -> {args.report}")


if __name__ == "__main__":
    main()
