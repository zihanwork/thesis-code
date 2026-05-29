#!/usr/bin/env python3
"""
读取官方 generate_prompts 产出的 helm_prompt.json，调用 OpenAI 生成 llm_output，
写成 <model>_outputs.json（需环境变量 OPENAI_API_KEY）。

注意：模型未必严格遵守 name_id 格式；若 parsing 仍高，请缩小 --max-prompts 做调试，
或优先使用 build_action_sequencing_gold_outputs.py 作为「合法格式上界」对照。
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helm-prompt", required=True, help="helm_prompt.json 路径")
    parser.add_argument("--out-dir", required=True, help="helm_output/virtualhome/action_sequencing 目录")
    parser.add_argument("--model-name", default="openai_generated")
    parser.add_argument("--api-model", default="gpt-4o-mini")
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: 请设置 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: 请 pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=key)
    with open(args.helm_prompt, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    system = (
        "You output ONLY a compact JSON action sequence for VirtualHome. "
        "Format: concatenate one or more objects like "
        '{"WALK":["object_name","id"]}{"SWITCHON":["object_name","id"]} '
        "with NO spaces between objects. Keys are UPPERCASE action names. "
        "Each action value is a JSON array of strings: pairs of class_name and numeric id as strings. "
        "STANDUP uses empty array []. Two-object actions use 4 strings. Do not wrap in markdown."
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out: list[dict] = []

    for i, row in enumerate(prompts):
        if i >= args.max_prompts:
            break
        pid = row["identifier"]
        user = row["llm_prompt"]
        resp = client.chat.completions.create(
            model=args.api_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user[:120000]},
            ],
            temperature=0,
            max_tokens=2048,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        out.append({"identifier": pid, "llm_output": text})
        time.sleep(args.sleep)

    path = osp.join(args.out_dir, f"{args.model_name}_outputs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[DONE] wrote {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
