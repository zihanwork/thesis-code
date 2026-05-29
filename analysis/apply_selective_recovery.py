#!/usr/bin/env python3
"""Apply deterministic selective recovery to an existing action output file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from scene_graph_rag import SceneGraphRetriever
from selective_recovery import recover_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input *_outputs.json file")
    parser.add_argument("--output", type=Path, required=True, help="Output *_outputs.json file")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON recovery report path")
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    retriever = SceneGraphRetriever.shared()
    out_rows = []
    report_rows = []
    changed = 0
    inserted_total = 0
    skipped: Dict[str, int] = {}

    for row in rows:
        identifier = str(row.get("identifier", ""))
        original = str(row.get("llm_output", ""))
        scene_objects = retriever.load_scene_objects(identifier)
        recovered, report = recover_sequence(original, scene_objects)
        if report.changed:
            changed += 1
            inserted_total += report.inserted_actions
        else:
            reason = report.skipped_reason or "unchanged"
            skipped[reason] = skipped.get(reason, 0) + 1
        out_rows.append({"identifier": identifier, "llm_output": recovered})
        report_rows.append(
            {
                "identifier": identifier,
                "changed": report.changed,
                "inserted_actions": report.inserted_actions,
                "skipped_reason": report.skipped_reason,
                "inserted_steps": list(report.inserted_steps),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total_rows": len(rows),
        "changed_rows": changed,
        "inserted_actions": inserted_total,
        "skipped": skipped,
        "rows": report_rows,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[DONE] selective recovery changed {changed}/{len(rows)} rows; "
        f"inserted {inserted_total} actions -> {args.output}"
    )


if __name__ == "__main__":
    main()
