from __future__ import annotations


def figure_manifest() -> dict[str, str]:
    return {
        "planner_by_harness": "Use summary.json to plot success rates for each P/H cell.",
        "error_breakdown": "Use metrics.jsonl error_counts to plot EAI-style error taxonomy.",
    }
