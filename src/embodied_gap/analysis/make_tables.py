from __future__ import annotations

from typing import Any


def summary_to_markdown(summary: dict[str, Any]) -> str:
    headers = [
        "method",
        "task_sr",
        "exec_sr",
        "safe_sr",
        "risk",
        "reject",
        "patches",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for method, row in sorted(summary.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    f"{row['task_success_rate']:.3f}",
                    f"{row['execution_success_rate']:.3f}",
                    f"{row['safe_success_rate']:.3f}",
                    f"{row['risk_rate']:.3f}",
                    f"{row['rejection_rate']:.3f}",
                    f"{row['patch_count_avg']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
