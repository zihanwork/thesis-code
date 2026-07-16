from __future__ import annotations

from embodied_gap.core.task_schema import Task


def export_task_graph(task: Task) -> dict[str, object]:
    edges = []
    for action, spec in task.action_model.items():
        for fact in spec.preconditions:
            edges.append({"source": fact, "relation": "precondition_for", "target": action})
        for fact in spec.add_effects:
            edges.append({"source": action, "relation": "adds", "target": fact})
        for fact in spec.del_effects:
            edges.append({"source": action, "relation": "deletes", "target": fact})
    return {"task_id": task.id, "edges": edges}
