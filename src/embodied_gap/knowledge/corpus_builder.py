from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_gap.core.action_schema import parse_call, predicate_name
from embodied_gap.core.task_schema import Task, dump_jsonl, load_tasks


class KnowledgeCorpusBuilder:
    """Build retrieval and KG artifacts from canonical task records."""

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = sorted(tasks, key=lambda task: task.id)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "KnowledgeCorpusBuilder":
        return cls(load_tasks(path))

    def export(self, out_dir: str | Path) -> dict[str, Any]:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = [self._retrieval_doc(task) for task in self.tasks]
        edges = [edge for task in self.tasks for edge in self._kg_edges(task)]

        docs_path = output_dir / "retrieval_corpus.jsonl"
        edges_path = output_dir / "kg_edges.jsonl"
        dump_jsonl(docs_path, docs)
        dump_jsonl(edges_path, edges)

        manifest = {
            "task_count": len(self.tasks),
            "retrieval_doc_count": len(docs),
            "kg_edge_count": len(edges),
            "files": {
                "retrieval_corpus": str(docs_path),
                "kg_edges": str(edges_path),
            },
            "policy": {
                "default_source": "rag_train split",
                "leakage_note": "Use train-split artifacts for planner RAG/KG in final experiments.",
            },
            "summary": self._summary(edges),
        }
        (output_dir / "knowledge_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def _retrieval_doc(self, task: Task) -> dict[str, Any]:
        objects = task.metadata.get("objects", {})
        object_terms = " ".join(sorted(objects)[:80]) if isinstance(objects, dict) else ""
        action_terms = " ".join(predicate_name(action) for action in task.gold_plan)
        goal_terms = " ".join(task.goal_facts)
        text = " ".join(
            part
            for part in [
                task.instruction,
                task.slots.get("dataset", ""),
                task.slots.get("task_family", ""),
                goal_terms,
                object_terms,
                action_terms,
            ]
            if part
        )
        return {
            "id": task.id,
            "task_id": task.id,
            "split": task.split,
            "dataset": task.slots.get("dataset"),
            "task_family": task.slots.get("task_family"),
            "instruction": task.instruction,
            "text": text,
            "goal_facts": list(task.goal_facts),
            "gold_plan": list(task.gold_plan),
            "metadata": {
                "difficulty": task.metadata.get("difficulty"),
                "object_count": task.metadata.get("object_count"),
                "goal_fact_count": len(task.goal_facts),
                "gold_plan_length": len(task.gold_plan),
            },
        }

    def _kg_edges(self, task: Task) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        dataset = task.slots.get("dataset", "unknown")
        task_family = task.slots.get("task_family", "unknown")
        task_node = f"task:{task.id}"
        edges.append(self._edge(task, task_node, "belongs_to_dataset", f"dataset:{dataset}"))
        edges.append(self._edge(task, task_node, "belongs_to_family", f"family:{dataset}:{task_family}"))

        objects = task.metadata.get("objects", {})
        if isinstance(objects, dict):
            for object_name, object_type in sorted(objects.items()):
                object_node = f"object:{dataset}:{object_name}"
                edges.append(self._edge(task, task_node, "has_object", object_node))
                edges.append(self._edge(task, object_node, "has_type", f"type:{dataset}:{object_type}"))

        for fact in task.initial_facts:
            fact_node = f"fact:{dataset}:{fact}"
            edges.append(self._edge(task, task_node, "has_initial_fact", fact_node))
            edges.append(
                self._edge(task, fact_node, "uses_predicate", f"predicate:{dataset}:{predicate_name(fact)}")
            )

        for fact in task.goal_facts:
            fact_node = f"fact:{dataset}:{fact}"
            edges.append(self._edge(task, task_node, "has_goal_fact", fact_node))
            edges.append(
                self._edge(task, fact_node, "uses_predicate", f"predicate:{dataset}:{predicate_name(fact)}")
            )

        previous_step_node = ""
        for index, action in enumerate(task.gold_plan):
            call = parse_call(action)
            action_node = f"action_call:{task.id}:{index}:{action}"
            edges.append(self._edge(task, task_node, "has_gold_action", action_node))
            edges.append(self._edge(task, action_node, "calls_action", f"action:{dataset}:{call.name}"))
            for arg in call.args:
                edges.append(self._edge(task, action_node, "uses_object", f"object:{dataset}:{arg}"))
            if previous_step_node:
                edges.append(self._edge(task, previous_step_node, "next_action", action_node))
            previous_step_node = action_node
        return edges

    def _edge(self, task: Task, source: str, relation: str, target: str) -> dict[str, str]:
        return {
            "task_id": task.id,
            "dataset": task.slots.get("dataset", "unknown"),
            "task_family": task.slots.get("task_family", "unknown"),
            "source": source,
            "relation": relation,
            "target": target,
        }

    def _summary(self, edges: list[dict[str, Any]]) -> dict[str, Any]:
        relations: dict[str, int] = {}
        by_dataset: dict[str, int] = {}
        for edge in edges:
            relation = str(edge["relation"])
            dataset = str(edge["dataset"])
            relations[relation] = relations.get(relation, 0) + 1
            by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        return {
            "relations": dict(sorted(relations.items())),
            "edges_by_dataset": dict(sorted(by_dataset.items())),
        }
