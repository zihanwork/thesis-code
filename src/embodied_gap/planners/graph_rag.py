from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.knowledge.retriever import adapt_plan
from embodied_gap.llm.clients import LLMClient, last_call_metadata
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.prompts import PLANNING_PROMPT_VERSION, render_planning_prompt

from .prompt_only import EngineeredPromptPlanner


@dataclass(frozen=True)
class GraphRetrievedExample:
    task: Task
    score: float
    edges: tuple[dict[str, str], ...]
    components: dict[str, float]


class GraphSubgraphRetriever:
    """Retrieve training task subgraphs from a frozen KG edge artifact."""

    def __init__(
        self,
        examples: list[Task],
        graph_path: str | Path,
    ) -> None:
        self.examples = {task.id: task for task in examples if task.gold_plan}
        self.graph_path = Path(graph_path)
        self.edges_by_task = self._load_edges(self.graph_path)

    def retrieve(self, query: Task, k: int = 1) -> list[GraphRetrievedExample]:
        if k <= 0:
            return []
        query_tokens = _task_tokens(query)
        scored: list[GraphRetrievedExample] = []
        for task_id, task in self.examples.items():
            if task_id == query.id:
                continue
            edges = self.edges_by_task.get(task_id, ())
            if not edges:
                continue
            graph_tokens = {
                token
                for edge in edges
                for token in _edge_tokens(edge)
            }
            overlap = len(query_tokens & graph_tokens)
            union = len(query_tokens | graph_tokens)
            score = overlap / union if union else 0.0
            relation_overlap = len(
                {edge["relation"] for edge in edges} & _task_relations(query)
            )
            scored.append(
                GraphRetrievedExample(
                    task=task,
                    score=round(score, 6),
                    edges=edges,
                    components={
                        "graph_token_jaccard": round(score, 6),
                        "shared_token_count": float(overlap),
                        "relation_overlap": float(relation_overlap),
                    },
                )
            )
        scored.sort(key=lambda item: (-item.score, item.task.id))
        return scored[:k]

    @staticmethod
    def _load_edges(path: Path) -> dict[str, tuple[dict[str, str], ...]]:
        if not path.exists():
            raise FileNotFoundError(f"GraphRAG knowledge graph not found: {path}")
        grouped: dict[str, list[dict[str, str]]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    edge = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid graph edge at {path}:{line_number}") from exc
                required = {"task_id", "source", "relation", "target"}
                if not required <= edge.keys():
                    missing = sorted(required - edge.keys())
                    raise ValueError(f"Graph edge missing {missing} at {path}:{line_number}")
                grouped.setdefault(str(edge["task_id"]), []).append(
                    {key: str(edge[key]) for key in required}
                )
        return {task_id: tuple(edges) for task_id, edges in grouped.items()}


class GraphRAGPlanner:
    """P2: graph-retrieved subgraph context followed by action generation."""

    name = "P2_graph_rag"

    def __init__(
        self,
        examples: list[Task],
        *,
        graph_path: str | Path = "data/knowledge/eai_train/kg_edges.jsonl",
        top_k: int = 1,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retriever = GraphSubgraphRetriever(examples, graph_path)
        self.graph_path = str(graph_path)
        self.top_k = top_k
        self.fallback = EngineeredPromptPlanner()
        self.llm_client = llm_client

    def plan(self, task: Task) -> PlanCandidate:
        retrieved = self.retriever.retrieve(task, k=self.top_k)
        context = self._render_graph_context(retrieved)
        prompt = render_planning_prompt(
            task,
            strategy="retrieval_augmented",
            extra_context=context,
            profile="engineered",
        )
        if self.llm_client and retrieved:
            return self._llm_plan(prompt, retrieved)

        if not retrieved:
            fallback = self.fallback.plan(task)
            return PlanCandidate(
                planner_name=self.name,
                actions=fallback.actions,
                raw_response=fallback.raw_response,
                prompt=prompt,
                metadata=self._base_metadata(
                    retrieved=[],
                    fallback=fallback.planner_name,
                ),
            )

        example = retrieved[0].task
        actions = adapt_plan(example, task)
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=str(list(actions)),
            prompt=prompt,
            metadata=self._base_metadata(retrieved),
        )

    def _llm_plan(self, prompt: str, retrieved: list[GraphRetrievedExample]) -> PlanCandidate:
        raw_response = self.llm_client.generate(prompt)
        call_metadata = last_call_metadata(self.llm_client)
        metadata = self._base_metadata(retrieved)
        metadata.update(
            {
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "llm_call": call_metadata,
            }
        )
        try:
            actions = parse_action_list(raw_response)
        except Exception as exc:  # noqa: BLE001 - parse failures are observations.
            actions = ()
            metadata["parse_error"] = str(exc)
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=raw_response,
            prompt=prompt,
            metadata=metadata,
        )

    def _base_metadata(
        self,
        retrieved: list[GraphRetrievedExample],
        *,
        fallback: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "planner_family": "graph_rag",
            "engine": "graph_subgraph_retrieval",
            "prompt_version": "p2_graph_rag_v1",
            "prompt_template_version": PLANNING_PROMPT_VERSION,
            "graph_path": self.graph_path,
            "graph_retrieval_top_k": self.top_k,
            "graph_retrieved_ids": [item.task.id for item in retrieved],
            "graph_retrieval_scores": [item.score for item in retrieved],
            "graph_retrieval_components": [item.components for item in retrieved],
            "graph_retrieved_edge_counts": [len(item.edges) for item in retrieved],
            "retrieval_corpus": "training_task_knowledge_graph",
        }
        if fallback:
            metadata["fallback"] = fallback
        return metadata

    def _render_graph_context(self, retrieved: list[GraphRetrievedExample]) -> str:
        if not retrieved:
            return "No graph subgraph matched above threshold."
        blocks: list[str] = []
        for index, item in enumerate(retrieved, start=1):
            action_chain = _action_chain(item.edges)
            triples = "\n".join(
                f"- {edge['source']} --{edge['relation']}--> {edge['target']}"
                for edge in item.edges
            )
            blocks.append(
                "\n".join(
                    [
                        f"Graph-retrieved subgraph {index}:",
                        f"Source task ID: {item.task.id}",
                        f"Graph match score: {item.score:.6f}",
                        "Graph triples:",
                        triples,
                        f"Graph action chain: {action_chain}",
                    ]
                )
            )
        return "\n\n".join(blocks)


def _task_tokens(task: Task) -> set[str]:
    values = [task.instruction, *task.initial_facts, *task.goal_facts, *task.allowed_actions]
    objects = task.metadata.get("objects", {})
    if isinstance(objects, dict):
        values.extend(str(value) for value in objects)
    return set(_tokens(" ".join(values)))


def _task_relations(task: Task) -> set[str]:
    relations = {"has_initial_fact", "has_goal_fact"}
    if task.gold_plan:
        relations.update({"has_gold_action", "next_action", "calls_action"})
    return relations


def _edge_tokens(edge: dict[str, str]) -> list[str]:
    return _tokens(" ".join(edge.values()))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _action_chain(edges: tuple[dict[str, str], ...]) -> list[str]:
    actions = [
        edge["source"].split(":", 3)[-1]
        for edge in edges
        if edge["relation"] == "calls_action"
    ]
    return actions
