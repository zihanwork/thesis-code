from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from embodied_gap.core.plan_schema import PlanCandidate
from embodied_gap.core.task_schema import Task
from embodied_gap.knowledge.retriever import adapt_plan
from embodied_gap.llm.clients import LLMClient, last_call_metadata
from embodied_gap.llm.parsers import parse_action_list
from embodied_gap.llm.prompts import PLANNING_PROMPT_VERSION, render_planning_prompt

from .prompt_only import EngineeredPromptPlanner


_EMBEDDING_DIMENSION = 64
_RELATION_WEIGHTS = {
    "has_goal_fact": 1.6,
    "has_initial_fact": 1.4,
    "uses_predicate": 1.4,
    "calls_action": 1.3,
    "next_action": 1.3,
    "uses_object": 1.2,
    "has_object": 1.1,
    "has_type": 1.0,
    "belongs_to_family": 1.2,
    "belongs_to_dataset": 0.6,
    "has_gold_action": 1.1,
}


@dataclass(frozen=True)
class GraphRetrievedExample:
    task: Task
    score: float
    edges: tuple[dict[str, str], ...]
    components: dict[str, float]
    linked_entities: tuple[str, ...] = ()
    evidence_paths: tuple[tuple[str, ...], ...] = ()


class GraphSubgraphRetriever:
    """Retrieve task subgraphs from one global, relation-aware training graph."""

    def __init__(
        self,
        examples: list[Task],
        graph_path: str | Path,
    ) -> None:
        self.examples = {task.id: task for task in examples if task.gold_plan}
        self.graph_path = Path(graph_path)
        self.edges_by_task, self.edges = self._load_edges(self.graph_path)
        self.adjacency = self._build_adjacency(self.edges)
        self.nodes = tuple(sorted(self.adjacency))
        self.node_tokens = {node: frozenset(_tokens(node)) for node in self.nodes}
        self.task_nodes = {
            task_id: f"task:{task_id}"
            for task_id in self.examples
            if f"task:{task_id}" in self.adjacency
        }
        self.node_embeddings = self._relational_graph_embeddings()

    def retrieve(self, query: Task, k: int = 1) -> list[GraphRetrievedExample]:
        if k <= 0:
            return []
        linked = self._link_entities(query)
        if not linked:
            return []
        seed_weights = {node: score for node, score in linked}
        ppr = self._personalized_page_rank(seed_weights)
        query_embedding = self._query_embedding(query, linked)
        query_tokens = _task_tokens(query)
        query_relations = _query_relations(query)
        query_state = _state_signature(query)
        ranked: list[GraphRetrievedExample] = []

        for task_id, task_node in self.task_nodes.items():
            task = self.examples[task_id]
            edges = self.edges_by_task.get(task_id, ())
            graph_tokens = {
                token
                for edge in edges
                for token in _edge_tokens(edge)
            }
            lexical = _jaccard(query_tokens, graph_tokens)
            relation_overlap = len(query_relations & {edge["relation"] for edge in edges})
            relation_score = relation_overlap / max(1, len(query_relations))
            embedding_score = max(
                0.0,
                _cosine(query_embedding, self.node_embeddings.get(task_node, ())),
            )
            ppr_score = ppr.get(task_node, 0.0)
            state_score = _state_alignment(query_state, _state_signature(task))
            paths = self._evidence_paths(tuple(seed_weights), task_node, max_hops=3, limit=4)
            path_score = max(
                (1.0 / max(1, (len(path) - 1) // 2) for path in paths),
                default=0.0,
            )
            score = (
                0.20 * lexical
                + 0.20 * embedding_score
                + 0.25 * _bounded_ppr(ppr_score)
                + 0.15 * relation_score
                + 0.10 * path_score
                + 0.10 * state_score
            )
            ranked.append(
                GraphRetrievedExample(
                    task=task,
                    score=round(score, 6),
                    edges=edges,
                    components={
                        "lexical_jaccard": round(lexical, 6),
                        "graph_embedding_cosine": round(embedding_score, 6),
                        "personalized_page_rank": round(ppr_score, 8),
                        "relation_overlap": float(relation_overlap),
                        "relation_overlap_score": round(relation_score, 6),
                        "multi_hop_path_score": round(path_score, 6),
                        "state_constraint_score": round(state_score, 6),
                    },
                    linked_entities=tuple(node for node, _ in linked),
                    evidence_paths=paths,
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.task.id))
        return ranked[:k]

    def _link_entities(self, query: Task) -> list[tuple[str, float]]:
        dataset = str(query.slots.get("dataset", "unknown"))
        exact: dict[str, float] = {f"dataset:{dataset}": 0.4}
        objects = query.metadata.get("objects", {})
        if isinstance(objects, dict):
            for name, object_type in objects.items():
                exact[f"object:{dataset}:{name}"] = 1.0
                exact[f"type:{dataset}:{object_type}"] = 0.8
        for fact in (*query.initial_facts, *query.goal_facts):
            exact[f"predicate:{dataset}:{_predicate_name(fact)}"] = 1.0
            exact[f"fact:{dataset}:{fact}"] = 1.0
        for action in query.allowed_actions:
            exact[f"action:{dataset}:{_predicate_name(action)}"] = 0.8
        family = query.slots.get("task_family")
        if family:
            exact[f"family:{dataset}:{family}"] = 0.7

        linked = {node: weight for node, weight in exact.items() if node in self.adjacency}
        query_tokens = _task_tokens(query)
        for node in self.nodes:
            if node.startswith("task:") or node in linked:
                continue
            overlap = _jaccard(query_tokens, self.node_tokens[node])
            if overlap >= 0.5:
                linked[node] = max(linked.get(node, 0.0), overlap * 0.5)
        return sorted(linked.items(), key=lambda item: (-item[1], item[0]))[:32]

    def _personalized_page_rank(
        self,
        seeds: dict[str, float],
        *,
        damping: float = 0.85,
        iterations: int = 24,
    ) -> dict[str, float]:
        total = sum(seeds.values()) or 1.0
        personalization = {node: weight / total for node, weight in seeds.items()}
        rank = dict(personalization)
        for _ in range(iterations):
            updated = {node: (1.0 - damping) * weight for node, weight in personalization.items()}
            for node, score in rank.items():
                neighbors = self.adjacency.get(node, ())
                weight_sum = sum(weight for _, _, weight in neighbors)
                if not neighbors or weight_sum <= 0:
                    for seed, weight in personalization.items():
                        updated[seed] = updated.get(seed, 0.0) + damping * score * weight
                    continue
                for neighbor, _, weight in neighbors:
                    updated[neighbor] = updated.get(neighbor, 0.0) + damping * score * weight / weight_sum
            rank = updated
        return rank

    def _relational_graph_embeddings(self) -> dict[str, tuple[float, ...]]:
        embeddings = {node: _hashed_embedding(self.node_tokens[node]) for node in self.nodes}
        for _ in range(2):
            updated: dict[str, tuple[float, ...]] = {}
            for node in self.nodes:
                aggregate = [0.5 * value for value in embeddings[node]]
                total_weight = 0.5
                for neighbor, relation, weight in self.adjacency[node]:
                    relation_vector = _hashed_embedding((relation,))
                    neighbor_vector = embeddings[neighbor]
                    for index in range(_EMBEDDING_DIMENSION):
                        aggregate[index] += weight * (
                            0.8 * neighbor_vector[index] + 0.2 * relation_vector[index]
                        )
                    total_weight += weight
                updated[node] = _normalize(value / total_weight for value in aggregate)
            embeddings = updated
        return embeddings

    def _query_embedding(
        self,
        query: Task,
        linked: list[tuple[str, float]],
    ) -> tuple[float, ...]:
        base = list(_hashed_embedding(_task_tokens(query)))
        total_weight = 1.0
        for node, weight in linked:
            vector = self.node_embeddings.get(node)
            if not vector:
                continue
            for index, value in enumerate(vector):
                base[index] += weight * value
            total_weight += weight
        return _normalize(value / total_weight for value in base)

    def _evidence_paths(
        self,
        seeds: tuple[str, ...],
        target: str,
        *,
        max_hops: int,
        limit: int,
    ) -> tuple[tuple[str, ...], ...]:
        paths: list[tuple[str, ...]] = []
        for seed in seeds:
            if seed == target:
                paths.append((seed,))
                continue
            frontier = deque([(seed, (seed,), 0)])
            visited = {seed}
            while frontier:
                node, path, depth = frontier.popleft()
                if depth >= max_hops:
                    continue
                for neighbor, relation, _ in self.adjacency.get(node, ()):
                    if neighbor in visited:
                        continue
                    next_path = (*path, f"--{relation}-->", neighbor)
                    if neighbor == target:
                        paths.append(next_path)
                        frontier.clear()
                        break
                    visited.add(neighbor)
                    frontier.append((neighbor, next_path, depth + 1))
            if len(paths) >= limit:
                break
        return tuple(paths[:limit])

    @staticmethod
    def _build_adjacency(
        edges: Iterable[dict[str, str]],
    ) -> dict[str, tuple[tuple[str, str, float], ...]]:
        adjacency: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            relation = edge["relation"]
            weight = _RELATION_WEIGHTS.get(relation, 1.0)
            adjacency[source].append((target, relation, weight))
            adjacency[target].append((source, f"inverse:{relation}", weight * 0.9))
        return {
            node: tuple(sorted(neighbors, key=lambda item: (item[0], item[1])))
            for node, neighbors in adjacency.items()
        }

    @staticmethod
    def _load_edges(
        path: Path,
    ) -> tuple[dict[str, tuple[dict[str, str], ...]], tuple[dict[str, str], ...]]:
        if not path.exists():
            raise FileNotFoundError(f"GraphRAG knowledge graph not found: {path}")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        all_edges: list[dict[str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid graph edge at {path}:{line_number}") from exc
                required = {"task_id", "source", "relation", "target"}
                if not required <= raw.keys():
                    missing = sorted(required - raw.keys())
                    raise ValueError(f"Graph edge missing {missing} at {path}:{line_number}")
                edge = {key: str(raw[key]) for key in required}
                grouped[edge["task_id"]].append(edge)
                all_edges.append(edge)
        return (
            {task_id: tuple(edges) for task_id, edges in grouped.items()},
            tuple(all_edges),
        )


class GraphRAGPlanner:
    """P2: global graph retrieval followed by state-aware action generation."""

    name = "P2_graph_rag"

    def __init__(
        self,
        examples: list[Task],
        *,
        graph_path: str | Path = "data/knowledge/eai_train/kg_edges.jsonl",
        top_k: int = 3,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retriever = GraphSubgraphRetriever(examples, graph_path)
        self.graph_path = str(graph_path)
        self.top_k = top_k
        self.fallback = EngineeredPromptPlanner()
        self.llm_client = llm_client

    def plan(self, task: Task) -> PlanCandidate:
        retrieved = self.retriever.retrieve(task, k=self.top_k)
        context = self._render_graph_context(task, retrieved)
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
                metadata=self._base_metadata([], fallback=fallback.planner_name),
            )
        actions = adapt_plan(retrieved[0].task, task)
        return PlanCandidate(
            planner_name=self.name,
            actions=actions,
            raw_response=str(list(actions)),
            prompt=prompt,
            metadata=self._base_metadata(retrieved),
        )

    def _llm_plan(self, prompt: str, retrieved: list[GraphRetrievedExample]) -> PlanCandidate:
        raw_response = self.llm_client.generate(prompt)
        metadata = self._base_metadata(retrieved)
        metadata.update(
            {
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
                "llm_call": last_call_metadata(self.llm_client),
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
            "engine": "global_relation_aware_graph_retrieval",
            "prompt_version": "p2_graph_rag_global",
            "prompt_template_version": PLANNING_PROMPT_VERSION,
            "graph_path": self.graph_path,
            "graph_retrieval_top_k": self.top_k,
            "graph_retrieved_ids": [item.task.id for item in retrieved],
            "graph_retrieval_scores": [item.score for item in retrieved],
            "graph_retrieval_components": [item.components for item in retrieved],
            "graph_retrieved_edge_counts": [len(item.edges) for item in retrieved],
            "graph_linked_entities": [list(item.linked_entities) for item in retrieved],
            "graph_evidence_paths": [
                [list(path) for path in item.evidence_paths]
                for item in retrieved
            ],
            "graph_algorithms": [
                "entity_linking",
                "graph_neural_retrieval",
                "personalized_page_rank",
                "multi_hop_path_search",
                "relation_aware_reranking",
                "state_constraint_scoring",
            ],
            "graph_hyperparameters": {
                "embedding_dimension": _EMBEDDING_DIMENSION,
                "message_passing_layers": 2,
                "ppr_damping": 0.85,
                "ppr_iterations": 24,
                "max_path_hops": 3,
                "rerank_weights": {
                    "lexical": 0.20,
                    "graph_embedding": 0.20,
                    "personalized_page_rank": 0.25,
                    "relation_overlap": 0.15,
                    "multi_hop_path": 0.10,
                    "state_constraint": 0.10,
                },
            },
            "query_gold_plan_used": False,
            "retrieval_corpus": "training_global_knowledge_graph",
        }
        if fallback:
            metadata["fallback"] = fallback
        return metadata

    def _render_graph_context(
        self,
        query: Task,
        retrieved: list[GraphRetrievedExample],
    ) -> str:
        if not retrieved:
            return "No graph evidence matched the query entities and state constraints."
        blocks: list[str] = [
            "Use graph evidence only when it is compatible with the current initial state, goal, objects, and allowed actions."
        ]
        for index, item in enumerate(retrieved, start=1):
            triples = "\n".join(
                f"- {edge['source']} --{edge['relation']}--> {edge['target']}"
                for edge in item.edges
            )
            paths = "\n".join(
                f"- {' '.join(path)}"
                for path in item.evidence_paths
            ) or "- no path within three hops"
            chain = _action_chain(item.edges)
            blocks.append(
                "\n".join(
                    [
                        f"Graph-retrieved subgraph {index}:",
                        f"Source task ID: {item.task.id}",
                        f"Hybrid graph score: {item.score:.6f}",
                        f"Score components: {json.dumps(item.components, sort_keys=True)}",
                        f"Linked query entities: {list(item.linked_entities)}",
                        "Multi-hop evidence paths:",
                        paths,
                        "Graph triples:",
                        triples,
                        f"Graph action chain: {chain}",
                        f"Current initial facts: {list(query.initial_facts)}",
                        f"Current goal facts: {list(query.goal_facts)}",
                        f"Current allowed actions: {list(query.allowed_actions)}",
                    ]
                )
            )
        return "\n\n".join(blocks)


def _task_tokens(task: Task) -> set[str]:
    values = [task.instruction, *task.initial_facts, *task.goal_facts, *task.allowed_actions]
    objects = task.metadata.get("objects", {})
    if isinstance(objects, dict):
        values.extend(str(value) for value in objects)
        values.extend(str(value) for value in objects.values())
    return set(_tokens(" ".join(values)))


def _query_relations(task: Task) -> set[str]:
    relations = {"belongs_to_dataset", "has_initial_fact", "has_goal_fact", "uses_predicate"}
    if task.metadata.get("objects"):
        relations.update({"has_object", "has_type", "uses_object"})
    if task.allowed_actions:
        relations.add("calls_action")
    if task.slots.get("task_family"):
        relations.add("belongs_to_family")
    return relations


def _state_signature(task: Task) -> dict[str, set[str]]:
    return {
        "initial_predicates": {_predicate_name(fact) for fact in task.initial_facts},
        "goal_predicates": {_predicate_name(fact) for fact in task.goal_facts},
        "allowed_actions": {_predicate_name(action) for action in task.allowed_actions},
    }


def _state_alignment(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    scores = [
        _jaccard(left["initial_predicates"], right["initial_predicates"]),
        _jaccard(left["goal_predicates"], right["goal_predicates"]),
    ]
    if left["allowed_actions"] and right["allowed_actions"]:
        scores.append(_jaccard(left["allowed_actions"], right["allowed_actions"]))
    return sum(scores) / len(scores)


def _edge_tokens(edge: dict[str, str]) -> set[str]:
    return set(_tokens(" ".join(edge.values())))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _predicate_name(value: str) -> str:
    return value.strip().lower().split("(", 1)[0].split(" ", 1)[0]


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _hashed_embedding(tokens: Iterable[str]) -> tuple[float, ...]:
    vector = [0.0] * _EMBEDDING_DIMENSION
    for token in sorted(set(tokens)):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % _EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return _normalize(vector)


def _normalize(values: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(values)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(0.0 for _ in vector)
    return tuple(value / norm for value in vector)


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_vector = tuple(left)
    right_vector = tuple(right)
    if not left_vector or len(left_vector) != len(right_vector):
        return 0.0
    return sum(a * b for a, b in zip(left_vector, right_vector, strict=True))


def _bounded_ppr(value: float) -> float:
    return min(1.0, max(0.0, value * 20.0))


def _action_chain(edges: tuple[dict[str, str], ...]) -> list[str]:
    actions: list[tuple[int, str]] = []
    for edge in edges:
        if edge["relation"] != "calls_action":
            continue
        parts = edge["source"].split(":", 3)
        if len(parts) != 4:
            continue
        try:
            index = int(parts[2])
        except ValueError:
            continue
        actions.append((index, parts[3]))
    return [action for _, action in sorted(actions)]
