#!/usr/bin/env python3
"""Scene-Graph Retrieval-Augmented Grounding (SG-RAG) for VirtualHome.

Loads the raw VirtualHome init scene graph for a given task identifier
(e.g. ``"11_1"``), extracts a compact subgraph focused on objects that are
relevant to the task (task description mentions + vh_goal referenced ids),
and returns a serialised ``[Scene Subgraph] ... [/Scene Subgraph]`` block
that can be injected into the user prompt before the LLM call.

Design notes
------------
- The retriever is deliberately lightweight: no embedding model, no vector
  DB; relevance is driven by (i) object class names referenced in the task
  prompt, (ii) ids referenced by the task's ``vh_goal``, and (iii) their
  k-hop neighbours in the scene graph. This keeps SG-RAG reproducible on
  the dry-run provider and avoids a new dependency.
- Token budget is enforced by an object count cap rather than tokenisation
  so we stay provider-agnostic.
- The retriever is cheap enough to create a single shared instance per
  process; see ``SceneGraphRetriever.shared()``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EAI_ROOT = REPO_ROOT / "embodied-agent-interface-main" / "src" / "virtualhome_eval"
DATASET_ROOT = EAI_ROOT / "dataset" / "programs_processed_precond_nograb_morepreconds"
SCENE_GRAPH_ROOT = DATASET_ROOT / "init_and_final_graphs"
TASK_STATE_PATH = (
    EAI_ROOT / "resources" / "virtualhome" / "task_state_LTL_formula_accurate.json"
)

DEFAULT_SCENE_ID = 1
MAX_OBJECTS = 20           # hard cap on serialised objects
MAX_EDGES_PER_NODE = 4     # hard cap on neighbours per object


@dataclass(frozen=True)
class _Node:
    id: int
    class_name: str
    properties: Tuple[str, ...]
    states: Tuple[str, ...]


@dataclass(frozen=True)
class _Edge:
    from_id: int
    to_id: int
    relation_type: str


class SceneGraphRetriever:
    """Load VirtualHome init scene graphs and retrieve task-relevant subgraphs."""

    _SHARED: Optional["SceneGraphRetriever"] = None

    def __init__(
        self,
        scene_graph_root: Path = SCENE_GRAPH_ROOT,
        task_state_path: Path = TASK_STATE_PATH,
    ) -> None:
        self.scene_graph_root = Path(scene_graph_root)
        self.task_state_path = Path(task_state_path)
        self._goal_index: Dict[str, dict] = self._build_goal_index()

    # ------------------------------------------------------------------ public
    @classmethod
    def shared(cls) -> "SceneGraphRetriever":
        if cls._SHARED is None:
            backend = os.environ.get("KB_BACKEND", "default").lower()
            if backend == "persistent":
                try:
                    from analysis.kb.persistent_retriever import (
                        PersistentSceneGraphRetriever,
                    )
                    cls._SHARED = PersistentSceneGraphRetriever.shared()  # type: ignore[assignment]
                except Exception as exc:
                    log.warning(
                        "scene_graph_rag: persistent backend unavailable (%s); "
                        "falling back to file-based retriever", exc,
                    )
                    cls._SHARED = cls()
            else:
                cls._SHARED = cls()
        return cls._SHARED

    def retrieve(
        self,
        identifier: Optional[str],
        task_prompt: Optional[str] = None,
        k_neighbours: int = 1,
        max_objects: int = MAX_OBJECTS,
    ) -> Optional[str]:
        """Return a serialised subgraph block, or ``None`` if unavailable.

        Parameters
        ----------
        identifier
            Prompt identifier from ``helm_prompt.json`` (e.g. ``"11_1"``).
        task_prompt
            Optional user prompt text; class names appearing in it are
            added as seed objects. When ``None`` we fall back to the ids
            referenced by ``vh_goal``.
        """
        if not identifier:
            return None
        script_id = _extract_script_id(identifier)
        if script_id is None:
            return None
        graph = self._load_graph(script_id)
        if graph is None:
            log.warning("scene_graph_rag: scene graph not found for %s", identifier)
            return None
        nodes, edges = graph

        seeds = self._seed_ids(script_id, task_prompt, nodes)
        if not seeds:
            # Fall back to every node referenced in the prompt by class_name.
            seeds = {n.id for n in nodes}

        selected_ids = self._expand(seeds, edges, k_neighbours, max_objects)
        sel_nodes = [n for n in nodes if n.id in selected_ids]
        sel_edges = self._select_edges(selected_ids, edges)
        return _serialise(sel_nodes, sel_edges)

    def load_scene_objects(self, identifier: Optional[str]) -> Dict[int, _Node]:
        """Return ``{id: _Node}`` for PC-KG verification.

        Returns an empty dict when the scene graph cannot be resolved.
        """
        if not identifier:
            return {}
        script_id = _extract_script_id(identifier)
        if script_id is None:
            return {}
        graph = self._load_graph(script_id)
        if graph is None:
            return {}
        nodes, _ = graph
        return {n.id: n for n in nodes}

    # ---------------------------------------------------------------- internals
    def _build_goal_index(self) -> Dict[str, dict]:
        if not self.task_state_path.is_file():
            return {}
        try:
            blob = json.loads(self.task_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("scene_graph_rag: failed to load %s: %s", self.task_state_path, exc)
            return {}
        out: Dict[str, dict] = {}
        for scene, tasks in blob.items():
            if not isinstance(tasks, dict):
                continue
            for task_name, files in tasks.items():
                if not isinstance(files, dict):
                    continue
                for file_id, payload in files.items():
                    out[str(file_id)] = {
                        "scene": scene,
                        "task_name": task_name,
                        "payload": payload,
                    }
        return out

    @lru_cache(maxsize=256)
    def _load_graph(self, script_id: str) -> Optional[Tuple[Tuple[_Node, ...], Tuple[_Edge, ...]]]:
        scene_id = self._scene_id_for(script_id)
        path = (
            self.scene_graph_root
            / f"TrimmedTestScene{scene_id}_graph"
            / "results_intentions_march-13-18"
            / f"file{script_id}.json"
        )
        if not path.is_file():
            # Fallback: scan all scene dirs (some scripts live in scene_2, etc.).
            for scene_dir in sorted(self.scene_graph_root.glob("TrimmedTestScene*_graph")):
                cand = scene_dir / "results_intentions_march-13-18" / f"file{script_id}.json"
                if cand.is_file():
                    path = cand
                    break
            else:
                return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("scene_graph_rag: failed to load %s: %s", path, exc)
            return None
        init_graph = blob.get("init_graph") or {}
        nodes_raw = init_graph.get("nodes") or []
        edges_raw = init_graph.get("edges") or []
        nodes = tuple(
            _Node(
                id=int(n.get("id")),
                class_name=str(n.get("class_name", "")),
                properties=tuple(n.get("properties") or ()),
                states=tuple(n.get("states") or ()),
            )
            for n in nodes_raw
            if n.get("id") is not None
        )
        edges = tuple(
            _Edge(
                from_id=int(e.get("from_id")),
                to_id=int(e.get("to_id")),
                relation_type=str(e.get("relation_type", "")),
            )
            for e in edges_raw
            if e.get("from_id") is not None and e.get("to_id") is not None
        )
        return nodes, edges

    def _scene_id_for(self, script_id: str) -> int:
        record = self._goal_index.get(script_id)
        if record is None:
            return DEFAULT_SCENE_ID
        scene = record.get("scene", "")
        m = re.search(r"(\d+)", scene)
        return int(m.group(1)) if m else DEFAULT_SCENE_ID

    def _seed_ids(
        self,
        script_id: str,
        task_prompt: Optional[str],
        nodes: Iterable[_Node],
    ) -> Set[int]:
        seeds: Set[int] = set()
        record = self._goal_index.get(script_id)
        if record is not None:
            vh_goal = record.get("payload", {}).get("vh_goal", {})
            for item in vh_goal.get("goal", []) or []:
                for k in ("id", "from_id", "to_id"):
                    v = item.get(k)
                    if isinstance(v, int):
                        seeds.add(v)
        if task_prompt:
            lowered = task_prompt.lower()
            # Also catch compact "floor_lamp(1000)" style ids.
            for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(\d+)\s*\)", task_prompt):
                try:
                    seeds.add(int(m.group(2)))
                except ValueError:
                    pass
            for n in nodes:
                if n.class_name and n.class_name.lower() in lowered:
                    seeds.add(n.id)
        return seeds

    def _expand(
        self,
        seeds: Set[int],
        edges: Iterable[_Edge],
        k: int,
        max_objects: int,
    ) -> Set[int]:
        frontier = set(seeds)
        selected = set(seeds)
        edges_list = list(edges)
        for _ in range(max(0, k)):
            next_frontier: Set[int] = set()
            for e in edges_list:
                if e.from_id in frontier and e.to_id not in selected:
                    next_frontier.add(e.to_id)
                if e.to_id in frontier and e.from_id not in selected:
                    next_frontier.add(e.from_id)
            if not next_frontier:
                break
            selected |= next_frontier
            frontier = next_frontier
            if len(selected) >= max_objects:
                break
        if len(selected) > max_objects:
            # Keep seeds first, then deterministically trim by id for reproducibility.
            seed_ordered = [i for i in sorted(seeds) if i in selected]
            extras = [i for i in sorted(selected - set(seeds))]
            trimmed = seed_ordered + extras
            selected = set(trimmed[:max_objects])
        return selected

    @staticmethod
    def _select_edges(
        selected_ids: Set[int],
        edges: Iterable[_Edge],
    ) -> List[_Edge]:
        degree: Dict[int, int] = {}
        out: List[_Edge] = []
        for e in edges:
            if e.from_id in selected_ids and e.to_id in selected_ids:
                if degree.get(e.from_id, 0) >= MAX_EDGES_PER_NODE:
                    continue
                out.append(e)
                degree[e.from_id] = degree.get(e.from_id, 0) + 1
        return out


# ------------------------------------------------------------------- helpers
def _extract_script_id(identifier: str) -> Optional[str]:
    """Prompt identifier already equals VirtualHome script_id (e.g. ``11_1``)."""
    if not identifier:
        return None
    m = re.match(r"^(\d+_\d+)", identifier.strip())
    return m.group(1) if m else None


def _serialise(nodes: List[_Node], edges: List[_Edge]) -> str:
    if not nodes:
        return ""
    lines: List[str] = ["[Scene Subgraph]", "Objects:"]
    for n in sorted(nodes, key=lambda x: x.id):
        props = ", ".join(sorted(n.properties)) if n.properties else "-"
        states = ", ".join(sorted(n.states)) if n.states else "-"
        lines.append(
            f"- {n.class_name}(id={n.id}) properties=[{props}] states=[{states}]"
        )
    if edges:
        lines.append("Relations:")
        name_by_id = {n.id: n.class_name for n in nodes}
        for e in edges:
            a = f"{name_by_id.get(e.from_id, '?')}({e.from_id})"
            b = f"{name_by_id.get(e.to_id, '?')}({e.to_id})"
            lines.append(f"- {a} --{e.relation_type}-- {b}")
    lines.append("[/Scene Subgraph]")
    return "\n".join(lines)


def retrieve_subgraph(identifier: Optional[str], task_prompt: Optional[str] = None) -> Optional[str]:
    """Module-level convenience wrapper around ``SceneGraphRetriever.shared``."""
    return SceneGraphRetriever.shared().retrieve(identifier, task_prompt=task_prompt)


__all__ = [
    "SceneGraphRetriever",
    "retrieve_subgraph",
]
