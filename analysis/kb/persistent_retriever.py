"""Persistent SG-RAG retriever backed by Chroma + Neo4j.

Drop-in replacement for ``analysis.scene_graph_rag.SceneGraphRetriever``:
exposes ``retrieve(identifier, task_prompt, k_neighbours, max_objects)``
and ``load_scene_objects(identifier)`` with identical signatures so
existing callers (planning / verification / reports) work unchanged.

Retrieval pipeline
------------------
1. Resolve the VirtualHome ``script_id`` from the prompt identifier
   (matches the legacy logic in ``scene_graph_rag``).
2. Use Chroma's ``scene_objects`` collection filtered by ``file_id`` to
   pull semantic seeds for the task prompt (BGE embeddings).
3. Expand seeds via Neo4j Cypher k-hop traversal of ``Object``/``RELATION``.
4. Reuse ``analysis.scene_graph_rag._serialise`` so the serialized
   subgraph format is byte-identical to the legacy retriever.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from analysis.scene_graph_rag import _Edge, _Node, _serialise
except ModuleNotFoundError:
    from scene_graph_rag import _Edge, _Node, _serialise  # type: ignore[no-redef]

from .config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    FAILURE_CASES_COLLECTION,
    MODELS_CACHE,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    SCENE_OBJECTS_COLLECTION,
)

log = logging.getLogger(__name__)

MAX_OBJECTS = 20
MAX_EDGES_PER_NODE = 4


def _extract_script_id(identifier: str) -> Optional[str]:
    if not identifier:
        return None
    m = re.match(r"^(\d+_\d+)", identifier.strip())
    return m.group(1) if m else None


class PersistentSceneGraphRetriever:
    """Chroma + Neo4j backed retriever (signature-compatible)."""

    _SHARED: Optional["PersistentSceneGraphRetriever"] = None

    def __init__(self) -> None:
        self._driver = None
        self._client = None
        self._embedder = None
        self._scene_coll = None

    # ------------------------------------------------------------------ public
    @classmethod
    def shared(cls) -> "PersistentSceneGraphRetriever":
        if cls._SHARED is None:
            cls._SHARED = cls()
        return cls._SHARED

    def retrieve(
        self,
        identifier: Optional[str],
        task_prompt: Optional[str] = None,
        k_neighbours: int = 1,
        max_objects: int = MAX_OBJECTS,
    ) -> Optional[str]:
        script_id = _extract_script_id(identifier or "")
        if script_id is None:
            return None

        seeds = self._semantic_seeds(script_id, task_prompt, max_objects)
        # Fall back: if Chroma returned nothing, use any node in the scene.
        all_nodes = self._scene_objects(script_id)
        if not all_nodes:
            log.warning("persistent retriever: no scene found for %s", script_id)
            return None
        if not seeds:
            seeds = {nid for nid in all_nodes.keys()}

        selected_ids = self._expand(script_id, seeds, k_neighbours, max_objects)
        sel_nodes = [all_nodes[i] for i in selected_ids if i in all_nodes]
        sel_edges = self._select_edges(script_id, selected_ids)
        return _serialise(sel_nodes, sel_edges)

    def load_scene_objects(self, identifier: Optional[str]) -> Dict[int, _Node]:
        script_id = _extract_script_id(identifier or "")
        if script_id is None:
            return {}
        return self._scene_objects(script_id)

    def query_similar_failures(
        self,
        text: str,
        n_results: int = 5,
        failure_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve past failure cases similar to ``text`` (diagnostic helper)."""
        client = self._chroma_client()
        try:
            coll = client.get_collection(name=FAILURE_CASES_COLLECTION)
        except Exception:
            return []
        embedder = self._embedder_instance()
        emb = embedder.encode([text], normalize_embeddings=True).tolist()
        kwargs: Dict[str, Any] = {"query_embeddings": emb, "n_results": n_results}
        if failure_type:
            kwargs["where"] = {"failure_type": failure_type}
        res = coll.query(**kwargs)
        out: List[Dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for uid, meta, doc, dist in zip(ids, metas, docs, dists):
            out.append({"uid": uid, "metadata": meta, "document": doc, "distance": dist})
        return out

    # ------------------------------------------------------------------ chroma
    def _chroma_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return self._client

    def _scene_collection(self):
        if self._scene_coll is None:
            self._scene_coll = self._chroma_client().get_collection(
                name=SCENE_OBJECTS_COLLECTION
            )
        return self._scene_coll

    def _embedder_instance(self):
        if self._embedder is None:
            import os
            from sentence_transformers import SentenceTransformer

            # Use cached model; avoid HuggingFace network calls at query time.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self._embedder = SentenceTransformer(
                EMBEDDING_MODEL, cache_folder=str(MODELS_CACHE)
            )
        return self._embedder

    def _semantic_seeds(
        self,
        script_id: str,
        task_prompt: Optional[str],
        max_objects: int,
    ) -> Set[int]:
        seeds: Set[int] = set()
        # Always include explicit ids "name(123)" mentioned in the prompt.
        if task_prompt:
            for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(\d+)\s*\)", task_prompt):
                try:
                    seeds.add(int(m.group(2)))
                except ValueError:
                    pass

        if not task_prompt:
            return seeds

        try:
            coll = self._scene_collection()
        except Exception as exc:
            log.warning("persistent retriever: chroma collection missing: %s", exc)
            return seeds
        embedder = self._embedder_instance()
        emb = embedder.encode([task_prompt], normalize_embeddings=True).tolist()
        try:
            res = coll.query(
                query_embeddings=emb,
                n_results=max_objects,
                where={"file_id": script_id},
            )
        except Exception as exc:
            log.warning("persistent retriever: chroma query failed: %s", exc)
            return seeds
        for meta in (res.get("metadatas") or [[]])[0]:
            nid = meta.get("node_id")
            if isinstance(nid, int):
                seeds.add(nid)
        return seeds

    # ------------------------------------------------------------------ neo4j
    def _neo4j_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
        return self._driver

    @lru_cache(maxsize=256)
    def _scene_objects(self, script_id: str) -> Dict[int, _Node]:
        driver = self._neo4j_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            res = session.run(
                """
                MATCH (s:Scene {file_id: $fid})-[:CONTAINS]->(o:Object)
                RETURN o.node_id AS id, o.class_name AS class_name,
                       o.properties AS properties, o.states AS states
                """,
                fid=script_id,
            )
            out: Dict[int, _Node] = {}
            for rec in res:
                nid = rec["id"]
                if nid is None:
                    continue
                out[int(nid)] = _Node(
                    id=int(nid),
                    class_name=str(rec.get("class_name") or ""),
                    properties=tuple(rec.get("properties") or ()),
                    states=tuple(rec.get("states") or ()),
                )
        return out

    def _expand(
        self,
        script_id: str,
        seeds: Set[int],
        k: int,
        max_objects: int,
    ) -> Set[int]:
        if not seeds:
            return set()
        driver = self._neo4j_driver()
        depth = max(0, int(k))
        query = (
            "MATCH (o:Object)-[:RELATION*0..%d]-(n:Object) "
            "WHERE o.file_id = $fid AND o.node_id IN $seeds "
            "AND n.file_id = $fid "
            "RETURN DISTINCT n.node_id AS id LIMIT $limit"
        ) % depth
        with driver.session(database=NEO4J_DATABASE) as session:
            res = session.run(
                query,
                fid=script_id,
                seeds=list(seeds),
                limit=max_objects * 4,
            )
            ids = [int(r["id"]) for r in res if r["id"] is not None]
        selected = set(ids) | set(seeds)
        if len(selected) > max_objects:
            seed_ordered = [i for i in sorted(seeds) if i in selected]
            extras = sorted(selected - set(seeds))
            trimmed = seed_ordered + extras
            selected = set(trimmed[:max_objects])
        return selected

    def _select_edges(
        self,
        script_id: str,
        selected_ids: Set[int],
    ) -> List[_Edge]:
        if not selected_ids:
            return []
        driver = self._neo4j_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            res = session.run(
                """
                MATCH (a:Object {file_id: $fid})-[r:RELATION]->(b:Object {file_id: $fid})
                WHERE a.node_id IN $ids AND b.node_id IN $ids
                RETURN a.node_id AS from_id, b.node_id AS to_id, r.type AS rel
                """,
                fid=script_id,
                ids=list(selected_ids),
            )
            rows: List[Tuple[int, int, str]] = [
                (int(r["from_id"]), int(r["to_id"]), str(r.get("rel") or ""))
                for r in res
            ]
        # Cap fan-out per source node to stay close to legacy serialisation.
        degree: Dict[int, int] = {}
        edges: List[_Edge] = []
        for fid, tid, rel in rows:
            if degree.get(fid, 0) >= MAX_EDGES_PER_NODE:
                continue
            edges.append(_Edge(from_id=fid, to_id=tid, relation_type=rel))
            degree[fid] = degree.get(fid, 0) + 1
        return edges

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


__all__ = ["PersistentSceneGraphRetriever"]
