"""Build the persistent Chroma vector store for SG-RAG.

Two collections:

- ``scene_objects``  one document per (file_id, node_id); used as the
  semantic seed for retrieval. The document text is a compact natural
  language description (class + properties + states) that BGE encodes
  well.
- ``failure_cases``  one document per LLM failure case in
  ``output/diagnostics``; used to retrieve similar past failures during
  diagnostic and self-check loops.

Run once::

    python -m analysis.kb.build_vector_store
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import (
    CHROMA_DIR,
    DIAGNOSTICS_DIR,
    EMBEDDING_MODEL,
    FAILURE_CASES_COLLECTION,
    MODELS_CACHE,
    SCENE_GRAPH_ROOT,
    SCENE_OBJECTS_COLLECTION,
    ensure_dirs,
)

log = logging.getLogger("kb.build_vector_store")

# Chroma's metadata values must be primitives; we keep a separate document
# string for the human-readable form.


def _scene_files(root: Path) -> Iterable[Path]:
    for scene_dir in sorted(root.glob("TrimmedTestScene*_graph")):
        results = scene_dir / "results_intentions_march-13-18"
        if not results.is_dir():
            continue
        for f in sorted(results.glob("file*.json")):
            yield f


def _file_id(path: Path) -> str:
    # 'file11_1.json' -> '11_1'
    name = path.stem
    return name[len("file"):] if name.startswith("file") else name


def _scene_id_from_path(path: Path) -> str:
    # .../TrimmedTestScene1_graph/...  -> '1'
    for part in path.parts:
        if part.startswith("TrimmedTestScene") and part.endswith("_graph"):
            return part[len("TrimmedTestScene"):-len("_graph")]
    return "?"


def _node_document(class_name: str, properties: List[str], states: List[str]) -> str:
    props = ", ".join(properties) if properties else "no properties"
    sts = ", ".join(states) if states else "no states"
    return f"{class_name}; properties: {props}; states: {sts}"


def _iter_scene_object_docs(
    root: Path,
) -> Iterable[Tuple[str, Dict, str]]:
    for path in _scene_files(root):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skip %s: %s", path, exc)
            continue
        file_id = _file_id(path)
        scene_id = _scene_id_from_path(path)
        nodes = (blob.get("init_graph") or {}).get("nodes") or []
        for n in nodes:
            nid = n.get("id")
            if nid is None:
                continue
            class_name = str(n.get("class_name", ""))
            properties = list(n.get("properties") or [])
            states = list(n.get("states") or [])
            doc_id = f"{file_id}:{nid}"
            doc = _node_document(class_name, properties, states)
            metadata = {
                "file_id": file_id,
                "scene_id": scene_id,
                "node_id": int(nid),
                "class_name": class_name,
                "category": str(n.get("category", "")),
                # store list-valued props/states as joined strings (Chroma
                # only allows scalars in metadata).
                "properties": ",".join(properties),
                "states": ",".join(states),
            }
            yield doc_id, metadata, doc


def _iter_failure_docs(diag_dir: Path) -> Iterable[Tuple[str, Dict, str]]:
    candidate_files = [
        diag_dir / "goal_correct_but_action_fail_top_cases.json",
        diag_dir / "kg_planning_agent_report.json",
    ]
    for path in candidate_files:
        if not path.is_file():
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skip %s: %s", path, exc)
            continue
        records: List[dict] = []
        if isinstance(blob, list):
            records = [r for r in blob if isinstance(r, dict)]
        elif isinstance(blob, dict):
            for key in ("rows", "cases", "items", "records"):
                v = blob.get(key)
                if isinstance(v, list):
                    records = [r for r in v if isinstance(r, dict)]
                    break
        for i, rec in enumerate(records):
            file_id = str(rec.get("file_id") or rec.get("identifier") or "")
            model = str(rec.get("model") or "")
            failure_type = str(rec.get("failure_type") or rec.get("code") or "unknown")
            task = str(rec.get("task") or rec.get("task_name") or "")
            raw_text = str(rec.get("raw_failure_text") or rec.get("detail") or "")
            uid = f"{path.stem}:{i}:{file_id}:{model}"
            doc = f"task={task} failure_type={failure_type} :: {raw_text}"[:2000]
            yield uid, {
                "source_file": path.name,
                "file_id": file_id,
                "model": model,
                "failure_type": failure_type,
                "task": task,
            }, doc


def _load_embedder():
    import os

    from sentence_transformers import SentenceTransformer

    # If model is already cached, work offline to avoid slow HF retries.
    model_cache = MODELS_CACHE / "models--BAAI--bge-small-en-v1.5"
    if model_cache.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    log.info("loading embedding model %s (cache=%s)", EMBEDDING_MODEL, MODELS_CACHE)
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODELS_CACHE))
    return model


def _make_chroma_client():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _ingest(
    client,
    embedder,
    collection_name: str,
    docs: Iterable[Tuple[str, Dict, str]],
    batch_size: int = 256,
) -> int:
    coll = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    total = 0
    ids: List[str] = []
    metas: List[Dict] = []
    texts: List[str] = []
    for uid, meta, text in docs:
        ids.append(uid)
        metas.append(meta)
        texts.append(text)
        if len(ids) >= batch_size:
            embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
            coll.upsert(ids=ids, metadatas=metas, documents=texts, embeddings=embeddings)
            total += len(ids)
            ids, metas, texts = [], [], []
            log.info("[%s] ingested %d", collection_name, total)
    if ids:
        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
        coll.upsert(ids=ids, metadatas=metas, documents=texts, embeddings=embeddings)
        total += len(ids)
    log.info("[%s] done, total=%d", collection_name, total)
    return total


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", default=str(SCENE_GRAPH_ROOT))
    parser.add_argument("--diagnostics", default=str(DIAGNOSTICS_DIR))
    parser.add_argument("--skip-scenes", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    ensure_dirs()
    client = _make_chroma_client()
    embedder = _load_embedder()

    if not args.skip_scenes:
        _ingest(
            client,
            embedder,
            SCENE_OBJECTS_COLLECTION,
            _iter_scene_object_docs(Path(args.scene_root)),
            batch_size=args.batch_size,
        )
    if not args.skip_failures:
        _ingest(
            client,
            embedder,
            FAILURE_CASES_COLLECTION,
            _iter_failure_docs(Path(args.diagnostics)),
            batch_size=args.batch_size,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
