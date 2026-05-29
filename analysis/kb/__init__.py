"""Persistent knowledge base (Chroma RAG + Neo4j KG) for VirtualHome.

This package provides drop-in replacements for ``analysis.scene_graph_rag``
and ``analysis.precondition_kg`` backed by a real vector store and graph
database. The originals are kept as the default backend; opt in by setting
``KB_BACKEND=persistent``.
"""
from .config import (
    CHROMA_DIR,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EMBEDDING_MODEL,
    SCENE_OBJECTS_COLLECTION,
    FAILURE_CASES_COLLECTION,
)

__all__ = [
    "CHROMA_DIR",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "EMBEDDING_MODEL",
    "SCENE_OBJECTS_COLLECTION",
    "FAILURE_CASES_COLLECTION",
]
