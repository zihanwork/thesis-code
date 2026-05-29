"""Shared paths, model ids, and Neo4j connection parameters.

All values can be overridden via environment variables so the same code
runs locally, in CI, and in the Docker compose recipe.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EAI_ROOT = REPO_ROOT / "embodied-agent-interface-main" / "src" / "virtualhome_eval"
DATASET_ROOT = EAI_ROOT / "dataset" / "programs_processed_precond_nograb_morepreconds"
SCENE_GRAPH_ROOT = DATASET_ROOT / "init_and_final_graphs"
TASK_STATE_PATH = (
    EAI_ROOT / "resources" / "virtualhome" / "task_state_LTL_formula_accurate.json"
)
DIAGNOSTICS_DIR = REPO_ROOT / "output" / "diagnostics"

# ----- persistent artefacts
DATA_DIR = Path(os.environ.get("KB_DATA_DIR", REPO_ROOT / "data" / "kb"))
CHROMA_DIR = Path(os.environ.get("KB_CHROMA_DIR", DATA_DIR / "chroma"))
MODELS_CACHE = Path(os.environ.get("KB_MODELS_DIR", DATA_DIR / "models"))

# ----- chroma
EMBEDDING_MODEL = os.environ.get("KB_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SCENE_OBJECTS_COLLECTION = "scene_objects"
FAILURE_CASES_COLLECTION = "failure_cases"

# ----- neo4j
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "thesis-kb-password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# ----- backend switch (read by analysis.scene_graph_rag / precondition_kg)
KB_BACKEND = os.environ.get("KB_BACKEND", "default").lower()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE.mkdir(parents=True, exist_ok=True)
