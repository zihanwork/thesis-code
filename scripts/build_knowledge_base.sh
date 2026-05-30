#!/usr/bin/env bash
# One-shot bootstrap for the persistent knowledge base.
#
#   1. starts Neo4j (Docker)
#   2. builds the Chroma vector store (BGE embeddings)
#   3. populates the Neo4j KG (scenes + rules + failures)
#
# Re-runs are idempotent: Chroma upserts by id, Neo4j MERGEs by key.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export KB_BACKEND="${KB_BACKEND:-persistent}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-thesis-kb-password}"

echo "[kb] starting Neo4j..."
bash scripts/start_neo4j.sh

echo "[kb] building Chroma vector store..."
python3 -m analysis.kb.build_vector_store "$@"

echo "[kb] populating Neo4j graph database..."
python3 -m analysis.kb.build_graph_db "$@"

echo "[kb] done. set KB_BACKEND=persistent to use it."
