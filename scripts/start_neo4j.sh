#!/usr/bin/env bash
# Start a local Neo4j 5 container for the persistent KG.
#
# Usage:
#   NEO4J_PASSWORD=mypw bash scripts/start_neo4j.sh
#
# Bolt:    bolt://localhost:7687
# Browser: http://localhost:7474
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-thesis-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-thesis-kb-password}"
NEO4J_VERSION="${NEO4J_VERSION:-5.20}"
DATA_DIR="${KB_DATA_DIR:-$(pwd)/data/kb}/neo4j"
mkdir -p "${DATA_DIR}/data" "${DATA_DIR}/logs" "${DATA_DIR}/import" "${DATA_DIR}/plugins"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[start_neo4j] container ${CONTAINER_NAME} exists; (re)starting"
  docker start "${CONTAINER_NAME}" >/dev/null
  exit 0
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 7474:7474 -p 7687:7687 \
  -v "${DATA_DIR}/data":/data \
  -v "${DATA_DIR}/logs":/logs \
  -v "${DATA_DIR}/import":/var/lib/neo4j/import \
  -v "${DATA_DIR}/plugins":/plugins \
  -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" \
  -e NEO4J_PLUGINS='["apoc"]' \
  "neo4j:${NEO4J_VERSION}"

echo "[start_neo4j] waiting for bolt..."
for i in $(seq 1 60); do
  if docker exec "${CONTAINER_NAME}" cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" 'RETURN 1' >/dev/null 2>&1; then
    echo "[start_neo4j] ready"
    exit 0
  fi
  sleep 2
done
echo "[start_neo4j] timed out waiting for Neo4j" >&2
exit 1
