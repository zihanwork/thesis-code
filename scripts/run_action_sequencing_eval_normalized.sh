#!/usr/bin/env bash
set -euo pipefail

# Normalize official HELM outputs into name_id format, then evaluate.
# Usage:
#   ./scripts/run_action_sequencing_eval_normalized.sh [conda_env] [output_dir]

CONDA_ENV="${1:-eai-eval}"
OUTPUT_DIR="${2:-output}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${ROOT}/embodied-agent-interface-main"

DEFAULT_HELM_PATH="$(conda run -n "${CONDA_ENV}" python -c "import eai_eval; print(eai_eval.helm_output_path)")"
INPUT_DIR="${DEFAULT_HELM_PATH}/helm_output/virtualhome/action_sequencing"
NORMALIZED_ROOT="${ROOT}/output/normalized_helm_output"
NORMALIZED_DIR="${NORMALIZED_ROOT}/helm_output/virtualhome/action_sequencing"

echo "[1/3] Normalize action outputs"
conda run -n "${CONDA_ENV}" python "${ROOT}/analysis/normalize_action_outputs.py" \
  --repo-root "${REPO}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${NORMALIZED_DIR}"

echo "[2/3] Evaluate normalized outputs"
LLM_RESPONSE_PATH="${NORMALIZED_ROOT}/helm_output" NUM_WORKERS=1 \
  "${ROOT}/scripts/run_action_sequencing_eval.sh" virtualhome "${CONDA_ENV}" "${OUTPUT_DIR}"

echo "[3/3] Done. Compare summary with baseline parsing-heavy run"
echo "    - normalized source: ${NORMALIZED_DIR}"
echo "    - evaluate results:  ${OUTPUT_DIR}/virtualhome/evaluate_results/action_sequencing"
