#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_action_sequencing_eval.sh [dataset] [conda_env] [output_dir]
# Example:
#   ./scripts/run_action_sequencing_eval.sh virtualhome eai-eval output

DATASET="${1:-virtualhome}"
CONDA_ENV="${2:-eai-eval}"
OUTPUT_DIR="${3:-output}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LLM_RESPONSE_PATH="${LLM_RESPONSE_PATH:-}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/action_sequencing_eval_${TIMESTAMP}.log"
EXPECTED_DIR="${OUTPUT_DIR}/${DATASET}/evaluate_results/action_sequencing"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Log file: ${LOG_FILE}"

if [[ "${CONDA_ENV}" == "none" ]]; then
  CMD=(eai-eval)
else
  CMD=(conda run -n "${CONDA_ENV}" eai-eval)
fi
CMD+=(
  --dataset "${DATASET}"
  --eval-type action_sequencing
  --mode evaluate_results
  --num-workers "${NUM_WORKERS}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${LLM_RESPONSE_PATH}" ]]; then
  CMD+=(--llm-response-path "${LLM_RESPONSE_PATH}")
  echo "[INFO] Using llm-response-path: ${LLM_RESPONSE_PATH}"
fi

echo "[INFO] Running action_sequencing evaluation..."
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "[INFO] Checking output directory: ${EXPECTED_DIR}"
if [[ ! -d "${EXPECTED_DIR}" ]]; then
  echo "[ERROR] Expected output directory not found: ${EXPECTED_DIR}"
  echo "[HINT] Try setting LLM_RESPONSE_PATH if default benchmark outputs are unavailable."
  exit 1
fi

echo "[INFO] Generated summary files:"
shopt -s nullglob
summary_files=("${EXPECTED_DIR}"/*/summary.json)
if [[ ${#summary_files[@]} -eq 0 ]]; then
  echo "[WARN] No model summary.json found under ${EXPECTED_DIR}"
else
  for summary in "${summary_files[@]}"; do
    echo " - ${summary}"
  done
fi
shopt -u nullglob

echo "[DONE] action_sequencing evaluation completed."
