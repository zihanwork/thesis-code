#!/usr/bin/env bash
# 一键：generate_prompts（官方）→ gold 合法输出 → evaluate_results
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-eai-eval}"
REPO="${ROOT}/embodied-agent-interface-main"
OUT_HELM="${ROOT}/output/legal_helm_output"

echo "[1/3] generate_prompts (official)"
conda run -n "${CONDA_ENV}" eai-eval \
  --dataset virtualhome \
  --eval-type action_sequencing \
  --mode generate_prompts \
  --output-dir "${ROOT}/output"

echo "[2/3] build gold_oracle *_outputs.json (format-aligned, oracle baseline)"
PYTHONPATH="${REPO}/src" conda run -n "${CONDA_ENV}" python "${ROOT}/analysis/build_action_sequencing_gold_outputs.py" \
  --repo-root "${REPO}" \
  --out-helm-root "${OUT_HELM}" \
  --model-name gold_oracle

echo "[3/3] evaluate_results on gold_oracle only"
LLM_RESPONSE_PATH="${OUT_HELM}/helm_output" NUM_WORKERS=1 \
  "${ROOT}/scripts/run_action_sequencing_eval.sh" virtualhome "${CONDA_ENV}" "${ROOT}/output"

echo "[DONE] See summary: ${ROOT}/output/virtualhome/evaluate_results/action_sequencing/gold_oracle/summary.json"
