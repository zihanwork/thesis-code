#!/usr/bin/env bash
# 步骤 2（前半）：生成 VirtualHome + action_sequencing 的 helm_prompt.json
# 之后用你的模型 API 按 identifier 填 llm_output，合并为 <model>_outputs.json
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-eai-eval}"
DATASET="${1:-virtualhome}"
OUT="${2:-output/helm_prompts_action_sequencing}"

mkdir -p "${OUT}"

echo "[INFO] Writing prompts to ${OUT}/helm_prompt.json"
conda run -n "${CONDA_ENV}" eai-eval \
  --dataset "${DATASET}" \
  --eval-type action_sequencing \
  --mode generate_prompts \
  --output-dir "${OUT}"

echo "[DONE] Prompt file: ${OUT}/helm_prompt.json"
