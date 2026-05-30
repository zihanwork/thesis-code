#!/usr/bin/env bash
# Run full 200-prompt E1 experiment with persistent KB (sg_rag variant)
# and compare against in-memory sg_rag baseline.
#
# Usage:
#   export PROVIDER=openai_compatible
#   export API_MODEL=deepseek-v4-flash   # or your provider's model id
#   export BASE_URL=https://api.deepseek.com/v1
#   export API_KEY_ENV=DEEPSEEK_API_KEY   # name of env var holding the key
#   export KB_BACKEND=persistent
#   bash scripts/run_e1_persistent_200.sh
set -euo pipefail

PROVIDER="${PROVIDER:-openai_compatible}"
API_MODEL="${API_MODEL:-deepseek-chat}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-https://api.deepseek.com/v1}"
API_KEY_ENV="${API_KEY_ENV:-DEEPSEEK_API_KEY}"
MAX_PROMPTS="${MAX_PROMPTS:-200}"
SLEEP="${SLEEP:-0.4}"
CONDA_ENV="${CONDA_ENV:-eai-eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

HELM_PROMPT="output/improvement_run/prompts/virtualhome/generate_prompts/action_sequencing/helm_prompt.json"
OUT_DIR="output/e1_persistent_200/helm_output/virtualhome/action_sequencing"
NORM_DIR="output/e1_persistent_200/helm_output_norm/virtualhome/action_sequencing"
EVAL_OUT="output/e1_persistent_200"

export KB_BACKEND=persistent

log() { printf "[e1_persistent_200] %s\n" "$*"; }

# ---- Step 1: Generate 200 prompts with persistent sg_rag KB
log "Generating ${MAX_PROMPTS} prompts with persistent KB (sg_rag)..."
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" analysis/generate_outputs.py \
    --provider "${PROVIDER}" \
    --api-model "${API_MODEL}" \
    --model-name "${MODEL_NAME}" \
    --variant sg_rag \
    --eval-type action_sequencing \
    --helm-prompt "${HELM_PROMPT}" \
    --out-dir "${OUT_DIR}" \
    --max-prompts "${MAX_PROMPTS}" \
    --temperature 0 \
    --max-tokens 2048 \
    --sleep "${SLEEP}" \
    --base-url "${BASE_URL}" \
    --api-key-env "${API_KEY_ENV}" \
    --resume

log "Generation done. $(python3 -c "import json; d=json.load(open('${OUT_DIR}/${MODEL_NAME}_sg_rag_outputs.json')); print(len(d), 'entries')")"

# ---- Step 2: Normalize outputs into name/id format
log "Normalising outputs..."
mkdir -p "${NORM_DIR}"
"${PYTHON_BIN}" analysis/normalize_action_outputs.py \
    --input-dir "${OUT_DIR}" \
    --output-dir "${NORM_DIR}"

# ---- Step 3: Run EAI evaluate_results
log "Running EAI evaluate_results..."
if [[ "${CONDA_ENV}" == "none" ]]; then
    eai-eval \
        --dataset virtualhome \
        --eval-type action_sequencing \
        --mode evaluate_results \
        --output-dir "${EVAL_OUT}" \
        --llm-response-path "output/e1_persistent_200/helm_output_norm" \
        --num-workers 1
else
    conda run -n "${CONDA_ENV}" eai-eval \
        --dataset virtualhome \
        --eval-type action_sequencing \
        --mode evaluate_results \
        --output-dir "${EVAL_OUT}" \
        --llm-response-path "output/e1_persistent_200/helm_output_norm" \
        --num-workers 1
fi

# ---- Step 4: Print comparison
log "============================================"
log "COMPARISON: in-memory sg_rag vs persistent"
log "============================================"
"${PYTHON_BIN}" - <<'PY'
import json, pathlib

baseline_path = pathlib.Path(
    "output/improvement_run/virtualhome/evaluate_results"
    "/action_sequencing/deepseek-v4-flash_sg_rag/summary.json"
)
persistent_path = pathlib.Path(
    "output/e1_persistent_200/virtualhome/evaluate_results"
    f"/action_sequencing/deepseek-v4-flash_sg_rag/summary.json"
)

def load(p):
    if p.exists():
        return json.loads(p.read_text())
    return None

b = load(baseline_path)
p = load(persistent_path)

if b and p:
    keys = ["task_success_rate", "state_goal", "relation_goal", "action_goal"]
    print(f"{'Metric':<30} {'In-memory':>12} {'Persistent':>12} {'Delta':>10}")
    print("-" * 66)
    for k in keys:
        bv = b["goal_evaluation"].get(k, 0)
        pv = p["goal_evaluation"].get(k, 0)
        delta = pv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {k:<28} {bv:>11.2f}% {pv:>11.2f}% {sign}{delta:>8.2f}%")
    print()
    bex = b["trajectory_evaluation"]["execution_success_rate"]
    pex = p["trajectory_evaluation"]["execution_success_rate"]
    delta = pex - bex
    sign = "+" if delta >= 0 else ""
    print(f"  {'execution_success_rate':<28} {bex:>11.2f}% {pex:>11.2f}% {sign}{delta:>8.2f}%")
else:
    if not b:
        print("ERROR: baseline summary not found at", baseline_path)
    if not p:
        print("ERROR: persistent summary not found at", persistent_path)
PY

log "Done. Full results in ${EVAL_OUT}/virtualhome/evaluate_results/action_sequencing/"
