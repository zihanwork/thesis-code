#!/usr/bin/env bash
# End-to-end improvement pipeline for the EAI/VirtualHome diagnostic study.
#
# Steps performed:
#   1. Generate EAI prompts (action_sequencing and goal_interpretation)
#   2. For each prompt variant, run analysis/generate_outputs.py
#   3. Optionally run analysis/self_check_loop.py over the baseline outputs
#   4. Normalise action_sequencing outputs into name/id format
#   5. Run EAI evaluate_results
#   6. Update the multi-model materials and figures
#
# Most paths are configurable via environment variables so you can wire
# this script into a real provider (OpenAI, Anthropic, Gemini) or use
# ``--provider dry_run`` for a smoke test that does not hit the network.
set -euo pipefail

PROVIDER="${PROVIDER:-dry_run}"
API_MODEL="${API_MODEL:-gpt-4o-mini}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
BASE_URL="${BASE_URL:-}"
EXTRA_GEN_ARGS=()
if [[ -n "${BASE_URL}" ]]; then
  EXTRA_GEN_ARGS+=(--base-url "${BASE_URL}")
fi
ACTION_VARIANTS=(${ACTION_VARIANTS:-baseline format_constraints few_shot_valid_actions plan_then_ground sg_rag pc_kg_self_check sg_rag_pc_kg})
GOAL_VARIANTS=(${GOAL_VARIANTS:-baseline schema_constrained few_shot decompose_then_merge})
RUN_SELF_CHECK="${RUN_SELF_CHECK:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-10}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
SLEEP="${SLEEP:-0.4}"
CONDA_ENV="${CONDA_ENV:-eai-eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${CONDA_ENV}" == "none" ]]; then
  EAI_CMD=(eai-eval)
else
  EAI_CMD=(conda run -n "${CONDA_ENV}" eai-eval)
fi
DATASET="${DATASET:-virtualhome}"
RUN_EVAL="${RUN_EVAL:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/improvement_run}"
PROMPTS_DIR="${PROMPTS_DIR:-${OUTPUT_ROOT}/prompts}"
RESPONSES_DIR="${RESPONSES_DIR:-${OUTPUT_ROOT}/helm_output}"
NORMALISED_ROOT="${NORMALISED_ROOT:-${OUTPUT_ROOT}/helm_output_norm}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}}"

ACTION_OUT_DIR="${RESPONSES_DIR}/${DATASET}/action_sequencing"
GOAL_OUT_DIR="${RESPONSES_DIR}/${DATASET}/goal_interpretation"
NORMALISED_ACTION_DIR="${NORMALISED_ROOT}/${DATASET}/action_sequencing"

mkdir -p "${PROMPTS_DIR}" "${ACTION_OUT_DIR}" "${GOAL_OUT_DIR}" "${EVAL_OUTPUT_DIR}"

log() { printf "[run_improvement_pipeline] %s\n" "$*"; }

step_generate_prompts() {
  if [[ ! -f "${PROMPTS_DIR}/${DATASET}/generate_prompts/action_sequencing/helm_prompt.json" ]]; then
    log "Generating EAI prompts (action_sequencing)..."
    "${EAI_CMD[@]}" \
      --dataset "${DATASET}" \
      --eval-type action_sequencing \
      --mode generate_prompts \
      --output-dir "${PROMPTS_DIR}"
  else
    log "Action prompts already exist; skipping generation."
  fi

  if [[ ! -f "${PROMPTS_DIR}/${DATASET}/generate_prompts/goal_interpretation/helm_prompt.json" ]]; then
    log "Generating EAI prompts (goal_interpretation)..."
    "${EAI_CMD[@]}" \
      --dataset "${DATASET}" \
      --eval-type goal_interpretation \
      --mode generate_prompts \
      --output-dir "${PROMPTS_DIR}"
  else
    log "Goal prompts already exist; skipping generation."
  fi
}

step_action_variants() {
  local helm_prompt="${PROMPTS_DIR}/${DATASET}/generate_prompts/action_sequencing/helm_prompt.json"
  if [[ ! -f "${helm_prompt}" ]]; then
    log "[WARN] Missing ${helm_prompt}; skipping action variants."
    return
  fi
  for variant in "${ACTION_VARIANTS[@]}"; do
    log "Running action variant '${variant}' for ${MODEL_NAME}..."
    "${PYTHON_BIN}" analysis/generate_outputs.py \
      --provider "${PROVIDER}" \
      --api-model "${API_MODEL}" \
      --model-name "${MODEL_NAME}" \
      --variant "${variant}" \
      --eval-type action_sequencing \
      --helm-prompt "${helm_prompt}" \
      --out-dir "${ACTION_OUT_DIR}" \
      --max-prompts "${MAX_PROMPTS}" \
      --temperature "${TEMPERATURE}" \
      --max-tokens "${MAX_TOKENS}" \
      --sleep "${SLEEP}" \
      "${EXTRA_GEN_ARGS[@]}"
  done
}

step_goal_variants() {
  local helm_prompt="${PROMPTS_DIR}/${DATASET}/generate_prompts/goal_interpretation/helm_prompt.json"
  if [[ ! -f "${helm_prompt}" ]]; then
    log "[WARN] Missing ${helm_prompt}; skipping goal variants."
    return
  fi
  for variant in "${GOAL_VARIANTS[@]}"; do
    log "Running goal variant '${variant}' for ${MODEL_NAME}..."
    "${PYTHON_BIN}" analysis/improve_goal_interpretation.py \
      --provider "${PROVIDER}" \
      --api-model "${API_MODEL}" \
      --model-name "${MODEL_NAME}" \
      --variant "${variant}" \
      --helm-prompt "${helm_prompt}" \
      --out-dir "${GOAL_OUT_DIR}" \
      --max-prompts "${MAX_PROMPTS}" \
      --temperature "${TEMPERATURE}" \
      --max-tokens "${MAX_TOKENS}" \
      --sleep "${SLEEP}" \
      --validation-report "${GOAL_OUT_DIR}/${MODEL_NAME}_${variant}_validation.json" \
      "${EXTRA_GEN_ARGS[@]}"
  done
}

step_self_check() {
  if [[ "${RUN_SELF_CHECK}" != "1" ]]; then
    return
  fi
  local baseline="${ACTION_OUT_DIR}/${MODEL_NAME}_outputs.json"
  local error_info="${EVAL_OUTPUT_DIR}/${DATASET}/evaluate_results/action_sequencing/${MODEL_NAME}/error_info.json"
  local helm_prompt="${PROMPTS_DIR}/${DATASET}/generate_prompts/action_sequencing/helm_prompt.json"
  if [[ ! -f "${baseline}" ]]; then
    log "[WARN] Baseline ${baseline} not found; skipping self-check loop."
    return
  fi
  if [[ ! -f "${error_info}" ]]; then
    log "[WARN] error_info.json not found at ${error_info}; run evaluation first to enable self-check."
    return
  fi
  if [[ ! -f "${helm_prompt}" ]]; then
    log "[WARN] Missing helm_prompt for self-check loop; skipping."
    return
  fi
  log "Running self-check rewrite loop for ${MODEL_NAME}..."
  "${PYTHON_BIN}" analysis/self_check_loop.py \
    --provider "${PROVIDER}" \
    --api-model "${API_MODEL}" \
    --model-name "${MODEL_NAME}" \
    --baseline-outputs "${baseline}" \
    --error-info "${error_info}" \
    --helm-prompt "${helm_prompt}" \
    --out-dir "${ACTION_OUT_DIR}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --sleep "${SLEEP}" \
    --report "${ACTION_OUT_DIR}/${MODEL_NAME}_self_check_report.json" \
    "${EXTRA_GEN_ARGS[@]}"
}

step_pc_kg_self_check() {
  # Knowledge-grounded recovery: verify baseline drafts against the
  # precondition KG (no evaluator run required) and rewrite only failing
  # rows. Produces outputs for the ``pc_kg_self_check`` variant.
  if [[ "${RUN_PC_KG_SELF_CHECK:-1}" != "1" ]]; then
    return
  fi
  local baseline="${ACTION_OUT_DIR}/${MODEL_NAME}_outputs.json"
  local helm_prompt="${PROMPTS_DIR}/${DATASET}/generate_prompts/action_sequencing/helm_prompt.json"
  if [[ ! -f "${baseline}" ]]; then
    log "[WARN] Baseline ${baseline} not found; skipping PC-KG self-check."
    return
  fi
  if [[ ! -f "${helm_prompt}" ]]; then
    log "[WARN] Missing helm_prompt; skipping PC-KG self-check."
    return
  fi
  log "Running PC-KG self-check loop for ${MODEL_NAME}..."
  "${PYTHON_BIN}" analysis/self_check_loop.py \
    --provider "${PROVIDER}" \
    --api-model "${API_MODEL}" \
    --model-name "${MODEL_NAME}" \
    --baseline-outputs "${baseline}" \
    --helm-prompt "${helm_prompt}" \
    --out-dir "${ACTION_OUT_DIR}" \
    --variant-label pc_kg_triage \
    --verifier pc_kg \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --sleep "${SLEEP}" \
    --report "${ACTION_OUT_DIR}/${MODEL_NAME}_pc_kg_triage_report.json" \
    "${EXTRA_GEN_ARGS[@]}"
}

step_normalise() {
  log "Normalising action_sequencing outputs into name/id format..."
  mkdir -p "${NORMALISED_ACTION_DIR}"
  "${PYTHON_BIN}" analysis/normalize_action_outputs.py \
    --input-dir "${ACTION_OUT_DIR}" \
    --output-dir "${NORMALISED_ACTION_DIR}"
}

step_evaluate() {
  if [[ "${RUN_EVAL}" != "1" ]]; then
    log "RUN_EVAL=0; skipping EAI evaluate_results."
    return
  fi
  log "Running EAI evaluate_results for action_sequencing..."
  LLM_RESPONSE_PATH="${NORMALISED_ROOT}" NUM_WORKERS="${NUM_WORKERS:-1}" \
    ./scripts/run_action_sequencing_eval.sh "${DATASET}" "${CONDA_ENV}" "${EVAL_OUTPUT_DIR}"

  if [[ -d "${RESPONSES_DIR}/${DATASET}/goal_interpretation" ]]; then
    log "Running EAI evaluate_results for goal_interpretation..."
    "${EAI_CMD[@]}" \
      --dataset "${DATASET}" \
      --eval-type goal_interpretation \
      --mode evaluate_results \
      --output-dir "${EVAL_OUTPUT_DIR}" \
      --llm-response-path "${RESPONSES_DIR}" \
      --num-workers "${NUM_WORKERS:-1}"
  fi
}

step_summary() {
  log "Refreshing materials and figures..."
  "${PYTHON_BIN}" analysis/prepare_multimodel_experiment_materials.py
}

step_generate_prompts
step_action_variants
step_goal_variants
step_evaluate
step_self_check
step_pc_kg_self_check
step_normalise
step_evaluate
step_summary

log "Done. Inspect ${EVAL_OUTPUT_DIR} and output/diagnostics for results."
