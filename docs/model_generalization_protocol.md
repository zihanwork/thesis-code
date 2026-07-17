# Model Generalization Protocol and Development Evidence

## Purpose

This stage tests whether the planning-time RAG effect transfers beyond the two
primary models. It does not expand the full planner-by-harness grid to every
model. Recovery is deliberately excluded so that the estimand remains the
effect of P1 over P0 rather than a mixture of planning and repair.

All results in this document are development evidence. They are neither frozen
held-out results nor official EAI benchmark scores.

## Model selection gates

The One API inventory was checked on 2026-07-17. A listed model was not included
solely because its name appeared in `/v1/models`.

1. The minimal chat-completions request must succeed and return parseable JSON.
2. A realistic EAI planning canary must complete with token and latency
   telemetry.
3. The 2048-token output-budget canary must not end by length.
4. The model must add useful coverage rather than only increase model count.

`GLM-5-Turbo` passed these gates and adds a third model family.
`DeepSeek-V4-Pro` passed the minimal API check but was not promoted because one
of its two 2048-token EAI canary calls ended by length. `MiniMax-M3` and
`Kimi-K2.6` remain conditional. Qwen and Llama are not exposed by this account,
and the tested Claude routes are unavailable through the current compatible
chat endpoint.

Fine-tuning remains an optional appendix or future-work experiment. It is not a
prerequisite for the main thesis experiment.

## Frozen development pilot

Configuration:

- Config: `configs/experiments/eai_model_generalization_dev20.json`
- Tasks: `data/processed/tasksets/balanced_eval_20.jsonl`
- Task SHA256: `0fba7d3e9d3d1220167155b8d844cc9c6440ef186d57d4ac53c0e1d23d3fb2ec`
- Evaluation task-ID SHA256:
  `a9cab142834e789a2be2e5e78f6fb9860a9608cc98d108f448caae71c988d2dc`
- RAG corpus: `data/processed/tasksets/rag_train.jsonl`, 75 tasks
- Cells: P0 engineered prompt with H0; P1 RAG with H0
- Models: `DeepSeek-V4-Flash`, `gpt-5.5`, `GLM-5-Turbo`
- Temperature: 0
- Maximum output: 2048 tokens
- Repair calls: none
- Total planned and completed calls: 120

Run ID: `20260717T040505406556Z_2cf0d6b2`.

The run manifest records commit `1677419f3cf2eaf37a2f374b1168db742cd2da92`
and `dirty_worktree=true` because the preregistered Stage 10 configuration had
not yet been committed when the development run started. The exact config was
snapshotted in the run directory and its hash is
`2562084632fc09718f2f68bd6465bf3bd37de9098593a0a5816c4030a97cf003`.
This is acceptable only for a development pilot. The final run gate requires a
clean committed worktree.

## Results

| Model | P0 success (95% CI) | P1 success (95% CI) | Paired uplift (95% CI) | McNemar p | Tokens | Latency | Length-truncated calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 5/20, 25.0% (11.2–46.9) | 14/20, 70.0% (48.1–85.5) | +45 pp (+20–+70) | 0.0117 | 129,059 | 381.6 s | 8/40 |
| gpt-5.5 | 8/20, 40.0% (21.9–61.3) | 13/20, 65.0% (43.3–81.9) | +25 pp (+5–+50) | 0.1250 | 105,754 | 447.2 s | 0/40 |
| GLM-5-Turbo | 3/20, 15.0% (5.2–36.0) | 13/20, 65.0% (43.3–81.9) | +50 pp (+30–+70) | 0.0020 | 119,920 | 524.6 s | 3/40 |

The direction of the RAG effect is positive for all three models. On this small
paired pilot, it is statistically significant for DeepSeek-V4-Flash and
GLM-5-Turbo but not for gpt-5.5. This is evidence of transfer to a third model
family, not proof that the effect is universal.

The dataset strata remain important:

| Model | BEHAVIOR P0 → P1 | VirtualHome P0 → P1 |
|---|---:|---:|
| DeepSeek-V4-Flash | 50.0% → 50.0% | 8.3% → 83.3% |
| gpt-5.5 | 37.5% → 37.5% | 41.7% → 83.3% |
| GLM-5-Turbo | 12.5% → 50.0% | 16.7% → 75.0% |

The overall gain is still driven mainly by VirtualHome for the two primary
models. The final thesis must retain per-dataset reporting and must not describe
the aggregate uplift as equally strong on BEHAVIOR.

## Reporting and final-run decisions

- Keep DeepSeek-V4-Flash and gpt-5.5 as the two full-grid models.
- Include GLM-5-Turbo only in the preregistered P0/H0 and P1/H0
  generalization cells; do not multiply the full recovery grid by this model.
- Do not include DeepSeek-V4-Pro in the final matrix unless a new protocol is
  preregistered before any final-test access.
- Keep the same 2048-token resource cap across the three reported models and
  report `finish_reason=length` counts. This is a resource-matched comparison,
  not an unlimited-generation comparison.
- Monetary cost is unavailable because One API pricing was not configured.
  Tokens, calls, and latency are available and must still be reported.
- The frozen held-out set remains unopened until the final clean-commit gate.

The machine-readable evidence is in
`docs/model_generalization_evidence.json`. It can be regenerated with:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli summarize-model-generalization \
  --run-dir runs/eai_model_generalization_dev20/20260717T040505406556Z_2cf0d6b2 \
  --out docs/model_generalization_evidence.json
```
