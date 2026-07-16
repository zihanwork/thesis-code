# Revised Experiment Design

## Thesis Claim

The study separates planning-time augmentation from execution-time recovery.
It asks how structured prompting and retrieval affect the initial plan, how
isolated recovery mechanisms repair failures, and whether the effects transfer
across model families at acceptable cost.

The historical 3x3 matrix is a pilot. It is not the final confirmatory design
because the old P2 and H2 both invoked the same symbolic PDDL search.

## Research Questions

- RQ1: What is gained by structured prompts, engineered prompts, and RAG?
- RQ2: Which failures are repaired by local, LLM-feedback, and PDDL recovery?
- RQ3: Are planning-time RAG and execution-time recovery complementary?
- RQ4: Do the effects transfer across model families and capability levels?
- RQ5: What are the token, latency, monetary, and symbolic-search costs?

## Planning Ablation

Run every method under `H0_open_loop` so the initial plan is measured without
recovery contamination.

| ID | Paper name | Role |
|---|---|---|
| `B0_minimal_prompt` | Minimal Prompt Baseline | Instruction and allowed actions only |
| `P0_structured_prompt` | Structured PDDL-Informed Prompt | State, goal, objects, and PDDL signatures |
| `P0_engineered_prompt` | Engineered Structured Prompt | P0 plus a fixed constraint checklist |
| `P1_rag` | Retrieval-Augmented Planning | P0-PE plus task-conditioned demonstrations |
| `P2_symbolic_pddl` | Symbolic PDDL Reference | Model-independent reference, run once |

P2 is not a knowledge-graph method and is not claimed to be an upper bound.

## Recovery Ablation

| ID | Paper name | Allowed repair mechanism |
|---|---|---|
| `H0_open_loop` | Open Loop | None |
| `H1_verifier_gated` | Safety/Validity Gate | Detect and block only |
| `H2_local_recovery` | Local Recovery | Safety rule and local deterministic patch |
| `H2_llm_reflection` | LLM Feedback Replanning | Explicit validator feedback to the original model |
| `H2_error_specific` | Error-Specific LLM Repair | Validator feedback plus error-type guidance |
| `H2_memory` | Memory-Augmented LLM Repair | Validator feedback plus frozen failure-repair example |
| `H2_combined` | Combined LLM Harness | Local patch, error-specific guidance, and frozen memory |
| `H2_pddl_recovery` | Symbolic PDDL Recovery | Symbolic search fallback |

`H2_full_recovery` remains available only to reproduce historical pilot runs.
It must not be used as a final thesis method because it mixes mechanisms.

The combined harness excludes PDDL fallback so its contribution is not confused
with the symbolic reference. `H2_combined_no_local`,
`H2_combined_no_error`, and `H2_combined_no_memory` provide
leave-one-component-out ablations. Cross-task memory is built only from train/development data and
is read-only during final evaluation.

The first frozen development memory contains 357 successful failure-to-repair
pairs. All were produced by the historical symbolic PDDL recovery, so this
version must be described as **symbolic-teacher failure memory**, not ordinary
LLM self-memory. Later LLM-only development runs may support a separate
LLM-derived memory corpus and teacher-source ablation.

## Experiment Phases

1. Model compatibility smoke: validate API routing, parseability, and telemetry.
2. Planning development ablation: run B0/P0/P0-PE/P1 under H0 on the 202-task
   development set.
3. Recovery development ablation: run P0-PE and P1 under isolated recovery modes.
4. Symbolic reference: run P2 once without an LLM or recovery fallback.
5. RAG and recovery tuning: development data only.
6. Safety evaluation: use a separate frozen safety dataset for H1 claims.
7. Generalization: rerun only preregistered key cells on additional model families.
8. Final evaluation: frozen local held-out and official challenge submission.

The 202-task `balanced_eval.jsonl` set is development-only. Its results must not
be described as held-out or official benchmark performance.

The local held-out split is frozen at 119 VirtualHome-only tasks with task-ID
hash `036ed8d9c943477bdc704d4d1e4fd3e84541352f8a132984b68c3b6c51f22eac`.
See `docs/data_split_protocol.md`. It must not be executed until final methods
and reporting rules are frozen.

## One API Model Policy

The account model directory and smoke results are recorded in
`configs/models/one_api_catalog.json`. As of 2026-07-16:

- Verified existing models: `DeepSeek-V4-Flash`, `gpt-5.5`.
- Newly verified: `DeepSeek-V4-Pro`, `GLM-5-Turbo`.
- Conditional: `MiniMax-M3`, `Kimi-K2.6`.
- Listed but unavailable through the current chat route: tested Claude models
  and `grok-4.5`.

The primary generalization addition is `GLM-5-Turbo` because it adds a new
model family. `DeepSeek-V4-Pro` is a useful within-family capability comparison.
Qwen and Llama are not exposed by this One API account. A LoRA/PEFT experiment
therefore requires a separate local open-weight backend. Fine-tuning is an
optional appendix/future-work experiment and does not block the main thesis.

## Reporting Guardrails

- Keep planner quality and recovery quality as separate estimands.
- Reuse the same initial plan across harness modes for paired comparison.
- Cap repair calls equally and report conditional recovery success.
- Report VirtualHome and BEHAVIOR separately.
- Record exact model IDs and per-model decoding overrides.
- Report 95% confidence intervals, paired McNemar tests, calls, tokens, latency,
  cost, repair attempts, PDDL time, and explored states.
- Never write final-test failures into RAG or failure memory.

## Runnable Development Configurations

- `configs/experiments/eai_planning_ablation_dev.json`
- `configs/experiments/eai_recovery_ablation_dev.json`
- `configs/experiments/eai_harness_leave_one_out_dev.json`
- `configs/experiments/eai_symbolic_reference_dev.json`
- `configs/experiments/eai_model_generalization_smoke.json`

These configurations define the structure but should not be launched as paid
full runs until model pricing, run budgets, and the development/final protocol
are frozen.

Build or refresh a frozen development memory with:

```bash
uv run embodied-gap build-failure-memory \
  --tasks data/processed/tasksets/balanced_eval.jsonl \
  --runs <development-runs.jsonl> \
  --out data/knowledge/failure_memory_dev.jsonl
```
