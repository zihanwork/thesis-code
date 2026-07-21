# Final Experiment Protocol

## Confirmatory source runs

The final protocol is a complete factorial planner-harness-model matrix. Every
cell is executed; there are no intentional empty cells or model-specific gaps.
All cells use the same 84-task official VirtualHome Action Sequencing cohort.

| Dimension | Conditions | Count |
|---|---|---:|
| Planner | B0 Minimal, P0-S Structured, P0-E Engineered, P1 Flat RAG, P2 GraphRAG | 5 |
| Harness | H0 Open Loop, H2-R Reflection, H2-M Memory, H2-P PDDL Recovery | 4 |
| Model | DeepSeek-V4-Flash, gpt-5.5, GLM-5-Turbo | 3 |
| Official tasks | Frozen compatible VirtualHome cohort | 84 |
| **Total cells** | 5 x 4 x 3 | **60** |
| **Total records** | 60 x 84 | **5040** |

The machine-readable source-run configuration is
`configs/experiments/final_full_matrix_v2.json`. The previous staged final
configs were removed so that there is only one formal experiment definition.

## Planner-Harness Matrix

Every entry in this matrix is run with all three models and all 84 tasks.

| Planner / initial planning method | H0 open loop | H2-R validator-feedback recovery | H2-M memory-augmented recovery | H2-P PDDL recovery |
|---|---|---|---|---|
| **B0 Minimal Prompt** | 3 models | 3 models | 3 models | 3 models |
| **P0-S Structured Prompt** | 3 models | 3 models | 3 models | 3 models |
| **P0-E Engineered Prompt** | 3 models | 3 models | 3 models | 3 models |
| **P1 Flat RAG** | 3 models | 3 models | 3 models | 3 models |
| **P2 GraphRAG** | 3 models | 3 models | 3 models | 3 models |
| **Total** | **15** | **15** | **15** | **15** |

This gives 20 planner-harness combinations and 60 model-specific cells. H2-P
is deliberately included for every planner so that recovery comparisons do not
silently change the initial planning method. P2 is the new GraphRAG treatment:
it retrieves training-only graph subgraphs and graph-derived action chains. It
does not reuse the deleted symbolic P2 artifacts.

Every source run preserves raw plans, prompts, model-call telemetry, retrieval
provenance, local verifier traces, and repair traces. Local PDDL execution is
implementation telemetry only; it is never reported as the thesis outcome.

## Official evaluation population

Outcome reporting uses the frozen cohort at
`data/processed/tasksets/official_virtualhome_action_sequencing_v1.jsonl`.
Every one of the 60 model-specific cells is exported with exactly these 84 task
identifiers and scored by the pinned official VirtualHome Action Sequencing
evaluator. Failed or malformed predictions remain in the denominator.

The primary endpoint is official task success. Secondary endpoints are official
total-goal completion, state/relation/action goal completion, execution success,
and evaluator error categories.

## Method attribution

- B0, P0-S, P0-E, P1, and P2 vary planning-time information.
- H2-R, H2-M, and H2-P vary execution-time recovery while holding the initial
  planner condition fixed within each row.
- H2-P is a symbolic recovery mechanism, not the P2 planning method.
- P2-GraphRAG is a graph retrieval treatment and is evaluated independently of
  PDDL search.
- The pinned official evaluator is the sole outcome authority.

## Statistical policy

Comparisons are paired by task and stratified by model, planner, and harness.
Exact two-sided McNemar tests quantify discordant binary outcomes. Confidence
intervals resample all observations in a task family together, using 10,000
bootstrap samples and seed 13.

## Reproducibility

The machine-readable protocol is `configs/experiments/final_protocol_v1.json`.
Official export and evaluation commands are in
`docs/official_eai_protocol.md`. All 60 cells and the official evaluator run are
complete; frozen evidence is in `docs/final_official_virtualhome_results_v4.*`.
