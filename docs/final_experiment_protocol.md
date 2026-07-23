# Final Experiment Protocol

## Evidence structure

The primary factorial study covers B0, P0-S, P0-E, and P1 across all harnesses
and models. The GraphRAG follow-up directly compares P1 and P2 across the same
harnesses, models, and 84-task cohort. Both use the pinned official evaluator;
results are reported as primary and matched follow-up evidence respectively.

| Dimension | Conditions | Count |
|---|---|---:|
| Planner | B0 Minimal, P0-S Structured, P0-E Engineered, P1 Flat RAG, P2 GraphRAG | 5 |
| Harness | H0 Open Loop, H2-R Reflection, H2-M Memory, H2-P PDDL Recovery | 4 |
| Model | DeepSeek-V4-Flash, gpt-5.5, GLM-5-Turbo | 3 |
| Official tasks | Frozen compatible VirtualHome cohort | 84 |
| **Total cells** | 5 x 4 x 3 | **60** |
| **Total records** | 60 x 84 | **5040** |

The machine-readable primary configuration is
`configs/experiments/final_full_matrix_v2.json`. The P1/P2 follow-up is defined
by `configs/experiments/graph_rag_replacement_replication.json` and its recorded
transient-error amendments.

## Planner-Harness Matrix

Every claim-bearing cell contains all three models and the same 84 tasks.

| Planner / initial planning method | H0 open loop | H2-R validator-feedback recovery | H2-M memory-augmented recovery | H2-P PDDL recovery |
|---|---|---|---|---|
| **B0 Minimal Prompt** | 3 models | 3 models | 3 models | 3 models |
| **P0-S Structured Prompt** | 3 models | 3 models | 3 models | 3 models |
| **P0-E Engineered Prompt** | 3 models | 3 models | 3 models | 3 models |
| **P1 Flat RAG** | 3 models | 3 models | 3 models | 3 models |
| **P2 GraphRAG** | 3 models | 3 models | 3 models | 3 models |
| **Total** | **15** | **15** | **15** | **15** |

The combined evidence covers 20 planner-harness combinations and 60
model-specific cells. H2-P is included for every planner so recovery comparisons
hold the initial planning method fixed. P2 retrieves training-only graph
subgraphs and graph-derived action chains as an optimisation of P1.

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
- H2-P is a composite symbolic recovery mechanism (macro plan, grounded PDDL
  search, then action-model fallback), not the P2 planning method.
- P2-GraphRAG is a graph retrieval treatment and is evaluated independently of
  PDDL search.
- The pinned official evaluator is the sole outcome authority.

## Statistical policy

Comparisons are paired by task and stratified by model, planner, and harness.
Exact two-sided McNemar tests quantify discordant binary outcomes. Confidence
intervals resample all observations in a task family together, using 10,000
bootstrap samples and seed 13.

## Reproducibility and deviations

The machine-readable protocol is `configs/experiments/final_protocol_v1.json`.
Official export and evaluation commands are in `docs/official_eai_protocol.md`.
Frozen evidence is in `docs/final_official_virtualhome_results_v4.*` and
`docs/final_official_virtualhome_graph_rag_replacement.*`.

The run manifests record a dirty worktree, so the commit alone is not a complete
reconstruction key; artifact hashes, frozen configs, raw outputs, and official
exports define the evidence of record. The GraphRAG follow-up also used recorded
operational amendments for missing model outputs after HTTP 429/503 failures:
model generation settings were unchanged, while transient transport retries
increased to five or six attempts with exponential backoff. These amendments
reran only missing model outputs and are not interpreted as additional reasoning
attempts.
