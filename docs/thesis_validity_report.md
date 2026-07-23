# Thesis Validity Audit

## Evidence structure

The study has two evidence layers over the same 84-task VirtualHome Action
Sequencing cohort and three models. The primary factorial experiment compares
B0, P0-S, P0-E, and P1 across H0, H2-R, H2-M, and H2-P. A matched follow-up
compares Flat RAG (P1) with its GraphRAG optimisation (P2) under the same four
harnesses. This follow-up does not require rerunning unrelated B0 or P0 cells.

## Evaluation authority

The pinned official VirtualHome Action Sequencing evaluator is the sole outcome
authority. Local execution checks are verifier/recovery telemetry and cannot
produce thesis outcomes. Every claim-bearing cell uses the same task IDs;
failed or malformed predictions remain in the 84-task denominator.

## Supported comparisons

The frozen evidence supports task-paired comparisons, separately for all three
models:

- planner progression from B0 through P1 within each harness in the primary
  factorial experiment;
- recovery contrasts from H0 to H2-R, H2-M, and H2-P while holding the initial
  planner fixed;
- matched P1-to-P2 contrasts in the GraphRAG follow-up.

The P1-to-P2 results are post-hoc same-cohort evidence. They support an
incremental GraphRAG comparison but not an untouched confirmatory claim.

## Statistical validity

The cohort contains eight task families. Confidence intervals resample all
observations in a family together using 10,000 bootstrap samples and seed 13.
Paired binary changes use exact two-sided McNemar tests. Results remain
stratified by model, planner, and harness.

## Reproducibility boundary

Run manifests record a dirty worktree, so a commit hash alone is insufficient
to reconstruct the evidence. Frozen configurations, artifact hashes, raw model
outputs, official exports, and evaluator outputs define the evidence of record.
Recorded HTTP 429/503 amendments reran missing GraphRAG-follow-up outputs with
five or six transient transport attempts and unchanged generation settings;
they did not give models additional reasoning attempts.

## Remaining limitations

- The 84 tasks are a compatibility-screened subset, not the complete hidden
  challenge set.
- The cohort contains eight task families.
- Retrieval has zero task-ID overlap, but all instructions and families are seen
  and gold-plan overlap is high; P1 tests seen-family, unseen-ID template
  transfer rather than unseen-family generalisation.
- GraphRAG conclusions are limited to a post-hoc matched comparison on this
  cohort and the three tested models.
