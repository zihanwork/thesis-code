# Thesis Validity Audit

## Current status

The previous validity report was retired with the old result artifacts. The
current study is a planned, uniform VirtualHome Action Sequencing experiment:
60 model-specific cells, 3 models in every cell, and the same 84-task official
cohort in every cell.

## Evaluation authority

The pinned official VirtualHome Action Sequencing evaluator is the sole outcome
authority. Local PDDL checks remain only as verifier/recovery telemetry and
cannot produce thesis outcomes.

## Planned comparisons

The complete matrix supports the following paired comparisons, separately for
all three models:

- planner progression under each harness: `B0 -> P0-S -> P0-E -> P1 -> P2`;
- recovery progression within every planner: `H0 -> H2-R`, `H0 -> H2-M`, and
  `H0 -> H2-P`;
- planner-by-recovery interaction, estimated from the full 5 x 4 grid.

No numerical claim is valid until all 60 cells have been generated on all 84
tasks and scored by the pinned evaluator.

## Statistical validity

The 84-task cohort contains eight task families. Primary intervals resample whole
families, and paired binary changes use exact McNemar tests. Contrasts are
stratified by model, planner, and harness, with all three models included in
every planned comparison.

## Remaining limitations

- The 84 tasks are a compatible subset, not the complete hidden challenge set.
- The cohort contains eight task families after official compatibility screening.
- RAG and GraphRAG retrieval overlap must be audited before interpreting gains.
- GraphRAG is a new treatment; interpretation remains limited to this frozen cohort and the three tested models.
- Full factorial coverage estimates interaction on this cohort; it does not
  establish generalization beyond the frozen cohort or the three models.
