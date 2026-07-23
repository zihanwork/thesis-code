# RAG Ablation and Leakage Audit

## Treatment

P1 adds one retrieved VirtualHome training demonstration to the same engineered
state–goal prompt used by P0-E. The final retrieval corpus contains 57 tasks and
no non-VirtualHome examples.

## Overlap audit

| Check | Result |
|---|---:|
| Task-ID overlap | 0/84 |
| Instruction seen in retrieval set | 84/84 |
| Task family seen | 84/84 |
| Gold plan seen somewhere | 76/84 (90.5%) |
| Selected demo from same family | 84/84 (100.0%) |
| Selected demo has exact gold plan | 75/84 (89.3%) |
| Selected-demo retrieval score | median 0.955; range 0.770-1.000 |

The correct interpretation is seen-family, unseen-ID template transfer. The
study must not describe P1 as demonstrating unseen-family generalization.

## Official outcome effect

The uniform run is complete for all three models and all 84 tasks. Official
outcomes and paired contrasts are reported in
`docs/final_official_virtualhome_results_v4.md`; local verifier scores are not
used as outcomes.

## GraphRAG follow-up boundary

P1 uses flat example retrieval. P2 adds deterministic relation-aware graph
propagation and graph-structured ranking as an incremental optimisation of P1.
It was developed against P1 under H0 on 120 development tasks disjoint from the
84-task outcome cohort. The matched P1/P2 follow-up across all four harnesses is
reported in `docs/final_official_virtualhome_graph_rag_replacement.md`. Because
that cohort had already been observed, these results are post-hoc same-cohort
evidence rather than an untouched confirmatory outcome. The version notice in
the frozen primary report preserves provenance for its P2 rows.
