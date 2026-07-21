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

## Graph boundary

P1 remains flat example retrieval. P2-GraphRAG is the separate graph-conditioned
control and uses the same three models, 84 tasks, and H0 execution condition.
