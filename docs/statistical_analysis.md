# Statistical Analysis — Official Outcome Only

## Unit of analysis

The outcome unit is one task in the fixed 84-task VirtualHome Action Sequencing
cohort. The primary endpoint is the pinned official evaluator's binary task
success. Secondary endpoints are official total-goal completion, execution
success, and official error categories.

The eight task families are treated as dependence clusters. Task-level
percentages remain descriptive, while primary uplift intervals resample whole
families with 10,000 bootstrap samples and seed 13. Exact paired McNemar tests
are reported for binary task changes.

## Reported paired contrasts

The primary study reports the planning progression
`B0/H0 -> P0-S/H0 -> P0-E/H0 -> P1/H0` and the H0-to-recovery contrasts within
each planner. The matched GraphRAG follow-up reports P1-to-P2 and H0-to-recovery
contrasts for P1 and P2. Every contrast is computed separately for
DeepSeek-V4-Flash, gpt-5.5, and GLM-5-Turbo.

The clustered interval and exact paired McNemar test answer different
questions. A family-bootstrap interval measures sensitivity of average uplift
to family composition; McNemar tests whether paired task changes are
asymmetric. Both use the same fixed 84-task denominator.

## Multiplicity and interpretation

Exact p-values are evidence-strength indicators, not independent discovery
claims. Interpretation emphasizes effect sizes, family-clustered intervals,
consistency across models, and the distinction between official task success,
total-goal completion, and execution success.

Frozen results are in `docs/final_official_virtualhome_results_v4.*` and
`docs/final_official_virtualhome_graph_rag_replacement.*`. Both were generated
from the pinned VirtualHome Action Sequencing evaluator.
