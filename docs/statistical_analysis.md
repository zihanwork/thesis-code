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

## Planned paired contrasts

No result table is currently valid. All former numeric contrasts were removed
because the old runs did not satisfy the current uniform model/task protocol.
After the new run, the primary comparisons will be computed separately for
DeepSeek-V4-Flash, gpt-5.5, and GLM-5-Turbo:

- `B0/H0 -> P0-S/H0 -> P0-E/H0 -> P1/H0`
- `P1/H0 -> P1/H2-R`
- `P1/H0 -> P1/H2-M`
- `P1/H0 -> P1/H2-P`
- `P1/H0 -> P2-GraphRAG/H0`

The clustered interval and exact paired McNemar test answer different
questions. A family-bootstrap interval measures sensitivity of the average
uplift to family composition; McNemar tests whether paired task changes are
asymmetric. Both will be retained for the new 84-task cohort.

## Multiplicity and interpretation

The matrix contains multiple planned contrasts. Exact p-values are evidence
strength indicators, not independent discovery claims. The analysis will
emphasize effect sizes, family-clustered intervals, consistency across all
three models, and the distinction between official task success, total-goal
completion, and execution success.

All result files will be generated only from the pinned VirtualHome Action
Sequencing evaluator after the complete 60-cell run completes.
