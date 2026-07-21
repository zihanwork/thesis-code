# Data Split and Official Cohort Protocol

## Split roles

- **Retrieval training:** 57 VirtualHome tasks with gold plans.
- **Development:** 120 family-balanced VirtualHome tasks for method selection.
- **Frozen source pool:** 119 task-ID-disjoint VirtualHome tasks used to produce
  all final plans.
- **Official outcome cohort:** 84 source-pool tasks supported by the pinned
  official Action Sequencing evaluator.

Task identifiers are disjoint between development and the source pool. Task
families, instructions, and action templates overlap substantially, so the
study is a seen-family/unseen-ID transfer evaluation, not unseen-family
generalization.

## Official cohort rule

Compatibility is determined before looking at treatment outcomes. A task is
included only if:

1. it has a matching official Action Sequencing prompt;
2. every gold-plan action is implemented by the pinned evaluator;
3. each referenced object has one unambiguous task-specific official ID.

This yields 84 included and 35 excluded tasks. The same 84 identifiers are used
for every cell. Model-specific conversion failures remain in the denominator
and are scored by the official evaluator as failed predictions.

## Statistical dependence

The 84 tasks belong to eight task families. Paired effect intervals therefore
resample whole families rather than treating all tasks as independent. Exact
paired McNemar tests are also reported.

## Claim boundary

The cohort is a reproducible official-evaluator compatible subset of the
project's frozen source pool. It is not the full official hidden challenge set
and does not support a leaderboard rank claim.
