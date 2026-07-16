# Statistical and cost analysis protocol

Every new experiment run writes `analysis.json` beside `metrics.jsonl` and
`runs.jsonl`. The report is generated from task-level outcomes and contains:

- Wilson 95% confidence intervals for task, execution, and safe success rates;
- exact two-sided McNemar tests for every task-paired method comparison;
- paired task-success uplift with a deterministic percentile-bootstrap 95% CI;
- separate summaries by dataset, difficulty, and task family;
- failure-type counts, average attempts, and average repairs;
- attributable LLM calls, tokens, latency, estimated cost, and cost per success;
- PDDL/symbolic explored-state counts and measured search time.

An initial planner call is computed once and reused across harness variants by
the matrix runner. For method-level comparison, `analysis.json` attributes that
same initial call to each harness method as the cost it would incur if run alone.
The matrix-level manifest telemetry remains the source for the actual amount
spent by that invocation.

Cost remains `null` with `pricing_not_configured` unless both input and output
rates are recorded in the model config. Do not substitute guessed prices after
the run; retain the original configuration and, if necessary, make a separate
clearly labelled post-hoc cost table.

Historical runs can be analyzed without rerunning models:

```bash
uv run --frozen python -m embodied_gap.cli analyze-run \
  --metrics RUN_DIR/metrics.jsonl \
  --runs RUN_DIR/runs.jsonl \
  --out RUN_DIR/analysis.json
```

Statistical significance is supporting evidence, not a replacement for effect
size, confidence interval, dataset-stratified results, and failure analysis.
