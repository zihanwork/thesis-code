# Architecture

The project uses a two-layer architecture.

## Layer 1: Initial Planning

Initial planners generate the first candidate action sequence.

- `P0_prompt_only`: structured prompt-only baseline.
- `P1_retrieval_augmented`: retrieves similar demonstrations and adapts the plan.
- `P2_graph_grounded`: searches an object-action-state precondition/effect graph.

## Layer 2: Execution Harness

Harness modes control what happens after a candidate plan is generated.

- `H0_open_loop`: execute directly.
- `H1_verifier_gated`: validate before execution and block/reject unsafe plans.
- `H2_full_recovery`: validate, repair, replan, and retry.

## Research Claims Supported

This architecture supports three separable ablations:

1. Planner improvement: P0 vs P1 vs P2 under the same harness.
2. Harness improvement: H0 vs H1 vs H2 under the same planner.
3. Interaction effect: whether stronger planners still benefit from stronger harnesses.

## Data Flow

```text
Task JSONL
  -> Task schema
  -> Initial planner
  -> Harness controller
  -> Validator / executor / repair router
  -> Execution trace
  -> Evaluation metrics
  -> Runs and summary artifacts
```
