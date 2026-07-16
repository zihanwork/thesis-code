# Architecture

The project uses a two-layer architecture.

## Layer 1: Initial Planning

Initial planners generate the first candidate action sequence.

- `B0_minimal_prompt`: minimal instruction/action-list baseline.
- `P0_structured_prompt`: structured PDDL-informed prompt baseline.
- `P0_engineered_prompt`: structured prompt with a fixed constraint checklist.
- `P1_rag`: retrieves task demonstrations on top of the engineered prompt.
- `P2_symbolic_pddl`: model-independent symbolic planning reference.

## Layer 2: Execution Harness

Harness modes control what happens after a candidate plan is generated.

- `H0_open_loop`: execute directly.
- `H1_verifier_gated`: validate before execution and block/reject unsafe plans.
- `H2_local_recovery`: safety rules and isolated local deterministic repair.
- `H2_llm_reflection`: explicit validator feedback to the original model.
- `H2_error_specific`: error-type-specific LLM repair guidance.
- `H2_memory`: LLM repair with a frozen failure-repair example.
- `H2_combined`: local, error-specific, and memory recovery without PDDL fallback.
- `H2_pddl_recovery`: isolated symbolic PDDL fallback.
- `H2_full_recovery`: legacy mixed pilot policy; excluded from final experiments.

## Research Claims Supported

This architecture supports separable ablations:

1. Planning-time improvement: B0 vs P0 vs P0-PE vs P1 under H0.
2. Recovery improvement: isolated Local vs LLM vs typed vs memory vs PDDL recovery.
3. Interaction effect: whether RAG and LLM feedback recovery are complementary.
4. Symbolic reference: P2 once, outside the LLM model matrix.

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
