# Architecture: Two-Stage Control With One Outcome Authority

The project uses a two-layer architecture.

## Layer 1: Initial Planning

Initial planners generate the first candidate action sequence.

- `B0_minimal_prompt`: minimal instruction/action-list baseline.
- `P0_structured_prompt`: structured PDDL-informed prompt baseline.
- `P0_engineered_prompt`: structured prompt with a fixed constraint checklist.
- `P1_rag`: retrieves task demonstrations on top of the engineered prompt.
- `P2_graph_rag`: global training-graph planning with entity linking, relation-aware graph-neural embeddings, Personalized PageRank, multi-hop evidence paths, relation-aware reranking, and state constraints.

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
3. Conditional recovery effect: how much LLM feedback recovery improves P1 RAG
   plans. A true interaction requires the currently absent P0/Reflection cell.
4. Symbolic reference: P2 once, outside the LLM model matrix.

P2 is now in this list as a distinct GraphRAG treatment. It reads the frozen
training-only KG edge artifact and renders retrieved graph triples before
planning. It is not the same implementation as the deleted PDDL-backed P2.

## Data Flow

```text
Task JSONL
  -> Task schema
  -> Initial planner
  -> Harness controller
  -> Validator / executor / repair router
  -> Candidate final action sequence
  -> Official-format adapter (fixed 84-task cohort)
  -> Pinned VirtualHome Action Sequencing evaluator
  -> Official goal, trajectory, and error metrics
```

The validator/executor trace is internal intervention telemetry. Only the
pinned official evaluator is allowed to award outcome success.
