# Architecture: Two-Stage Control With One Outcome Authority

The project uses a two-layer architecture.

## Layer 1: Initial Planning

Initial planners generate the first candidate action sequence.

- `B0_minimal_prompt`: minimal instruction/action-list baseline.
- `P0_structured_prompt`: structured PDDL-informed prompt baseline.
- `P0_engineered_prompt`: structured prompt with a fixed constraint checklist.
- `P1_rag`: retrieves task demonstrations on top of the engineered prompt.
- `P2_graph_rag`: a matched P1 optimisation using entity linking, deterministic relation-aware graph propagation, Personalized PageRank, multi-hop evidence paths, reranking, and state constraints.

## Layer 2: Execution Harness

Harness modes control what happens after a candidate plan is generated.

- `H0_open_loop`: execute directly.
- `H2_llm_reflection`: replan once from explicit validator feedback.
- `H2_memory`: reflection with one frozen development repair example.
- `H2_pddl_recovery`: composite symbolic recovery using a macro plan, grounded
  PDDL search, and an action-model fallback.

## Research Claims Supported

This architecture supports separable comparisons:

1. Planning-time improvement from B0 through P1 under a fixed harness.
2. Recovery improvement within every initial planner.
3. Conditional recovery effects across the complete planner-harness cells.
4. Matched P1-to-P2 comparison for the GraphRAG optimisation.

P2 reads the frozen training-only KG edge artifact and renders retrieved graph
triples, paths, score components, and action chains before generation. H2-P is
the separately labelled symbolic recovery reference.

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
