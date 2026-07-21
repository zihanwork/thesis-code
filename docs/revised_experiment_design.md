# Converged Experiment Design

## Research position

The thesis studies a two-stage control problem in embodied planning:

1. **Planning-time knowledge constraint** controls what information the LLM
   receives before producing its first action sequence.
2. **Execution-time closed-loop control** diagnoses and repairs a plan after a
   verifier detects a concrete failure.

This framing is deeper than a component list such as “structured + RAG” and
more precise than the engineering label “harness engineering.” It separates
where information enters the pipeline, which failure it can affect, and which
mechanism deserves credit for an improvement.

## Why the historical 3 × 3 was retired

The historical matrix crossed prompt-only, RAG, and “graph grounded” planners
with open loop, verifier gating, and full recovery. It was not a valid
factorial design:

- the final P2 path used `pddl_grounded_search`, not the knowledge graph;
- P2 and H2 both called the same symbolic PDDL machinery, so their cells were
  not independent mechanisms;
- verifier-only H1 blocks invalid actions but does not repair them, making many
  H0/H1 cells mechanically identical;
- “full recovery” bundled local patching, LLM feedback, memory, and symbolic
  fallback, preventing causal attribution.

The old grid looked simple, but its apparent symmetry concealed confounding.
It must not be presented as the confirmatory thesis experiment.

## Final interventions

### Planning-time knowledge constraint

| ID | Paper label | Information added before generation |
|---|---|---|
| B0 | Minimal prompt | Instruction and output requirement only |
| P0-S | Structured state–goal prompt | Objects, state, goals, and action signatures |
| P0-E | Constraint-engineered prompt | P0-S plus a fixed precondition/order checklist |
| P1 | Retrieval-grounded planning | P0-E plus a task-conditioned training demonstration |
| P2 | GraphRAG planning | Retrieve training-only task subgraphs and graph-derived action chains before generation |

### Execution-time closed-loop control

| ID | Paper label | Response to detected failure |
|---|---|---|
| H0 | Open loop | No post-generation intervention |
| H2-R | Validator-feedback reflection | Return the concrete validation failure to the same LLM and replan once |
| H2-M | Memory-augmented repair | Reflection plus one frozen development repair example |
| H2-P | Symbolic recovery reference | Replace the failed plan using PDDL search |

The verifier is an intervention component. Its internal PDDL/state checks are
not an outcome metric and are never reported as task success.

## Actual confirmatory design

The final design is a complete 5 x 4 x 3 factorial matrix:

- **Five planners:** B0, P0-S, P0-E, P1 Flat RAG, and P2 GraphRAG.
- **Four harnesses:** H0, H2-R, H2-M, and H2-P.
- **Three models:** DeepSeek-V4-Flash, gpt-5.5, and GLM-5-Turbo.
- **One cohort:** the same 84 official-compatible VirtualHome tasks in every
  model-specific cell.

This yields 20 planner-harness combinations, 60 model-specific cells, and 5040
official records. There are no empty cells. The former symbolic P2 source runs
and result rows were deleted and cannot be reused. Crossing every planner with
every harness supports direct planning, recovery, and planner-by-recovery
comparisons without changing model coverage or task denominators.

## Research questions and identifiable claims

- **RQ1 — Planning-time grounding:** How much do structured constraints and
  retrieved procedural examples improve official task success?
- **RQ2 — Failure diagnosis:** Which official trajectory and goal failures
  remain after each planning intervention?
- **RQ3 — Recovery and interaction:** How much do reflection, frozen repair
  memory, and symbolic recovery improve each planning method, and do their
  effects interact with planning-time grounding?
- **RQ4 — Model transfer:** Do the planning-time effects reproduce across model
  families?
- **RQ5 — Cost:** What extra calls, tokens, latency, and symbolic search are
  required?

RQ3 is estimable because every planning method is crossed with every recovery
method. Interaction claims remain model-stratified and paired by task; no cell
may be omitted from confirmatory reporting.

## GraphRAG treatment definition

P2-GraphRAG is a distinct planning treatment. It reads the frozen, training-only
`data/knowledge/eai_train/kg_edges.jsonl` as one global graph and performs entity
linking, relation-aware graph-neural message passing, Personalized PageRank,
three-hop path search, relation-aware reranking, and state-constraint scoring.
The selected graph evidence is rendered with triples, paths, score components,
and action chains before action generation. Query gold plans are never read.

The flat `P1_rag` condition remains the direct control. Development uses
`configs/experiments/graph_rag_development.json` on 120 tasks with zero overlap
with the observed 84-task cohort. The P2 implementation in the archived v4 run
has been superseded; its P2 rows are historical evidence only. The replacement
P1/P2 replication is reported in
`docs/final_official_virtualhome_graph_rag_replacement.md`. It is post-hoc
same-cohort evidence; confirmatory generalization claims still require a new
untouched compatible cohort.

## Single evaluation authority

All thesis outcome claims use the pinned official VirtualHome Action
Sequencing evaluator. The fixed cohort is built before examining treatment
outcomes by retaining tasks whose gold plan is supported by the pinned action
vocabulary and whose task-specific object IDs are unambiguous. Failed model
predictions remain in the common denominator and are passed to the official
evaluator as empty sequences.

Primary outcome: official task success rate.

Secondary official outcomes: total goal completion, state/relation/action goal
completion, execution success, and official error categories.

Inference resamples whole task families and reports exact paired McNemar tests.
