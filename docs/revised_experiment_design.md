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
| H2-P | Composite symbolic recovery reference | Replace the failed plan using a macro plan, grounded PDDL search, then action-model fallback |

The verifier is an intervention component. Its internal PDDL/state checks are
not an outcome metric and are never reported as task success.

## Evidence design

The primary factorial study crosses B0, P0-S, P0-E, and P1 with four harnesses
and three models on the same 84-task cohort. It estimates planning, recovery,
and planner-by-recovery effects through Flat RAG.

GraphRAG is a matched follow-up optimisation of P1. The P1/P2 follow-up crosses
both planners with the same four harnesses, three models, and 84 tasks. This
2 x 4 x 3 design directly estimates P1-to-P2 changes without rerunning unrelated
prompt baselines.

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
linking, deterministic relation-aware graph propagation, Personalized PageRank,
three-hop path search, relation-aware reranking, and state-constraint scoring.
The selected graph evidence is rendered with triples, paths, score components,
and action chains before action generation. Query gold plans are never read.

The flat `P1_rag` condition is the direct control. Development uses
`configs/experiments/graph_rag_development.json` on 120 tasks with zero task-ID
overlap with the observed 84-task cohort. The matched P1/P2 follow-up is reported
in `docs/final_official_virtualhome_graph_rag_replacement.md`; its claim boundary
is the observed cohort and tested model APIs.

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
