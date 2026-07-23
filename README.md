# Bridging the Goal-to-Action Gap

Research code and evidence for diagnosing and improving LLM embodied planning
on VirtualHome Action Sequencing.

## Final research framing

The study separates two intervention stages:

- **Planning-time knowledge constraint:** minimal prompting, structured
  state–goal prompting, constraint-engineered prompting, flat retrieval-grounded
  planning, and the new graph-retrieved P2-GraphRAG treatment.
- **Execution-time closed-loop control:** open loop, validator-feedback
  reflection, frozen repair memory, and a separately labelled symbolic PDDL
  recovery reference.

## One evaluation authority

All thesis outcome claims use the pinned official VirtualHome Action Sequencing
evaluator:

`external/embodied-agent-interface/src/virtualhome_eval/evaluation/action_sequencing/scripts/evaluate_results.py`

The official source submodule is pinned at
`531c62f8df2cb392bdf1907923c76da41cad4fe6`.

Local state/PDDL checking exists only inside the verifier and recovery
implementation; it is not a competing scoring standard and its success rate
is not reported as a result. Previous alternative benchmark evaluators and
standalone safety-benchmark entry points are not part of this project anymore.

## Current experiment table

The evidence covers the five planning treatments and four recovery conditions
on one 84-task VirtualHome cohort. B0 through P1 form the primary factorial
study; P1 and P2 form the matched GraphRAG follow-up.

| Dimension | Conditions | Count |
|---|---|---:|
| Planner | B0, P0-S, P0-E, P1 Flat RAG, P2 GraphRAG | 5 |
| Harness | H0, H2-R, H2-M, H2-P | 4 |
| Model | DeepSeek-V4-Flash, gpt-5.5, GLM-5-Turbo | 3 |
| **Total model-specific cells** | 5 x 4 x 3 | **60** |

The combined evidence contains 5,040 official records. P2-GraphRAG performs
entity linking, deterministic relation-aware graph propagation, Personalized
PageRank, multi-hop search, reranking, and state-constraint scoring over the
training-only graph. It was developed on 120 task-ID-disjoint development tasks
and evaluated against P1 on the same 84-task outcome cohort. Results are in
`docs/final_official_virtualhome_graph_rag_replacement.md`.

## Final official evidence


- Fixed official cohort:
  `data/processed/tasksets/official_virtualhome_action_sequencing_v1.jsonl`
- Cohort audit:
  `data/processed/tasksets/official_virtualhome_action_sequencing_v1.manifest.json`
- Official result report:
  `docs/final_official_virtualhome_results_v4.md`
- Machine-readable primary evidence:
  `docs/final_official_virtualhome_results_v4.json`
- Matched GraphRAG follow-up:
  `docs/final_official_virtualhome_graph_rag_replacement.md`
- Evaluation protocol:
  `docs/official_eai_protocol.md`
- Converged experiment design:
  `docs/revised_experiment_design.md`

The fixed cohort has 84 tasks across eight task families. It is screened using
gold plans and the pinned evaluator contract before treatment comparison. Every
cell retains all 84 predictions in the denominator; malformed or unmappable
model outputs are passed as empty sequences and counted as failures by the
official evaluator.

## Results status

All claim-bearing cells have been scored with the pinned VirtualHome Action
Sequencing evaluator. Primary B0-to-P1 results are in the v4 evidence files;
the matched P1/P2 follow-up is in the GraphRAG replacement evidence files.

## Reproduction

Create the main environment:

```bash
UV_CACHE_DIR=/tmp/thesis-uv-cache uv sync --locked --python 3.12
```

Run the unit suite:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Build the fixed cohort:

```bash
PYTHONPATH=src .venv/bin/python -m embodied_gap.cli \
  build-official-virtualhome-cohort \
  --tasks data/processed/tasksets/heldout_virtualhome_119.jsonl \
  --out data/processed/tasksets/official_virtualhome_action_sequencing_v1.jsonl
```

See `docs/official_eai_protocol.md` for export, official execution, and
family-clustered analysis commands.

## Claim boundary

This is an official-evaluator compatible-subset study, not a complete hidden
EAI leaderboard submission. The primary factorial study and matched GraphRAG
follow-up support model-stratified planning and recovery analyses on the frozen
84-task cohort; they do not establish generalization beyond this cohort.
