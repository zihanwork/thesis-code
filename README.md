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

The historical “RAG + graph × harness” 3 × 3 is retired. Its graph-grounded
planner and full-recovery harness both called the same PDDL search, so the old
P2 data and charts are invalid and have been removed. P2 is now a new
GraphRAG treatment that reads training-only graph edges and must be evaluated
from a fresh run.

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

The historical 3 x 3 planner-by-harness grid has been retired. The completed
experiment is a full 60-cell factorial matrix. Every planner-harness condition
runs DeepSeek-V4-Flash, gpt-5.5, and GLM-5-Turbo on the same 84-task official
VirtualHome Action Sequencing cohort:

| Dimension | Conditions | Count |
|---|---|---:|
| Planner | B0, P0-S, P0-E, P1 Flat RAG, P2 GraphRAG | 5 |
| Harness | H0, H2-R, H2-M, H2-P | 4 |
| Model | DeepSeek-V4-Flash, gpt-5.5, GLM-5-Turbo | 3 |
| **Total model-specific cells** | 5 x 4 x 3 | **60** |

Every cell uses the same frozen 84-task cohort, yielding 5040 official records.
P2-GraphRAG retrieves training-only task subgraphs from `kg_edges.jsonl` and
cannot reuse deleted symbolic P2 outputs. The complete factorial matrix supports
planner, recovery, and planner-by-recovery comparisons for all three models.

## Final official evidence


- Fixed official cohort:
  `data/processed/tasksets/official_virtualhome_action_sequencing_v1.jsonl`
- Cohort audit:
  `data/processed/tasksets/official_virtualhome_action_sequencing_v1.manifest.json`
- Official result report:
  `docs/final_official_virtualhome_results_v4.md`
- Machine-readable evidence:
  `docs/final_official_virtualhome_results_v4.json`
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

The complete 60-cell matrix has been run with all three models on all 84 tasks
and scored by the pinned VirtualHome Action Sequencing evaluator. Current
results and paired statistics are reported only in the v4 evidence files above.

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
EAI leaderboard submission. The complete 5 x 4 planner-harness grid supports
model-stratified planning, recovery, and interaction analyses on the frozen
84-task cohort; it does not establish generalization beyond this cohort.
