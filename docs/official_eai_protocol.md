# Official VirtualHome Action Sequencing Protocol

## Authority and scope

The thesis uses one outcome authority only: the VirtualHome Action Sequencing
evaluator in the pinned EAI source submodule at commit
`531c62f8df2cb392bdf1907923c76da41cad4fe6`.

Authoritative evaluator:

`external/embodied-agent-interface/src/virtualhome_eval/evaluation/action_sequencing/scripts/evaluate_results.py`

Local PDDL execution is retained only inside the verifier and recovery
controller; its success flag is not a reported outcome.

## Fixed official cohort

The 119-task source pool contains tasks that the pinned evaluator cannot score
through the reviewed adapter. The cohort is frozen before comparing treatment
outputs:

```bash
PYTHONPATH=src python -m embodied_gap.cli build-official-virtualhome-cohort \
  --tasks data/processed/tasksets/heldout_virtualhome_119.jsonl \
  --out data/processed/tasksets/official_virtualhome_action_sequencing_v1.jsonl
```

The resulting cohort contains 84 tasks. Thirty-five are excluded because their
gold plans require `PLUGIN/PLUGOUT`, which the pinned executable evaluator does
not implement, or because task-specific official object IDs are ambiguous.
Selection uses benchmark resources and gold plans only, never treatment
outcomes.

## Common-denominator export

Every experimental cell exports exactly 84 records:

```bash
PYTHONPATH=src python -m embodied_gap.cli export-official-virtualhome \
  --runs runs/<experiment>/<model>/runs.jsonl \
  --tasks data/processed/tasksets/official_virtualhome_action_sequencing_v1.jsonl \
  --planner <planner> \
  --harness <harness> \
  --include-failed-predictions \
  --out runs/official_virtualhome_action_sequencing_v4/virtualhome/action_sequencing/<cell>_outputs.json
```

The adapter never guesses object IDs or silently drops actions. If a tested
model output is empty, rejected, malformed, or cannot be mapped, it is emitted
as `[]`; the pinned official evaluator counts it as a parsing failure. This
keeps failures in the common denominator.

## Pinned evaluator execution

```bash
PYTHONPATH=external/embodied-agent-interface/src \
MPLCONFIGDIR=/tmp/matplotlib-cache \
conda run -n eai-eval python -m eai_eval.cli \
  --dataset virtualhome \
  --eval-type action_sequencing \
  --mode evaluate_results \
  --llm-response-path runs/official_virtualhome_action_sequencing_v4 \
  --output-dir runs/official_virtualhome_action_sequencing_v4/results \
  --num-workers 8
```

The archived official run evaluated 60 model-specific cells: all five planners,
all four harnesses, and all three models on the same 84-task cohort. Its P2
implementation has since been replaced, so archived P2 outputs do not evaluate
the current `P2_graph_rag`. Non-P2 summaries remain valid for the archived run.
Official summaries and error files are stored under:

`runs/official_virtualhome_action_sequencing_v4/results/virtualhome/evaluate_results/action_sequencing/`

## Statistical extraction

The official log is parsed to recover per-task task-success outcomes, then
joined to the frozen task-family labels:

```bash
PYTHONPATH=src python -m embodied_gap.cli summarize-official-virtualhome \
  --results runs/official_virtualhome_action_sequencing_v4/results/virtualhome/evaluate_results/action_sequencing \
  --cohort data/processed/tasksets/official_virtualhome_action_sequencing_v1.jsonl \
  --log logs/action_sequencing_eval_<timestamp>.log \
  --out-json docs/final_official_virtualhome_results_v4.json \
  --out-markdown docs/final_official_virtualhome_results_v4.md
```

Primary inference resamples whole task families and reports exact paired
McNemar tests. This is an official-evaluator compatible-subset study, not a full
hidden-set leaderboard submission.
