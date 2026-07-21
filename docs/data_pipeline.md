# VirtualHome Data Pipeline

## Source

The only benchmark environment in the study is VirtualHome from the pinned
Embodied Agent Interface checkout. Clean import reads the distributed PDDL
problems, gold PDDL plans, task names, predicates, and action metadata. It does
not read historical model outputs.

```bash
PYTHONPATH=src python -m embodied_gap.cli prepare-eai \
  --source-root external/embodied-agent-interface \
  --datasets virtualhome \
  --out-dir data/processed/eai_clean
```

Imported inventory: 338 VirtualHome tasks, 26 task families, 69 deterministic
training IDs, and 269 evaluation IDs. Fifty-seven training tasks have usable
gold plans for retrieval.

## Development and source-generation sets

`build-tasksets` is hard-scoped to VirtualHome. Current outputs are:

| Artifact | Rows | Purpose |
|---|---:|---|
| `rag_train.jsonl` | 57 | Frozen retrieval examples |
| `balanced_eval.jsonl` | 120 | Development only |
| `balanced_eval_20.jsonl` | 20 | Development smoke |
| `balanced_eval_50.jsonl` | 50 | Development pilot |
| `executable_eval.jsonl` | 239 | Source inventory with gold plans/goals |
| `heldout_virtualhome_119.jsonl` | 119 | Frozen source-generation pool |

The 119-task pool is task-ID-disjoint from development but family- and
template-overlapping. It is not described as unseen-family generalization.

## Retrieval and graph artifacts

The 57 training examples generate a flat retrieval corpus and a 4,986-edge
knowledge graph. Final P1 uses the flat retrieval corpus. The graph is retained
for exploratory analysis but is not a final treatment.

## Official evaluation cohort

The official evaluator-compatible cohort contains 84 tasks across eight task
families. It is produced by checking each gold plan against the pinned official
action vocabulary and task-specific object IDs. This pre-outcome screen removes
35 tasks that cannot be represented faithfully by the pinned evaluator.

The 84-task cohort is the only outcome population. Each experimental cell
contributes all 84 predictions; failed predictions are not dropped.

## Leakage controls

- Retrieval and repair memory use training/development artifacts only.
- Final task failures are never added to RAG or memory.
- Cohort selection uses gold plans and evaluator compatibility, not treatment
  outcomes.
- Task-family clustered inference addresses dependence among repeated task
  templates.
