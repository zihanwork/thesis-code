# Clean EAI Data Pipeline

This project treats Embodied Agent Interface (EAI) source resources as the
authoritative benchmark input for the thesis experiments. Historical model
outputs and diagnostics are excluded from the new experimental data.

## Source Policy

The EAI adapter reads only raw benchmark resources from the repository-local
checkout at `external/embodied-agent-interface`. Paths stored in generated
metadata are relative to that checkout, not tied to a user home directory.

Included raw resources:

- `problem_pddl/**/*.pddl`
- `id2task.json`
- `gold_pddl_plan.json`
- `id2action.json`
- `id2predicate.json`
- `success_task.json`
- `failed_task.json`

Excluded historical or dirty resources:

- `output/`
- `output_norm_all/`
- `output_single_norm/`
- `diagnostics/`
- `evaluate_results/`

## Generated Files

The command below generates the clean processed benchmark files:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli prepare-eai \
  --source-root external/embodied-agent-interface \
  --out-dir data/processed/eai_clean
```

For an EAI checkout outside this repository, either pass `--source-root` or
set `EAI_SOURCE_ROOT`. Runtime path resolution tries the environment override,
the metadata hint, and the repository-local checkout in that order. Stale
absolute paths from older JSONL exports are ignored when a portable candidate
exists.

### PDDL layout

- VirtualHome domain: `src/virtualhome_eval/resources/virtualhome/virtualhome.pddl`
- VirtualHome problems: `src/virtualhome_eval/resources/virtualhome/problem_pddl/`
- Normalized BEHAVIOR domain used by the action-sequencing evaluator:
  `src/virtualhome_eval/resources/behavior/behavior.pddl`
- Normalized BEHAVIOR problems:
  `src/virtualhome_eval/resources/behavior/problem_pddl/`
- Original BEHAVIOR transition-modeling domain fallback:
  `src/behavior_eval/evaluation/transition_modeling/resources/behavior_new.pddl`
- Original BEHAVIOR problem source fallback:
  `src/behavior_eval/evaluation/transition_modeling/resources/pddl_behavior/`

Generated outputs:

- `data/processed/eai_clean/virtualhome_tasks.jsonl`
- `data/processed/eai_clean/behavior_tasks.jsonl`
- `data/processed/eai_clean/all_tasks.jsonl`
- `data/processed/eai_clean/virtualhome_raw_pddl.jsonl`
- `data/processed/eai_clean/behavior_raw_pddl.jsonl`
- `data/processed/eai_clean/manifest.json`

Current clean import summary:

| Dataset | PDDL tasks | Task families | Gold plans | Empty goals | Train | Eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VirtualHome | 338 | 26 | 296 | 32 | 69 | 269 |
| BEHAVIOR | 100 | 100 | 100 | 0 | 18 | 82 |
| Combined | 438 | 126 | 396 | 32 | 87 | 351 |

VirtualHome has 342 task identifiers in `id2task.json`, but only 338 matching
PDDL problem files are present in the EAI source tree. The clean import uses
the PDDL files as the ground truth task set. The missing PDDL identifiers are:
`339_1`, `627_1`, `84_1`, and `93_1`.

## Schema Notes

Each processed task stores:

- natural-language instruction
- PDDL initial facts
- PDDL goal facts
- object/type map
- gold PDDL plan when available
- non-leaky domain action names in `allowed_actions`
- task-level action and predicate names in metadata
- source-relative PDDL path

The import keeps `gold_plan` separate from `allowed_actions`. This prevents
LLM planners from seeing the answer sequence while still allowing gold-plan
validation and supervised analysis.

The current execution layer includes a PDDL-backed interpreter for EAI
action-sequencing tasks. It supports conjunction, disjunction, negation,
existential/universal quantifiers, and conditional effects.

## Task Set Expansion

The command below builds thesis-ready task subsets from the clean combined EAI
tasks:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli build-tasksets \
  --tasks data/processed/eai_clean/all_tasks.jsonl \
  --out-dir data/processed/tasksets \
  --per-family 8
```

Generated task sets:

| Task set | Rows | Purpose |
| --- | ---: | --- |
| `rag_train.jsonl` | 75 | Retrieval demonstrations and memory construction |
| `full_eval.jsonl` | 351 | Full non-train evaluation inventory |
| `executable_eval.jsonl` | 321 | Main candidates with non-empty goals and gold plans |
| `balanced_eval.jsonl` | 202 | Family-balanced main experiment set |
| `balanced_eval_20.jsonl` | 20 | Cost-controlled pilot sampled by dataset and difficulty |
| `balanced_eval_50.jsonl` | 50 | Medium pilot sampled by dataset and difficulty |

The task set builder also annotates each record with a deterministic difficulty
label based on gold-plan length, goal count, and object count. Current
`balanced_eval` difficulty mix: 61 easy, 67 medium, and 74 hard tasks.
`balanced_eval_20` contains 8 BEHAVIOR and 12 VirtualHome tasks, with 7 easy,
6 medium, and 7 hard tasks. `balanced_eval_50` contains 20 BEHAVIOR and 30
VirtualHome tasks, with 16 easy, 16 medium, and 18 hard tasks.

## PDDL Gold-Plan Validation

Gold plans are validated with:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli validate-pddl-gold \
  --tasks data/processed/tasksets/balanced_eval.jsonl \
  --out-dir runs/pddl_gold_validation/balanced_eval_full
```

Current validation evidence:

| Task set | Rows | Executable rate | Goal success rate | Overall success |
| --- | ---: | ---: | ---: | ---: |
| `balanced_eval_20.jsonl` | 20 | 1.000 | 1.000 | 1.000 |
| `balanced_eval_50.jsonl` | 50 | 1.000 | 1.000 | 1.000 |
| `balanced_eval.jsonl` | 202 | 1.000 | 1.000 | 1.000 |
| `executable_eval.jsonl` | 321 | 1.000 | 1.000 | 1.000 |

This validation confirms that the clean EAI task import, domain PDDL files, and
gold PDDL plans are mutually consistent under the project executor.
