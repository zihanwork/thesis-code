# Official EAI evaluation protocol

Status: VirtualHome action-sequencing evaluator verified locally on 2026-07-17;
full eight-slot official submission is not yet ready.

## What has been verified

The pinned external EAI checkout runs in the existing `eai-eval` Conda
environment (Python 3.8.20). The official VirtualHome action-sequencing command
successfully read the repository smoke fixture and produced both
`summary.json` and `error_info.json`. This proves that prompt-output loading,
task lookup, simulation, trajectory classification, and official metric export
are connected.

The smoke fixture is:

`data/official_eai_smoke/virtualhome/action_sequencing/smoke_outputs.json`

It deliberately contains multiple tasks whose combined goals cover state,
relation, and action metrics. A single task can have zero goals in one or more
categories, and the pinned upstream evaluator divides by each category count
without guarding zero. Such a one-task smoke run can therefore crash even when
the input file was loaded correctly.

Run the official evaluator with:

```bash
PYTHONPATH=external/embodied-agent-interface/src \
MPLCONFIGDIR=/tmp/matplotlib-cache \
conda run -n eai-eval python -m eai_eval.cli \
  --dataset virtualhome \
  --eval-type action_sequencing \
  --mode evaluate_results \
  --llm-response-path data/official_eai_smoke \
  --output-dir /tmp/eai_official_smoke \
  --num-workers 1
```

The smoke score is only an integration check and must not be reported as an
experimental result.

## Pinned-format discrepancy

For VirtualHome action sequencing, the generated prompt prose says that model
outputs should contain object names without IDs. However, the evaluator code at
the pinned external commit checks that every action argument list contains
name/ID pairs and rejects name-only output. The working shape is therefore:

```json
{
  "identifier": "11_1",
  "llm_output": "[{\"WALK\":[\"floor_lamp\",1000]}]"
}
```

This project follows the executable pinned code and records the discrepancy.
It does not silently translate the current canonical PDDL plans because those
plans do not always retain the official scene object IDs.

BEHAVIOR uses a different action shape inside `llm_output`:

```json
[
  {"action": "LEFT_GRASP", "object": "candle_0"},
  {"action": "LEFT_PLACE_INSIDE", "object": "basket_0"}
]
```

## Automatic preflight

The project command below checks the two-dataset by four-module response tree,
validates action-sequencing shapes, and writes a machine-readable report:

```bash
uv run --frozen embodied-gap check-official-eai \
  --responses data/official_eai_smoke \
  --out reports/official_eai_preflight.json
```

The report distinguishes:

- `action_sequencing_shapes_valid`: files that satisfy the pinned input shape;
- `all_slots_present`: all four modules for both environments are present;
- `submission_ready`: both conditions hold.

The current smoke tree has only the VirtualHome action-sequencing slot, so it is
correctly marked `submission_ready: false`.

## Remaining blockers

1. Importing the official BEHAVIOR evaluator currently fails because `igibson`
   is not installed in the `eai-eval` environment. The official installer is a
   separate, potentially heavy dependency step and has not been run
   automatically.
2. The current thesis system produces plans for action sequencing. It does not
   yet produce official outputs for goal interpretation, subgoal decomposition,
   and transition modeling.
3. Canonical project actions cannot be declared official-format compatible
   until official object IDs and environment-specific action semantics are
   retained through generation.

Until these blockers are removed, all existing scores must remain labelled
`custom subset` or `local held-out`; none is an official leaderboard score.
