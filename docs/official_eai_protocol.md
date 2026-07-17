# Official EAI evaluation protocol

Status: VirtualHome action-sequencing evaluator and automatic export from a
real project run verified locally on 2026-07-17.
The BEHAVIOR evaluator and iGibson runtime import successfully, but official
execution is blocked until the separately licensed iGibson dataset is present.
The full eight-slot official submission is not yet ready.

## What has been verified

The pinned external EAI checkout runs in the existing `eai-eval` Conda
environment (Python 3.8.20). The official VirtualHome action-sequencing command
successfully read the repository smoke fixture and produced both
`summary.json` and `error_info.json`. This proves that prompt-output loading,
task lookup, simulation, trajectory classification, and official metric export
are connected.

The smoke fixture is:

`data/official_eai_smoke/virtualhome/action_sequencing/smoke_outputs.json`

## Final held-out compatible-subset diagnostic

After all four frozen final matrices completed, the selected non-symbolic
method (`gpt-5.5`, `P1_rag`, `H2_llm_reflection`) was passed through the strict
exporter. Of the 119 frozen VirtualHome tasks, 83 converted without guessing;
30 were rejected because the pinned evaluator does not execute
`PLUGIN/PLUGOUT`, and 6 were rejected because an object class had multiple
official scene IDs.

The pinned official evaluator ran on the 83 compatible tasks. All 83
trajectories were executable, while 43 satisfied the complete task criterion.
The official goal results were state 67/69, relation 64/69, action 1/38, and
combined 132/176. This is a compatibility-selected diagnostic rather than a
full held-out or leaderboard score. In particular, the action-goal result
proves that the custom PDDL final-state metric and official action/LTL metric
are not interchangeable. Exact hashes and exclusions are recorded in
`docs/final_official_virtualhome_evidence.json`.

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

## Exporting a real project method

The exporter selects one exact planner/harness pair from a run JSONL, restores
task-specific object IDs only from the pinned official prompts, validates the
pinned evaluator's action vocabulary and writes a non-overwriting official
response file plus an audit manifest:

```bash
embodied-gap export-official-virtualhome \
  --runs runs/<experiment>/<model>/runs.jsonl \
  --tasks data/processed/tasksets/<taskset>.jsonl \
  --planner P0_structured_prompt \
  --harness H0_open_loop \
  --out runs/<official-export>/virtualhome/action_sequencing/<model>_outputs.json
```

Strict mode writes no official response unless every expected VirtualHome task
can be converted; it still writes an audit manifest describing every rejection.
`--allow-partial` is for integration diagnosis only and marks the manifest
`complete: false` when tasks are skipped. It must not be used to claim a full
benchmark result. `--overwrite` must be explicit because official export
artifacts are non-overwriting by default.

The exporter intentionally refuses:

- an object class with more than one relevant scene ID;
- a canonical action without a reviewed semantic projection;
- actions advertised by prompt prose but absent from the pinned evaluator;
- a missing, duplicate, empty, or rejected selected run.

For example, canonical `put_on(character, object, surface)` becomes official
`PUTBACK(object, surface)`, while PDDL-only character arguments are removed.
No action is silently dropped.

### Real historical-run integration result

The historical GPT-5.5 `P0_prompt_only/H0_open_loop` run on
`balanced_eval_20.jsonl` contained 12 VirtualHome tasks. Eight converted without
guessing and passed the local format preflight. Four were correctly rejected:

- task `232_2`: `light` maps to three relevant official object IDs;
- tasks `327_2`, `819_1`, and `962_1`: `plug_in` maps to `PLUGIN`, which is in
  the prompt but missing from the pinned evaluator's executable action table.

The official evaluator ran successfully on the eight-record diagnostic file.
It then excluded tasks `496_1`, `540_1`, and `764_2` because their official gold
trajectories failed the evaluator's own gold-state gate, leaving an effective
denominator of five. On those five tasks it reported 100% execution success,
60% task success, 100% state goal, 100% relation goal, 0% action goal, and
81.8182% total goal. These numbers prove the end-to-end adapter path only. They
are a partial historical pilot with an upstream-filtered denominator and are
not a thesis result or leaderboard score. The immutable hashes, exact code
revisions, exclusions, denominator, and metrics are recorded in
`docs/official_virtualhome_pilot_evidence.json`.

The BEHAVIOR smoke fixture is:

`data/official_eai_smoke/behavior/action_sequencing/smoke_outputs.json`

It passes the pinned input-shape validator. The official command reaches
`ActionSequenceEvaluator` and attempts to construct the scene, proving that the
CLI, response lookup, parser, PyBullet, and iGibson imports are connected. It
then stops because `ig_dataset/scenes` is absent; no BEHAVIOR score has been
produced.

## Separate official-evaluator environment

The main thesis package remains locked by `uv.lock`. The official evaluator is
isolated because its pinned stack requires Python 3.8 and an old simulator
toolchain. Its direct dependencies are pinned in `environment.eai-eval.yml`,
while the exact source revisions are fixed by Git submodules.

Create the environment and install the pinned iGibson source with:

```bash
conda env create -f environment.eai-eval.yml
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  conda run -n eai-eval python -m pip install --no-deps -e external/iGibson
```

The CMake compatibility setting is required because the pinned iGibson build
predates CMake 4. On Apple Silicon, the conda-forge `pybullet` binary is used in
place of `pybullet-svl`, which has no compatible wheel and fails to compile.

BEHAVIOR evaluation additionally requires the iGibson scenes and BEHAVIOR
objects dataset. Stanford's official instructions require completing the
license form, placing the received `igibson.key`, and downloading approximately
20 GB of data. This project does not accept that agreement or download those
assets on a user's behalf. Follow the official dataset instructions:

<https://stanfordVL.github.io/iGibson/dataset.html>

The default location is `external/iGibson/igibson/data/ig_dataset`. To keep the
large licensed data outside Git, set `IGIBSON_DATASET_PATH` to another local
directory; both iGibson and the project preflight respect that environment
variable.

The current thesis does not require downloading these assets to continue. Its
BEHAVIOR numbers remain labelled `custom PDDL evaluation`; an official
BEHAVIOR score is an optional later run on a machine where the licensed assets
are available.

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
- `structurally_ready`: all slots are present and action shapes are valid;
- `official_runtime_ready`: both evaluator sources and required iGibson assets
  are present;
- `submission_ready`: both structural and runtime conditions hold.

The current smoke tree has validated action-sequencing fixtures for both
VirtualHome and BEHAVIOR, so two of the eight protocol slots are present. It is
still correctly marked `submission_ready: false`: the BEHAVIOR dataset is not
installed, and the other six module/environment slots are absent.

## Remaining blockers

1. The official BEHAVIOR evaluator imports and starts, but simulation requires
   the separately licensed, approximately 20 GB iGibson dataset. The user must
   complete Stanford's license process before it can be downloaded and tested.
2. The current thesis system produces plans for action sequencing. It does not
   yet produce official outputs for goal interpretation, subgoal decomposition,
   and transition modeling.
3. Canonical project actions cannot be declared official-format compatible
   until official object IDs and environment-specific action semantics are
   retained through generation.

Until these blockers are removed, all existing scores must remain labelled
`custom subset` or `local held-out`; none is an official leaderboard score.
