# Official EAI evaluation protocol

Status: VirtualHome action-sequencing evaluator verified locally on 2026-07-17.
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
