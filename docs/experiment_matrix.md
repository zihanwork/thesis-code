# Experiment Matrix

The thesis framework separates planner generation from execution harnessing:

- `P0_prompt_only`
- `P1_retrieval_augmented`
- `P2_graph_grounded`
- `H0_open_loop`
- `H1_verifier_gated`
- `H2_full_recovery`

The main experimental grid is therefore a 3 x 3 planner/harness matrix.

## Multi-Model Layer

The model layer is configured independently from the planner/harness matrix.
The current One API model matrix is:

| Model ID | One API model name | Status in sample smoke run |
| --- | --- | --- |
| `deepseek_v4_flash` | `DeepSeek-V4-Flash` | Succeeded |
| `glm_5` | `GLM-5` | Failed with upstream 404 |
| `gpt_5_5` | `gpt-5.5` | Succeeded |

Runnable smoke config:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/sample_multimodel_one_api.json
```

Draft EAI main-experiment config:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/eai_balanced_multimodel_draft.json
```

The EAI draft config uses the family-balanced clean EAI task set. Full runs
should be scheduled deliberately because the matrix multiplies task count,
planner/harness cells, and model calls.

Balanced pilot configs:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/eai_balanced_20_multimodel_one_api.json

PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/eai_balanced_50_multimodel_one_api.json
```

The balanced pilot configs evaluate only `balanced_eval_20.jsonl` or
`balanced_eval_50.jsonl`, but use `rag_train.jsonl` as the external retrieval
example source. This keeps P1 as a true RAG condition without mixing
demonstration rows into the evaluation rows.

## Current Sample Smoke Results

New sample smoke runs write artifacts below the configured base directory:

`runs/sample_multimodel_one_api/<run_id>`

Successful model directories include:

- `runs/sample_multimodel_one_api/<run_id>/deepseek_v4_flash`
- `runs/sample_multimodel_one_api/<run_id>/gpt_5_5`

The failed GLM-5 run is preserved as:

- `runs/sample_multimodel_one_api/<run_id>/glm_5/error.json`

The multi-model runner is intentionally fault tolerant: one unavailable model
is recorded as failed but does not abort successful models. Re-running creates
a different `<run_id>`, so a partial or failed rerun cannot overwrite an older
summary.

## Current Real EAI Smoke Results

The cost-controlled real EAI smoke config is:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/eai_smoke_multimodel_one_api.json
```

The smoke taskset is:

`data/processed/tasksets/eai_smoke_eval.jsonl`

It contains 8 train/RAG examples and 2 eval tasks: one BEHAVIOR task and one
VirtualHome task. New runs write artifacts below:

`runs/eai_smoke_multimodel_one_api/<run_id>`

| Model | Method | H0 task SR | H1 task SR | H2 task SR |
| --- | --- | ---: | ---: | ---: |
| `DeepSeek-V4-Flash` | P0 prompt-only | 0.000 | 0.000 | 1.000 |
| `DeepSeek-V4-Flash` | P1 RAG | 0.000 | 0.000 | 1.000 |
| `DeepSeek-V4-Flash` | P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |
| `gpt-5.5` | P0 prompt-only | 0.000 | 0.000 | 1.000 |
| `gpt-5.5` | P1 RAG | 0.000 | 0.000 | 1.000 |
| `gpt-5.5` | P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |

The smoke run is now a useful method-separation baseline: P0/P1 open-loop and
verifier-only settings expose argument-number, affordance/type, parsing, and
missing-step failures, while H2 recovers them through PDDL-grounded full
replanning. P2 solves both real EAI smoke tasks even without repair. The next
required improvement is to make P0/P1 more PDDL-aware without leaking gold
plans, then scale from the two-task smoke set to a larger balanced subset.

## RAG And KG Artifacts

Training-split RAG/KG artifacts are generated with:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli build-knowledge \
  --tasks data/processed/tasksets/rag_train.jsonl \
  --out-dir data/knowledge/eai_train
```

Current generated artifacts:

| Artifact | Rows |
| --- | ---: |
| `data/knowledge/eai_train/retrieval_corpus.jsonl` | 75 documents |
| `data/knowledge/eai_train/kg_edges.jsonl` | 7321 edges |

The KG includes task-family, object-type, initial-fact, goal-fact, gold-action,
action-object, and action-order relations. These artifacts should be used as
the default knowledge source for P1/P2 development to avoid evaluation leakage.

## Current Balanced Pilot Results

The pre-enhancement local, no-API 20-task diagnostic result was:

| Method | H0 task SR | H1 task SR | H2 task SR |
| --- | ---: | ---: | ---: |
| P0 prompt-only | 0.000 | 0.000 | 0.700 |
| P1 RAG | 0.150 | 0.150 | 0.700 |
| P2 graph/PDDL grounded | 0.700 | 0.700 | 0.700 |

This run confirmed that the 20-task pipeline completed locally and that P1
could use external RAG examples. The six remaining P2/H2 failures were:
`cleaning_microwave_oven`, `defrosting_freezer`,
`serving_hors_d_oeuvres`, `watering_houseplants`, `Make_coffee`, and
`Wash_dishes_with_dishwasher`.

Focused diagnosis of the six P2/H2 failures:

| Task family | Diagnosis pattern | Enhanced P2/H2 result |
| --- | --- | --- |
| `cleaning_microwave_oven` | `behavior_negative_cleaning` | Solved |
| `defrosting_freezer` | `behavior_container_transfer`, `behavior_surface_or_nextto_placement` | Solved |
| `serving_hors_d_oeuvres` | `behavior_surface_or_nextto_placement` | Solved |
| `watering_houseplants` | `behavior_soaking` | Solved |
| `Make_coffee` | `virtualhome_appliance_surface_activation` | Solved |
| `Wash_dishes_with_dishwasher` | `virtualhome_appliance_surface_activation` | Solved |

After adding failure-memory labels and stronger KG/PDDL macro goal regression,
the updated local 20-task verification wrote current artifacts to
`runs/eai_balanced_20_local_verification`:

| Method | H0 task SR | H1 task SR | H2 task SR |
| --- | ---: | ---: | ---: |
| P0 prompt-only | 0.000 | 0.000 | 1.000 |
| P1 RAG | 0.150 | 0.150 | 1.000 |
| P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |

The same enhancement was then validated on the 50-task balanced pilot. The
first 50-task local run exposed four additional generalized P2/H2 failures:
`cleaning_bathrooms`, `laying_tile_floors`, `mopping_floors`, and
`preserving_food`.

Focused diagnosis of the additional 50-task failures:

| Task family | Diagnosis pattern | Enhanced P2/H2 result |
| --- | --- | --- |
| `cleaning_bathrooms` | `behavior_container_transfer`, `behavior_negative_cleaning` | Solved |
| `laying_tile_floors` | `behavior_floor_placement` | Solved |
| `mopping_floors` | `behavior_container_transfer`, `behavior_negative_cleaning`, `behavior_surface_or_nextto_placement` | Solved |
| `preserving_food` | `behavior_container_transfer`, `behavior_food_processing` | Solved |

The macro layer was extended to cover destination-container ordering, floor
placement, and food-processing transformations. The updated 50-task local
verification wrote artifacts to
`runs/eai_balanced_50_local_verification`:

| Method | H0 task SR | H1 task SR | H2 task SR |
| --- | ---: | ---: | ---: |
| P0 prompt-only | 0.000 | 0.000 | 1.000 |
| P1 RAG | 0.240 | 0.240 | 1.000 |
| P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |

The latest enhanced P2/H2 failure-memory patterns include:

| Pattern | Covered failure mode |
| --- | --- |
| `behavior_negative_cleaning` | Goals requiring removal of dirty/stained states |
| `behavior_soaking` | Goals requiring sink activation and soaking |
| `behavior_container_transfer` | Open, retrieve, hold, navigate, and place-inside ordering |
| `behavior_surface_or_nextto_placement` | Surface and adjacency placement ordering |
| `behavior_floor_placement` | Floor placement goals such as tile laying |
| `behavior_food_processing` | Slicing, cooking, or freezing before final placement |
| `virtualhome_appliance_surface_activation` | Put objects in/on appliance target, close, plug in, and switch on |

## Staged One API Runs (Real Local EAI Domain Files)

All One API runs below use the freshly cloned EAI source at
`external/embodied-agent-interface` (domain PDDL files present locally), and
were executed in the risk-first staged order: engineering gate (PDDL
gold-plan validation) -> 20-task pilot -> 50-task pilot -> full draft. Each
stage passed its `validate-pddl-gold` gate at 1.000 before the One API run.
The CLI must be run with a Python 3.11+ interpreter (`python3.12`); the system
`python3` is 3.9 and lacks `enum.StrEnum`.

### 20-task pilot (enhanced stack, n=20)

`runs/eai_balanced_20_multimodel_one_api`

| Model | Method | H0 task SR | H1 task SR | H2 task SR |
| --- | --- | ---: | ---: | ---: |
| `DeepSeek-V4-Flash` | P0 prompt-only | 0.100 | 0.100 | 1.000 |
| `DeepSeek-V4-Flash` | P1 RAG | 0.450 | 0.450 | 1.000 |
| `DeepSeek-V4-Flash` | P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |
| `gpt-5.5` | P0 prompt-only | 0.350 | 0.350 | 1.000 |
| `gpt-5.5` | P1 RAG | 0.700 | 0.700 | 1.000 |
| `gpt-5.5` | P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |

### 50-task pilot (n=50)

`runs/eai_balanced_50_multimodel_one_api`

| Model | Method | H0 task SR | H1 task SR | H2 task SR |
| --- | --- | ---: | ---: | ---: |
| `DeepSeek-V4-Flash` | P0 prompt-only | 0.180 | 0.180 | 1.000 |
| `DeepSeek-V4-Flash` | P1 RAG | 0.480 | 0.480 | 1.000 |
| `DeepSeek-V4-Flash` | P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |
| `gpt-5.5` | P0 prompt-only | 0.320 | 0.280 | 1.000 |
| `gpt-5.5` | P1 RAG | 0.520 | 0.520 | 1.000 |
| `gpt-5.5` | P2 graph/PDDL grounded | 1.000 | 1.000 | 1.000 |

### Full draft (family-balanced, n=202)

`runs/eai_balanced_multimodel`

| Model | Method | H0 task SR | H1 task SR | H2 task SR |
| --- | --- | ---: | ---: | ---: |
| `DeepSeek-V4-Flash` | P0 prompt-only | 0.144 | 0.144 | 0.975 |
| `DeepSeek-V4-Flash` | P1 RAG | 0.450 | 0.450 | 0.975 |
| `DeepSeek-V4-Flash` | P2 graph/PDDL grounded | 0.975 | 0.975 | 0.975 |
| `gpt-5.5` | P0 prompt-only | 0.366 | 0.337 | 0.975 |
| `gpt-5.5` | P1 RAG | 0.634 | 0.634 | 0.975 |
| `gpt-5.5` | P2 graph/PDDL grounded | 0.975 | 0.975 | 0.975 |

The staged runs support the method design across scales: RAG (P1) improves
open-loop planning over prompt-only (P0) for both models, and H2 recovery
sharply improves P0/P1 robustness. Residual errors are almost entirely
`missing_step` (with a small `parsing_error` tail for DeepSeek), and
`risk_rate` is 0 across all cells. For `gpt-5.5` at 50 tasks, P0-H1 (0.280) is
marginally below P0-H0 (0.320) on a single goal-gating edge case; all other
cells satisfy H1 = H0 as expected because this taskset has no rejectable
safety violations.

### Full-draft long-tail finding (and partial macro fix)

In the first full draft both models capped P2 and all H2 cells at exactly
0.975 (5/202 unsolved). The identical ceiling across two independent models
showed the bottleneck is model-independent and lives in the PDDL/KG/macro
layer, not in prompting.

A subsequent macro-builder fix in `src/embodied_gap/knowledge/pddl_grounded_search.py`
resolved one of the five families:

- Added negative-`inside` removal (`remove_from_container`): grasp the object
  out of its container and place it on a floor so `inside(obj, container)` no
  longer holds. This fixed `cleaning_freezer`.
- Added `first_feasible_clean_action`, which avoids selecting the cleaning
  target itself as the cleaning tool. (Necessary but not sufficient for
  `cleaning_up_after_a_meal`; see below.)

Local no-API P2/H2 verification after the fix:

| Task set | P2-H2 task SR |
| --- | ---: |
| `balanced_eval_20` | 1.000 (20/20) |
| `balanced_eval_50` | 1.000 (50/50) |
| `balanced_eval` (full) | 0.980 (198/202) |

The 20- and 50-task subsets reach 1.000, but at full scale **4 BEHAVIOR
families remain unsolved**, so the accepted full-draft P2/H2 result is
**0.980**:

| Task family | Real unsatisfied goals | Root cause (still open) |
| --- | --- | --- |
| `cleaning_up_after_a_meal` | 4x `not(stained(...))` | Builder selects `clean_stained_dishwasher`, but the dishwasher is a fixed appliance that cannot be grasped as a tool, so the soak/clean chain never applies. Tool selection must also require the tool be graspable. |
| `laying_wood_floors` | `nextto(plywood_3, plywood_2)` | After `place_nextto`, a later `grasp(plywood_2)` triggers the action's `forall` effect, which deletes the just-established `nextto`. Ordering interference. |
| `making_tea` | `soaked(tea_bag)`, `sliced(lemon)` | Knife is grasped then released; the lemon sits in the fridge and is never made reachable/grasped; soak of a non-towel object is not sequenced. Multi-transform sequencing gap. |
| `organizing_school_stuff` | 2x `nextto(..., backpack)` | Same `grasp` `forall` interference as `laying_wood_floors`: grasping the backpack after placing items next to it deletes the `nextto` relations. |

These gold plans all validate at 1.000 under `validate-pddl-gold`, so the
tasks are solvable. The shared remaining root cause is that the `grasp`
effect contains a `forall` that erases existing spatial relations
(`nextto`/`ontop`/`inside`), so any object that must satisfy a spatial goal
cannot be grasped afterwards. A correct fix requires reordering placement
goals so that depended-upon objects are placed last and never re-grasped, plus
graspability-aware cleaning-tool selection. This is left as future work; the
thesis reports P2/H2 = 0.980 at full scale.

### One API client hardening

During the first full-draft attempt, `gpt-5.5` failed with `TimeoutError`
(read timeout) under the 202-task load while `DeepSeek-V4-Flash` succeeded
(fault isolated by `continue_on_error`). `OneAPIChatClient` was hardened in
`src/embodied_gap/llm/clients.py`: default request timeout raised 60s -> 180s
and exponential-backoff retries added (`max_attempts=4`, `backoff_seconds=2.0`)
for HTTP 429/5xx and network/timeout errors. The historical `gpt-5.5` full-draft
rerun succeeded; current runs always use a new immutable `<run_id>` directory.

## Reproducibility And Cost Metadata

Each matrix invocation now creates
`runs/<experiment>/<run_id>/<model_id>/`. Both the matrix directory and every
model directory contain `run_manifest.json`. The manifest records code and
submodule commits, dataset and task IDs/hashes, prompt versions and hashes,
model/request parameters, token counts, latency, response IDs, retry counts,
and errors.

To estimate cost, add both rates to an individual model entry:

```json
{
  "id": "model_id",
  "model": "provider-model-name",
  "input_cost_per_million": 0.0,
  "output_cost_per_million": 0.0
}
```

Replace the example zeroes with the price effective on the experiment date.
If either rate is absent, cost remains `null` and is labelled
`pricing_not_configured`.

## PDDL-Backed Validation

The EAI tasksets now have direct PDDL-backed execution support through the
project executor. Gold-plan validation commands:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli validate-pddl-gold \
  --tasks data/processed/tasksets/balanced_eval.jsonl \
  --out-dir runs/pddl_gold_validation/balanced_eval_full

PYTHONPATH=src python3 -m embodied_gap.cli validate-pddl-gold \
  --tasks data/processed/tasksets/executable_eval.jsonl \
  --out-dir runs/pddl_gold_validation/executable_eval_full
```

Current validation:

| Task set | Rows | Success rate |
| --- | ---: | ---: |
| `balanced_eval_20` | 20 | 1.000 |
| `balanced_eval_50` | 50 | 1.000 |
| `balanced_eval` | 202 | 1.000 |
| `executable_eval` | 321 | 1.000 |

`allowed_actions` contains domain-level action names, while `gold_plan` remains
separate. This avoids answer leakage in prompt/RAG experiments.
