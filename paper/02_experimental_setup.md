# 3. Experimental Setup

This chapter fixes the moving parts of the study so that the empirical
chapters that follow can be read as comparisons under a common protocol.
We describe the dataset and task definitions inherited from EAI
(Section 3.1), the metrics we report and how they map onto the research
questions (Section 3.2), the model inventory (Section 3.3), the prompt
variants we test against each baseline (Section 3.4), and the
end-to-end evaluation pipeline that turns model outputs into the
diagnostic tables and figures used in Chapters 4–6 (Section 3.5).

## 3.1 Dataset and Tasks

We use the VirtualHome subset of the Embodied Agent Interface (EAI)
benchmark.[^eai] EAI fixes a set of household tasks (e.g.
*Wash clothes*, *Turn on light*, *Read book*) and, for each task,
provides a scene graph, a natural-language description, and a gold
program in VirtualHome's symbolic format. We touch two of EAI's four
modules:

[^eai]: Li et al., *Embodied Agent Interface: Benchmarking LLMs for
Embodied Decision Making*, NeurIPS Datasets and Benchmarks 2024.

- **Goal interpretation.** Given the task description and scene, the
  model must emit a JSON object with three keys — `node goals`,
  `edge goals`, `action goals` — describing target states, spatial
  relations, and required actions.
- **Action sequencing.** Given the same input, the model must emit a
  sequence of VirtualHome actions in the compact form
  `{"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}`.
  Every action must use objects whose name and numeric id appear in
  the scene graph; preconditions are enforced by the simulator.

We work with the 305 task instances that EAI defines for VirtualHome,
which is the size of every per-model row in Chapter 4. A subset of 342
joined goal-action cases is used in Chapter 5 to study the
goal-to-action gap; the difference comes from goal-interpretation
items that share a `file_id` with multiple action-sequencing programs.

The benchmark deliberately excludes pixel-level perception. The scene
is given symbolically, and the model output is graded by a symbolic
simulator. This keeps the failure modes we observe attributable to the
LLM rather than to a perception module.

## 3.2 Metrics

For **goal interpretation** we report the four EAI F1 scores:
`node_f1`, `edge_f1`, `action_f1`, and the macro `all_f1` that we use
as the primary ranking metric. F1 is computed against the gold goal
specification with EAI's matcher.

For **action sequencing** we report:

- `task_success_rate` — the program executes *and* the resulting
  state graph satisfies all node, edge and action goals; this is the
  primary metric.
- `execution_success_rate` — the program executes without violating
  preconditions, regardless of whether the goal is satisfied.
- Seven fine-grained error rates (percentage of programs):
  `parsing`, `hallucination`, `predicate_argument_number`,
  `wrong_order`, `missing_step`, `affordance_error`,
  `additional_step`. These are read directly from EAI's per-program
  error log.

This split lets us answer the four research questions cleanly:

| RQ | Primary metric(s) | Reported in |
| --- | --- | --- |
| RQ1 (goal vs action) | `all_f1` and `task_success_rate` | §4 scatter |
| RQ2 (failure mix) | seven fine-grained error rates | §5 |
| RQ3 (family effects) | `task_success_rate` averaged by family | §4 |
| RQ4 (interventions) | `task_success_rate` and `all_f1` deltas vs baseline | §6 |

## 3.3 Model Inventory

The inventory we evaluate combines two sources, both materialised in
[`output/diagnostics/multimodel_existing_inventory.csv`](../output/diagnostics/multimodel_existing_inventory.csv):

1. **Existing EAI/HELM snapshots.** Seventeen models with completed
   runs across at least one of the two modules: OpenAI
   (`gpt-3.5-turbo-0125`, `gpt-4-turbo-2024-04-09`, `gpt-4o-2024-05-13`,
   `o1-mini-2024-09-12`, `o1-preview-2024-09-12`), Anthropic
   (`claude-3-haiku-20240307`, `claude-3-sonnet-2024022`,
   `claude-3-opus-20240229`, `claude-3-5-sonnet-20240620`), Google
   (`gemini-1.0-pro`, `gemini-1.5-flash-preview-0514`,
   `gemini-1.5-pro-preview-0409`), Meta (`llama-3-8b-chat`,
   `llama-3-70b-chat`), Mistral (`mistral-large-2402`,
   `mixtral-8x22b-instruct-v0.1`) and Cohere (`cohere-command-r`,
   `cohere-command-r-plus`). For these models we report the same
   columns as EAI and treat the snapshots as the cross-family backbone
   of Chapters 4 and 5. We also keep `gold_oracle` as an
   upper-bound / pipeline sanity-check reference, not as a model
   prediction.
2. **New runs from the improvement pipeline.** For Chapter 6 we re-run
   one or two main models (default `gpt-4o-mini` from OpenAI, plus an
   optional cross-family control such as `claude-3-5-sonnet`) under
   each prompt variant and self-check setting. Outputs land in
   `output/improvement_run/helm_output` and are evaluated with the
   same EAI runner.

Whenever the same model appears in multiple result roots
(`output`, `output_norm_all`, `output_single_norm`), we deduplicate
by preferring `output_norm_all` over `output_single_norm` over
`output`, mirroring the convention in
`prepare_multimodel_experiment_materials.py`.

## 3.4 Prompt Variants

All variants are defined programmatically in
[`analysis/prompt_variants.py`](../analysis/prompt_variants.py); the
descriptions below match the `description` field of each
`PromptVariant`.

For **action sequencing**:

- `baseline` — minimal output-only system prompt.
- `format_constraints` — strict JSON format rules (uppercase action
  names, alternating `name,id` arguments, no markdown).
- `few_shot_valid_actions` — `format_constraints` plus three legal
  VirtualHome examples covering single-object, sit, and put-in
  patterns.
- `plan_then_ground` — instructs the model to plan internally first,
  then emit only the JSON.
- `self_check_rewrite` — two-pass: produce a draft, then revise it
  with an executable-checks critique. The critique step targets
  precondition omissions and hallucinated ids.

For **goal interpretation**:

- `baseline` — JSON-only system prompt.
- `schema_constrained` — enumerates the legal node-state and
  edge-relation values.
- `few_shot` — schema plus a worked EAI example.
- `decompose_then_merge` — three internal stages
  (states → relations → actions) merged into one JSON object.

Each variant is run at `temperature=0`, `max_tokens=2048`, with the
same prompt source as the existing snapshots, so that any change in
metrics is attributable to the variant rather than to sampling noise.

## 3.5 Evaluation Pipeline

[`scripts/run_improvement_pipeline.sh`](../scripts/run_improvement_pipeline.sh)
chains six steps:

1. **Prompt generation.** EAI's runner produces
   `helm_prompt.json` for each module if not already present.
2. **Multi-vendor inference.**
   [`analysis/generate_outputs.py`](../analysis/generate_outputs.py)
   dispatches to OpenAI, Anthropic, Gemini, OpenAI-compatible, or a
   `dry_run` provider used for end-to-end smoke tests.
3. **Self-check rewrite (optional).**
   [`analysis/self_check_loop.py`](../analysis/self_check_loop.py)
   reads EAI's `error_info.json` for the baseline run, identifies
   failing rows, and asks the same model to revise them under a
   critique prompt.
4. **Normalisation.**
   [`analysis/normalize_action_outputs.py`](../analysis/normalize_action_outputs.py)
   converts free-form action strings into the canonical `name/id`
   pairs that EAI's matcher consumes.
5. **EAI evaluation.** `conda run -n eai-eval eai-eval` writes per-task
   error logs and a `summary.json` per model.
6. **Materials and figures.**
   [`analysis/prepare_multimodel_experiment_materials.py`](../analysis/prepare_multimodel_experiment_materials.py)
   regenerates the inventory CSV, the failure profile table, the
   ablation tables, and the seven SVG figures used in Chapters 4–6.

Reproducibility. The pipeline supports a `dry_run` provider that
returns deterministic stub outputs, which we use to verify that all
six steps connect end-to-end without spending on API calls. For
example, the smoke run reported in
[`output/diagnostics/progress_report.md`](../output/diagnostics/progress_report.md)
exercised four action variants and four goal variants, plus the
self-check loop, on a synthetic three-task prompt set.

Switching to a real provider only requires environment variables
(`PROVIDER`, `API_MODEL`, `MODEL_NAME`, `RUN_EVAL`, etc.). All
intermediate artefacts — prompts, raw outputs, normalised outputs,
self-check reports, evaluation logs, summary tables, and figures —
are versioned under `output/improvement_run` and `output/diagnostics`,
so a reviewer can audit any column of any chapter back to the run
that produced it.
