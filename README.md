# Bridging the Goal-to-Action Gap

Formal research framework for the thesis project:

**Bridging the Goal-to-Action Gap: Diagnosing and Improving LLM Embodied Planning**

The framework implements a two-layer experimental design inspired by *Embodied Agent Interface*:

- **Initial planners** answer how the first action sequence is generated.
- **Execution harnesses** answer how a generated plan is validated, executed, repaired, and recovered.

This avoids treating harness engineering as just another prompt/planner method.

## Experimental Design

### Initial Planners

| ID | Planner | Research Question |
| --- | --- | --- |
| B0 | Minimal Prompt Baseline | How far does an instruction/action-list prompt go? |
| P0 | Structured PDDL-Informed Prompt | What is gained from state, goal, objects, and action signatures? |
| P0-PE | Engineered Structured Prompt | What is gained from a fixed constraint checklist? |
| P1 | Retrieval-Augmented Planning | What additional gain comes from task-conditioned demonstrations? |
| P2 | Symbolic PDDL Reference | How does a model-independent symbolic planner perform? |

### Harness Modes

| ID | Harness | Research Question |
| --- | --- | --- |
| H0 | Open-loop Execution | What fails when the plan is executed directly? |
| H1 | Verifier-Gated Execution | How much does precondition/safety validation help without repair? |
| H2-Local | Local Recovery | What can deterministic local repair fix? |
| H2-LLM | LLM Feedback Replanning | What can the original model repair from explicit validator feedback? |
| H2-PDDL | Symbolic PDDL Recovery | What is gained from a symbolic fallback? |

The original 3x3 matrix is retained as a historical pilot. The final design
uses separate planning and recovery ablations so P2 and H2 cannot claim credit
for the same symbolic search. See `docs/revised_experiment_design.md`.

## Repository Structure

```text
configs/
  experiments/          Experiment matrix configs
  models/               Model/provider configs
  datasets/             Dataset adapter configs
data/
  raw/                  Raw benchmark exports
  processed/            Canonical JSONL tasks
  knowledge_graphs/     Graph facts and precondition/effect stores
  retrieval_corpus/     Demonstration and failure-memory corpora
src/embodied_gap/
  core/                 Task, state, goal, plan, violation, patch schemas
  datasets/             Dataset adapters
  llm/                  LLM clients, prompts, parsers, cache
  planners/             P0/P1/P2 initial planners
  execution/            Symbolic executor, validator, goal checker
  harness/              H0/H1/H2 controller and recovery policy
  repair/               Safety rejection, local patch, full replanning
  knowledge/            Retriever, action graph, affordance KB, failure memory
  evaluation/           Metrics, error taxonomy, statistical tests
  experiments/          Config, registry, runner, logger
  analysis/             Aggregation, table, and figure helpers
tests/                  Unit and integration tests
runs/                   Generated experiment artifacts
reports/                Thesis-ready tables and figures
notebooks/              Exploratory analysis
```

## Quick Start

Create the exact Python 3.12 environment recorded in `uv.lock`:

```bash
uv sync --locked
```

Run the offline historical sample matrix:

```bash
uv run --frozen embodied-gap run \
  --config configs/experiments/sample_matrix.json
```

Run tests:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Prepare clean EAI raw benchmark data:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli prepare-eai \
  --source-root external/embodied-agent-interface \
  --out-dir data/processed/eai_clean
```

Generated task metadata stores paths relative to the EAI checkout. If the
checkout is kept elsewhere, set `EAI_SOURCE_ROOT=/path/to/embodied-agent-interface`
or pass `--source-root`; processed JSONL files do not need to be rewritten.
The canonical `data/processed/tasksets/*.jsonl` files used by committed
experiment configs are version-controlled so a fresh clone can run them
immediately. Intermediate `data/processed/eai_clean/` conversion outputs remain
ignored and can be regenerated with `prepare-eai`.

The EAI import reads only source benchmark resources, not previous model outputs
or diagnostics. See `docs/data_pipeline.md` for the data policy and current
import summary.

## One API

Create a local `.env` file with:

```bash
ONE_API_KEY=your-key
ONE_API_BASE_URL=https://your-one-api-domain/v1
ONE_API_MODEL=your-model
```

Check connectivity:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli check-one-api
```

List the model IDs exposed to the configured account, or smoke-test a specific
model without changing `.env`:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli list-one-api-models
PYTHONPATH=src python3 -m embodied_gap.cli check-one-api --model "GLM-5-Turbo"
```

Run the matrix with One API-backed P0/P1 planners:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run \
  --config configs/experiments/one_api_matrix.json
```

Run the sample multi-model matrix:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/sample_multimodel_one_api.json
```

Run the cost-controlled real EAI smoke matrix:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/eai_smoke_multimodel_one_api.json
```

Run the 20-task real EAI balanced pilot after local validation:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/eai_balanced_20_multimodel_one_api.json
```

See `docs/experiment_matrix.md` for the historical pilot results. The revised
planning/recovery/generalization design is documented in
`docs/revised_experiment_design.md`.

Inspect and run the preregistered model-generalization pilot, then export its
auditable summary:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli inspect-model-matrix \
  --config configs/experiments/eai_model_generalization_dev20.json
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/eai_model_generalization_dev20.json
PYTHONPATH=src python3 -m embodied_gap.cli summarize-model-generalization \
  --run-dir runs/eai_model_generalization_dev20/<run-id> \
  --out docs/model_generalization_evidence.json
```

The completed Stage 10 development evidence and its claim limits are documented
in `docs/model_generalization_protocol.md`.

The one-shot final held-out cells, 1,666-call worst-case budget, Git-tag gate,
artifact hashes, and stopping rules are frozen in
`docs/final_experiment_protocol.md`. Before any final run, execute:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli verify-final-protocol
```

Before a paid model-matrix run, use `inspect-model-matrix` to validate paths and
review the task, record, and worst-case LLM-call counts. The staged 20-task
development procedure is documented in `docs/pilot_protocol.md`.

New runs also generate `analysis.json` with confidence intervals, task-paired
tests, stratified outcomes, failure counts, and cost/search summaries. See
`docs/statistical_analysis.md`.

The verified public benchmark comparison and literature positioning are in
`docs/benchmark_literature_review.md`; machine-readable evidence and the strict
custom-versus-official comparison boundary are in `docs/benchmark_evidence.json`.

The data-use boundary is frozen in `docs/data_split_protocol.md`: 202 tasks are
development-only and 119 VirtualHome tasks are reserved for one-shot local
held-out evaluation.

## Experiment Artifacts

Every invocation creates a new immutable directory below the configured
`output_dir`; an earlier run is never deleted or overwritten. A normal run is
stored as `output_dir/<run_id>/`. A model matrix uses
`output_dir/<run_id>/<model_id>/` and keeps its matrix-level summary at the
`<run_id>` level. `run_index.jsonl` provides a compact history of completed
runs.

Each run writes:

- `run_manifest.json` - run ID and timestamps; Git commit, dirty state, and
  pinned submodule commit; Python and `uv.lock` fingerprints; dataset hash and
  task IDs; prompt file hash and prompt version; model parameters; token usage,
  latency, estimated cost, response IDs, and failures.
- `config.json` - exact experiment configuration.
- `runs.jsonl` - initial plans, final plans, traces, violations, patches.
- `metrics.jsonl` - one evaluation record per task/planner/harness cell.
- `summary.json` - aggregate task success, execution success, safety, risk, rejection, patch, and error metrics.

Per-call cost is estimated only when both `input_cost_per_million` and
`output_cost_per_million` are configured for that model. Otherwise the
manifest records `pricing_not_configured` rather than inventing a price. API
keys and full prompt text are never written; prompts are identified by a
SHA-256 fingerprint.

## Canonical Task Format

Tasks are stored as JSONL with:

- `instruction`
- `initial_facts`
- `goal_facts`
- `allowed_actions`
- `action_model`
- optional `gold_plan`, `slots`, `tags`, `safety_rules`

This makes it possible to adapt EAI, ALFRED, ET-Plan-Bench, SafeAgentBench, or custom symbolic tasks into one shared evaluation pipeline.

## Current Implementation Status

Implemented:

- Canonical task/plan/state/violation/patch schemas.
- Minimal, structured, engineered, RAG, and symbolic-reference planners.
- Open-loop, verifier, local, LLM-feedback, and PDDL recovery modes.
- Clean EAI raw-resource import for VirtualHome and BEHAVIOR PDDL tasks.
- Thesis task set builder for RAG train, full eval, executable eval, balanced eval, and 20/50-task balanced pilots.
- Multi-model One API experiment runner with per-model fault isolation.
- External retrieval example files for P1 RAG experiments.
- PDDL-backed execution and gold-plan validation for clean EAI tasksets.
- Model-independent P2 symbolic PDDL planning.
- Isolated local, LLM-feedback, and symbolic PDDL recovery paths.
- Error-specific LLM repair, frozen failure memory, and combined leave-one-out modes.
- Failure-memory labels for recurring macro-recoverable failures.
- KG/macro goal-regression coverage for cleaning, soaking, container transfer,
  surface/next-to/floor placement, food processing, and VirtualHome appliance
  activation tasks.
- Non-leaky PDDL action signatures and object-type candidates in P0/P1 prompts.
- Bounded cached fallback search for scalable PDDL-grounded recovery.
- Robust parsing for common fenced JSON action-list outputs.
- EAI-style symbolic execution errors: hallucination, missing step, wrong order, affordance error, additional step, safety violation, goal unsatisfied.
- JSONL experiment logging and summary generation.
- Immutable run directories with provenance manifests, task/prompt hashes,
  model parameters, token/latency telemetry, and optional cost estimates.
- Locked Python 3.12 environment and CI verification.
- Local balanced 20-task and 50-task verification after failure-memory and
  KG/macro enhancement.
- Unit/integration tests for the sample matrix.
- One API-compatible LLM client.

Completed research infrastructure now also includes:

- Cost-controlled recovery, memory, prompt, and RAG pilot configurations.
- Wilson confidence intervals, paired tests, cost aggregation, and failure analysis.
- Automatic real-run export to the pinned official VirtualHome evaluator, with
  ambiguity/upstream exclusions recorded separately from benchmark claims.
- A 30-task frozen controlled safety set and dedicated H0/H1/H2-Local metrics;
  see `docs/safety_evaluation.md`.

The model-family expansion and one-shot final frozen matrix are complete. The
final local results, paired statistics, resource telemetry, and official
VirtualHome compatibility diagnostic are documented in
`docs/final_results.md` and `docs/final_results_evidence.json`.

The remaining external-scope limitations are not hidden: the project has no
official BEHAVIOR simulator score without the separately licensed iGibson
assets, and it does not implement all eight official challenge output slots.
Fine-tuning remains an optional control rather than a prerequisite.
