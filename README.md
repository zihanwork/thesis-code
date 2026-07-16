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
| P0 | Prompt-only Planning | How far can structured prompting go without external knowledge? |
| P1 | Retrieval-Augmented Planning | Can retrieved demonstrations reduce missing-step and task-prior errors? |
| P2 | Graph-Grounded Planning | Can object-action-state graphs reduce hallucination and affordance errors? |

### Harness Modes

| ID | Harness | Research Question |
| --- | --- | --- |
| H0 | Open-loop Execution | What fails when the plan is executed directly? |
| H1 | Verifier-Gated Execution | How much does precondition/safety validation help without repair? |
| H2 | Full Harness Recovery | How much do validation, local patching, graph replanning, and rejection improve robustness? |

The full experiment matrix is:

| Planner / Harness | H0 Open-loop | H1 Verifier | H2 Full Recovery |
| --- | --- | --- | --- |
| P0 Prompt-only | P0-H0 | P0-H1 | P0-H2 |
| P1 RAG | P1-H0 | P1-H1 | P1-H2 |
| P2 Graph-grounded | P2-H0 | P2-H1 | P2-H2 |

The expected primary method is **P2-H2: Graph-Grounded Planning with Full Harness Recovery**.

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

Run the sample 3x3 matrix:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run \
  --config configs/experiments/sample_matrix.json
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
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

See `docs/experiment_matrix.md` for the model matrix and current smoke-run
status.

## Experiment Artifacts

Each run writes:

- `config.json` - exact experiment configuration.
- `runs.jsonl` - initial plans, final plans, traces, violations, patches.
- `metrics.jsonl` - one evaluation record per task/planner/harness cell.
- `summary.json` - aggregate task success, execution success, safety, risk, rejection, patch, and error metrics.

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
- P0/P1/P2 initial planners.
- H0/H1/H2 harness modes.
- Clean EAI raw-resource import for VirtualHome and BEHAVIOR PDDL tasks.
- Thesis task set builder for RAG train, full eval, executable eval, balanced eval, and 20/50-task balanced pilots.
- Multi-model One API experiment runner with per-model fault isolation.
- External retrieval example files for P1 RAG experiments.
- PDDL-backed execution and gold-plan validation for clean EAI tasksets.
- P2 PDDL-grounded macro planning for real EAI smoke tasks.
- H2 PDDL-grounded full recovery for failed P0/P1 real EAI smoke plans.
- Failure-memory labels for recurring macro-recoverable failures.
- KG/macro goal-regression coverage for cleaning, soaking, container transfer,
  surface/next-to/floor placement, food processing, and VirtualHome appliance
  activation tasks.
- Non-leaky PDDL action signatures and object-type candidates in P0/P1 prompts.
- Bounded cached fallback search for scalable PDDL-grounded recovery.
- Robust parsing for common fenced JSON action-list outputs.
- EAI-style symbolic execution errors: hallucination, missing step, wrong order, affordance error, additional step, safety violation, goal unsatisfied.
- JSONL experiment logging and summary generation.
- Local balanced 20-task and 50-task verification after failure-memory and
  KG/macro enhancement.
- Unit/integration tests for the sample matrix.
- One API-compatible LLM client.

Next planned integrations:

- Resolve the 4 remaining P2/H2 long-tail BEHAVIOR families
  (`cleaning_up_after_a_meal`, `laying_wood_floors`, `making_tea`,
  `organizing_school_stuff`) that cap full-draft P2/H2 at 0.980. Shared root
  cause: the `grasp` `forall` effect erases existing spatial relations, plus
  graspability-aware cleaning-tool selection. (`cleaning_freezer` is already
  fixed via negative-`inside` removal.)
- Persistent retrieval corpus and failure-memory export/import.
- Domain KG export/import.
- Thesis-ready plotting scripts.
