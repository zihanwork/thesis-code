# Final Experiment Protocol v1

## Scope

The local final set contains 119 frozen VirtualHome tasks. It is a one-shot
local held-out evaluation, not an official EAI challenge score and not a
BEHAVIOR generalization estimate. The official evaluator results must be
reported separately from the custom PDDL evaluator.

The authoritative machine-readable protocol is
`configs/experiments/final_protocol_v1.json`. The protocol is valid only when:

- every frozen artifact hash matches;
- the worktree is clean;
- the current commit is tagged `final-protocol-v1`;
- no final output root already contains `run_index.jsonl`.

Verify all gates before the first final run:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli verify-final-protocol
```

## Frozen cells and budget

| Experiment | Models | Cells | Records | Worst-case LLM calls |
|---|---|---|---:|---:|
| Planning | DeepSeek-V4-Flash, gpt-5.5 | B0/H0, P0-structured/H0, P0-engineered/H0 | 714 | 714 |
| Recovery | DeepSeek-V4-Flash, gpt-5.5 | P1 with H0, Reflection, Memory, PDDL recovery | 952 | 714 |
| Generalization | GLM-5-Turbo | P0-engineered/H0, P1/H0 | 238 | 238 |
| Symbolic reference | Model-independent | P2-symbolic/H0 | 119 | 0 |
| Total |  |  | 2,023 | 1,666 |

P1/H0 for the two primary models is supplied by the Recovery matrix, avoiding a
duplicate P1 planning call in the Planning matrix. The Recovery matrix reuses
each P1 initial plan across all four harness modes.

The fixed decoding policy is temperature 0, maximum output 2048 tokens, timeout
180 seconds, and at most two transport attempts. Length-truncated calls are a
reported resource outcome and do not trigger selective reruns. Monetary cost
cannot be estimated until trustworthy One API pricing is configured; calls,
tokens, latency, and truncation remain mandatory.

## Method selection

- GLM-5-Turbo is the only additional family because it passed the realistic
  canary and development pilot.
- Plain Reflection is retained as the principal LLM recovery mechanism.
- Symbolic-teacher Memory is retained as a planned contrast despite being
  slightly worse and more expensive than Reflection in development.
- PDDL recovery is retained only as a separately labelled symbolic reference.
- H2 Local and H2 Error-specific remain development ablations.
- Combined Harness modes are not run because the isolated development evidence
  did not justify them.
- H1 safety claims come from the frozen safety benchmark, not from these normal
  household tasks.

See `docs/model_generalization_protocol.md` and
`docs/recovery_pilot_protocol.md` for the selection evidence.

## One-shot execution order

After the tag gate passes and the API-call budget is explicitly approved, run
the four configurations without inspecting per-task outcomes between them:

```bash
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/final_planning_heldout_v1.json
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/final_recovery_heldout_v1.json
PYTHONPATH=src python3 -m embodied_gap.cli run-model-matrix \
  --config configs/experiments/final_generalization_heldout_v1.json
PYTHONPATH=src python3 -m embodied_gap.cli run \
  --config configs/experiments/final_symbolic_heldout_v1.json
```

Do not change code, prompts, retrieval, memory, models, decoding, or statistics
after the first command starts. Do not selectively rerun failed tasks. If an
external service interruption prevents a complete matrix, preserve the failed
run and create an explicit protocol amendment before any rerun.

## Reporting

Report Wilson 95% intervals, exact paired McNemar tests, paired bootstrap uplift
intervals, VirtualHome-only scope, difficulty/task-family/failure strata, calls,
tokens, latency, length truncations, repair counts, PDDL explored states, and
PDDL search time. Never add held-out failures to RAG or failure memory.
