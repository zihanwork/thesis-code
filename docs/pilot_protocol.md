# Cost-controlled pilot protocol

The pilot stage selects methods on the 20-task development subset before any
larger or multi-model run. It never reads the frozen held-out task file.

## Gate 1: configuration preflight

Inspect every matrix before execution:

```bash
uv run --frozen python -m embodied_gap.cli inspect-model-matrix \
  --config configs/experiments/pilot_prompt_deepseek_20.json
```

The report must show `safe_for_development_selection: true`. Check the task,
model, record, and worst-case LLM-call counts before approving a paid run.

## Gate 2: DeepSeek development pilot

Run the prompt baseline pilot first. Then run lexical, BM25, and structured RAG
with the same tasks, model parameters, top-1 setting, and open-loop harness. The
only changed factor in the three RAG files is the retriever.

The planned budget is:

| Pilot | Tasks | Methods | Worst-case LLM calls |
|---|---:|---:|---:|
| Prompt baselines | 20 | 3 | 60 |
| Lexical RAG | 20 | 1 | 20 |
| BM25 RAG | 20 | 1 | 20 |
| Structured RAG | 20 | 1 | 20 |
| Total |  |  | 120 |

No paid run starts merely because a configuration exists. Record current model
pricing first; otherwise monetary cost remains explicitly unavailable.

## Gate 3: promotion

Choose the prompt and retriever candidates using task success, paired task-level
outcomes, token use, latency, and failure types. Only promoted candidates proceed
to top-k and field-profile ablations, followed by a GPT-5.5 confirmation run.
Do not tune rules, prompts, memory, or retrieval against held-out failures.

## Recovery mechanism pilot

After selecting the prompt baseline, compare open loop, local repair, plain LLM
reflection, error-specific LLM repair, frozen-memory LLM repair, and symbolic
PDDL recovery with `pilot_recovery_deepseek_20.json`. These are separate rows;
the legacy mixed `H2_full_recovery` is excluded. With one retry, the preflight
upper bound is 80 LLM calls for 120 execution records. The final Stage 10
resource-matched output cap of 2048 tokens is reused, and length-truncated calls
must be reported.

Only after the isolated comparison is interpretable, run
`pilot_recovery_combined_deepseek_20.json` for the combined method and its three
leave-one-component-out variants. Its upper bound is 100 LLM calls for 80
execution records. The frozen memory contains 357 development repairs generated
by a symbolic PDDL teacher; it must not be described as purely self-generated
LLM memory. Do not run this second pilot unless the isolated comparison provides
a concrete reason to test the combined mechanism.
