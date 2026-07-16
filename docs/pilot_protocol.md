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
