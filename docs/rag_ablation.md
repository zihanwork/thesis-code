# RAG Ablation Protocol

## Retrieval Methods

The RAG planner exposes three deterministic retrieval methods:

- `lexical`: Jaccard similarity over the selected fields.
- `bm25`: corpus-level BM25 with the monotonic bounded transform `s/(s+1)`.
- `structured`: weighted field-level Jaccard with instruction 0.40, goal 0.30,
  initial state 0.20, and action schema 0.10.

All reported retrieval scores are in `[0, 1]`. The former shared-slot and tag
bonuses were removed because they made scores exceed 1 and rewarded fields that
were common to nearly every task.

## Field Profiles

- `instruction`
- `instruction_goal`
- `instruction_state_goal`
- `instruction_state_goal_schema`

The schema field contains allowed action signatures/action names. Gold plans are
never included in the query representation. Retrieved training demonstrations
may include their gold plan in the prompt.

## Top-K

The planner supports top-1, top-3, and top-5. Every retrieved example is placed
in the prompt, and every run records IDs, normalized scores, component scores,
method, field profile, top-k, and minimum score.

## Staged Selection

Use only `balanced_eval_20.jsonl` during initial selection:

1. Compare lexical, BM25, and structured with full fields and top-1.
2. Select the best method by paired task success, not aggregate score alone.
3. Compare top-1, top-3, and top-5 for that method.
4. Compare the four field profiles at the selected top-k.
5. Confirm the selected configuration on the 202-task development set.

Do not use `heldout_virtualhome_119.jsonl` for retriever selection.

## Required Reporting

- Task success and execution success.
- VirtualHome and BEHAVIOR separately.
- Retrieval score distribution and selected-example IDs.
- Prompt/token growth for top-k.
- Paired confidence interval and McNemar test against no-RAG P0-PE.
- Retrieval failure examples, especially BEHAVIOR cases.
