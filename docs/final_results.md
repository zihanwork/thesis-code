# Final held-out results

## Scope and provenance

The final protocol ran once on 119 frozen VirtualHome tasks at clean commit
`50095ae9d48b6b5de9e172a40fa64966af89b9e5`, tagged
`final-protocol-v1`. All frozen artifact hashes still match after execution.
The four matrices contain 2,023 task-method records. These are local held-out
results under the custom PDDL final-state evaluator; they are not official EAI
challenge scores and do not estimate BEHAVIOR performance.

## Main task-success results

Rates are task success with Wilson 95% confidence intervals.

| Model | Method | Success | Rate | 95% CI |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V4-Flash | B0 Minimal Prompt + H0 | 0/119 | 0.0% | 0.0–3.1% |
| DeepSeek-V4-Flash | P0 Structured Prompt + H0 | 30/119 | 25.2% | 18.3–33.7% |
| DeepSeek-V4-Flash | P0 Engineered Prompt + H0 | 29/119 | 24.4% | 17.5–32.8% |
| DeepSeek-V4-Flash | P1 RAG + H0 | 107/119 | 89.9% | 83.2–94.1% |
| DeepSeek-V4-Flash | P1 RAG + LLM Reflection | 117/119 | 98.3% | 94.1–99.5% |
| DeepSeek-V4-Flash | P1 RAG + Memory | 117/119 | 98.3% | 94.1–99.5% |
| DeepSeek-V4-Flash | P1 RAG + PDDL Recovery | 119/119 | 100.0% | 96.9–100.0% |
| gpt-5.5 | B0 Minimal Prompt + H0 | 0/119 | 0.0% | 0.0–3.1% |
| gpt-5.5 | P0 Structured Prompt + H0 | 57/119 | 47.9% | 39.1–56.8% |
| gpt-5.5 | P0 Engineered Prompt + H0 | 56/119 | 47.1% | 38.3–56.0% |
| gpt-5.5 | P1 RAG + H0 | 112/119 | 94.1% | 88.4–97.1% |
| gpt-5.5 | P1 RAG + LLM Reflection | 119/119 | 100.0% | 96.9–100.0% |
| gpt-5.5 | P1 RAG + Memory | 119/119 | 100.0% | 96.9–100.0% |
| gpt-5.5 | P1 RAG + PDDL Recovery | 119/119 | 100.0% | 96.9–100.0% |
| GLM-5-Turbo | P0 Engineered Prompt + H0 | 25/119 | 21.0% | 14.7–29.2% |
| GLM-5-Turbo | P1 RAG + H0 | 105/119 | 88.2% | 81.2–92.9% |
| Model-independent | P2 Symbolic PDDL + H0 | 119/119 | 100.0% | 96.9–100.0% |

P2 is a model-independent symbolic reference and is reported once. It must not be
described as an LLM or as evidence that augmentation made a weaker model equal
to a stronger model.

## What the ablations show

Prompt engineering did not improve the structured baseline. It was lower by
0.8 percentage points for both primary models, with exact McNemar
`p = 1.0` in both comparisons. Therefore the engineered prompt is a useful
control, not a positive contribution.

RAG produced the largest LLM-side improvement:

| Model | P0 Engineered → P1 RAG | Paired uplift 95% CI | McNemar p |
| --- | ---: | ---: | ---: |
| DeepSeek-V4-Flash | +65.5 pp | +56.3 to +73.9 pp | 1.34e-22 |
| gpt-5.5 | +47.1 pp | +38.7 to +56.3 pp | 4.09e-16 |
| GLM-5-Turbo | +67.2 pp | +57.1 to +76.5 pp | 2.74e-21 |

Plain LLM Reflection then recovered 10/12 DeepSeek failures and all 7 GPT
failures. This raised DeepSeek from 89.9% to 98.3% (+8.4 pp, paired 95% CI
+4.2 to +13.4 pp, `p = .00195`) and GPT from 94.1% to 100% (+5.9 pp,
paired 95% CI +2.5 to +10.1 pp, `p = .0156`).

The frozen Memory condition had exactly the same paired outcomes as Reflection
for both models (`p = 1.0`) while using slightly more tokens. It therefore
does not support an incremental memory benefit. PDDL Recovery reached 100% for
both models but remains a separately labelled symbolic fallback. For DeepSeek,
its +1.7 pp advantage over Reflection was not significant (`p = .5`).

## Stratified and failure findings

The held-out set contains 77 easy, 39 medium, and only 3 hard tasks. P1/H0
success was 70/77, 36/39, and 1/3 for DeepSeek; 72/77, 37/39, and 3/3 for GPT;
and 70/77, 34/39, and 1/3 for GLM. The hard stratum is too small for a stable
model comparison and must be reported with this warning.

At the task-family level, DeepSeek P1/H0 was weakest on
`Wash_dishes_with_dishwasher` (2/4), `Drink` (7/10), and `Watch_TV` (5/7).
Its two failures remaining after both Reflection and Memory were the two hard
dishwasher tasks; PDDL Recovery solved them. GPT P1/H0 was weakest on `Drink`
and `Read_book` (both 8/10), and Reflection recovered every failure. GLM P1/H0
was weakest on `Drink` and dishwasher tasks (both 50%) and `Watch_TV` (4/7).
Complete task-family tables remain in each immutable `analysis.json`.

The dominant open-loop failure class was `missing_step`: 8 for DeepSeek P1,
8 for GPT P1, and 10 for GLM P1. GLM also had two parsing failures, and
DeepSeek had one action-argument-count failure. No rule, prompt, retrieval
example, or memory entry was changed after inspecting these held-out failures.

## Calls, tokens, latency, and symbolic search

| Matrix/model | Actual calls | Total tokens | Latency | Length truncations |
| --- | ---: | ---: | ---: | ---: |
| Planning / DeepSeek-V4-Flash | 357 | 645,910 | 2,146.4 s | 2 |
| Planning / gpt-5.5 | 357 | 551,563 | 2,464.3 s | 0 |
| Recovery / DeepSeek-V4-Flash | 143 | 364,665 | 824.8 s | 5 |
| Recovery / gpt-5.5 | 135 | 288,131 | 990.2 s | 0 |
| Generalization / GLM-5-Turbo | 238 | 581,937 | 4,361.8 s | 3 |
| Total | 1,230 | 2,432,206 | 10,787.4 s | 10 |

All 1,230 calls ultimately succeeded. GLM required 240 transport attempts for
238 calls. Monetary cost is intentionally unreported because trustworthy One
API per-model pricing was not configured. P2 required no LLM calls, explored
303 symbolic states, and used 0.411 seconds of search time. DeepSeek and GPT
PDDL Recovery explored 45 and 2 states respectively.

## Official VirtualHome compatibility diagnostic

The best non-symbolic method, gpt-5.5 P1 + Reflection, was exported to the
pinned official VirtualHome action-sequencing evaluator. Strict full export
reported an incomplete conversion safely: 83/119 tasks were mechanically convertible, while 30 used
`PLUGIN/PLUGOUT` actions unsupported by the pinned evaluator and 6 had
ambiguous official object IDs. No ID was guessed and no action was dropped.

On the 83-task compatible subset, the official evaluator reported 83/83
executable trajectories but only 43/83 complete task successes (51.8%). Goal
scores were state 67/69 (97.1%), relation 64/69 (92.8%), action 1/38 (2.6%),
and combined 132/176 (75.0%). The low action-goal score demonstrates that
custom PDDL final-state success is not equivalent to the official action/LTL
criterion. Because the 83 tasks are a compatibility-selected subset, this is
an integration diagnostic, not a leaderboard-comparable score.

The machine-readable results and exact artifact hashes are in
`docs/final_results_evidence.json` and
`docs/final_official_virtualhome_evidence.json`.
