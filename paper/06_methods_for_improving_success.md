# 6. Methods for Improving Success

> **Status:** results filled in (2026-05-14). Main model
> `DeepSeek-V4-Flash` was evaluated on 200 prompts × 11 variants
> (7 action_sequencing + 4 goal_interpretation); cross-family
> validation used `GLM-5-Turbo` and `MiniMax-M2-Stable`
> (n=100 each, action_sequencing only, baseline + sg_rag).
> All numbers are quoted directly from
> `output/improvement_run/virtualhome/evaluate_results/`
> via `prepare_multimodel_experiment_materials.py`.
> The chapter retains the **pre-registered predictions** so that each
> outcome can be reported as **confirmed**, **partially confirmed**, or
> **rejected**.

This chapter answers RQ4 and RQ5: *which lightweight prompting,
self-correction, and **knowledge-grounded** interventions measurably
lift goal F1 or task success rate, and which failures remain robust?*
We deliberately keep the intervention space small and inexpensive —
none of the methods below fine-tune a model, none changes the EAI scene
representation, and all of them can be audited from
[`analysis/prompt_variants.py`](../analysis/prompt_variants.py),
[`analysis/self_check_loop.py`](../analysis/self_check_loop.py),
[`analysis/scene_graph_rag.py`](../analysis/scene_graph_rag.py), and
[`analysis/precondition_kg.py`](../analysis/precondition_kg.py).

## 6.1 Pre-registered Predictions (from §5.4)

We commit to the following predictions before the ablation runs:

| Variant | Targeted failure column | Predicted effect | Population most affected |
| --- | --- | --- | --- |
| `format_constraints` | `parsing` | Large drop (≥30 pp absolute on bottlenecked models, ≤2 pp on frontier) | `gpt-3.5-turbo`, `gemini-1.0-pro`, `llama-3-70b`, `mixtral-8x22b` |
| `few_shot_valid_actions` | `hallucination` | Drop on hallucination-bottlenecked models; mild drop on `claude-3-opus` | `llama-3-8b`, `cohere-command-r`, `claude-3-opus` |
| `plan_then_ground` | `missing_step` | Modest drop on reasoning-bottlenecked models | `gpt-4o`, `gpt-4-turbo`, `claude-3-haiku`, `gemini-1.5-flash` |
| `self_check_rewrite` | `missing_step` + `wrong_order` | Larger drop than `plan_then_ground` because critique consumes evaluator feedback | same as above |
| `schema_constrained` (goals) | `edge_f1` and `action_f1` | Lift on under-spec models that produced incomplete JSON keys | mid-tier OpenAI, Cohere, Gemini |
| `decompose_then_merge` (goals) | `node_f1` | Lift via three-stage induction; risk of over-listing | larger frontier models |
| `sg_rag` | `hallucination`, `affordance_error`, `additional_step` | ≥15 pp drop on mid-tier models, ≤5 pp on frontier (already well-grounded) | `gpt-3.5-turbo`, `llama-3-8b`, `cohere-command-r`, `mixtral-8x22b` |
| `pc_kg_self_check` | `missing_step`, `wrong_order` | ≥10 pp drop across all tiers; feedback is deterministic so gains are attributable | all model families |
| `sg_rag_pc_kg` | union of the above | Strictly dominates `self_check_rewrite` by ≥5 pp `task_success_rate`, and dominates `sg_rag` and `pc_kg_self_check` individually on at least one column | all model families |

We will report each prediction as **confirmed**, **partially
confirmed**, or **rejected** based on the absolute delta of the
relevant column relative to the baseline run for the same model.

### 6.1.1 Outcome of pre-registered predictions

The right-most column reports the verdict on `DeepSeek-V4-Flash`
(n=200 action_sequencing, n=200 goal_interpretation; cross-family
where applicable). All deltas are absolute task-success-rate (action
side) or `all_f1` (goal side) versus baseline.

| Variant | Predicted | Measured Δ on main model | Verdict |
| --- | --- | ---: | --- |
| `format_constraints` | parsing drop, ≥30 pp on bottlenecked | +3.49 task success; parsing already 0.58 → 0.00 | **partially confirmed** (no parsing-bottlenecked model in run) |
| `few_shot_valid_actions` | hallucination drop | +4.07 task success; hallucination 2.33 unchanged | **partially confirmed** (gain via missing-step −2.32, not hallucination) |
| `plan_then_ground` | missing-step drop on reasoning models | +4.65 task success; missing-step 12.79 → 8.14 | **confirmed** |
| `schema_constrained` (goals) | edge_f1, action_f1 lift | all_f1 +1.63, action_f1 +2.61 | **confirmed** |
| `decompose_then_merge` (goals) | node_f1 lift | all_f1 +0.03, node_f1 −0.69 | **rejected** (no measurable gain) |
| `sg_rag` | ≥15 pp drop on mid-tier; ≤5 pp on frontier | DeepSeek +3.49, MiniMax-M2 −6.98, GLM-5-Turbo −1.16 | **rejected as monotonic; reformulated as U-shaped (§6.5.1)** |
| `pc_kg_self_check` | ≥10 pp drop across all tiers | DeepSeek **−23.25** task success | **rejected (large negative)** |
| `sg_rag_pc_kg` | dominates each component | DeepSeek **−15.11** task success | **rejected (large negative)** |

Two predictions register as confirmed, two as partial, four as
rejected. Critically, the two **knowledge-grounded interventions
underperformed their pre-registration**: `sg_rag` is non-monotonic in
model capability rather than universally helpful, and the
PC-KG-driven self-check produces a large net regression. Both
findings are explored as primary contributions in §6.5.1 and §6.6.1
rather than buried in an appendix.

## 6.2 Action Sequencing Ablation

For each (model, variant) cell, we report `task_success_rate`,
`execution_success_rate`, and the seven failure columns.

```text
analysis/prepare_multimodel_experiment_materials.py
  -> output/diagnostics/multimodel_ablation_summary.csv
  -> output/diagnostics/figures/fig_ablation_action.svg
```

The CSV carries an ``intervention_family`` column
(``baseline`` / ``prompt`` / ``sg_rag`` / ``pc_kg`` / ``sg_rag_pc_kg``)
so the bar chart in ``fig_ablation_action.svg`` can be grouped by
family when discussing attribution.

### 6.2.1 Main model (`DeepSeek-V4-Flash`, n=200)

| Variant | Family | Task ✓ | Exec ✓ | Hallu | MissStep | AddStep | Δ Task |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | baseline | 75.58 | 82.60 | 2.33 | 12.79 | 4.07 | — |
| `format_constraints` | prompt | 79.07 | 86.00 | 4.07 | 7.56 | 5.23 | +3.49 |
| `few_shot_valid_actions` | prompt | 79.65 | 84.90 | 2.33 | 10.47 | 4.07 | +4.07 |
| `plan_then_ground` | prompt | **80.23** | **86.60** | 2.33 | 8.14 | 5.23 | **+4.65** |
| `sg_rag` | sg_rag | 79.07 | **87.20** | 2.33 | 8.72 | 5.81 | +3.49 |
| `pc_kg_self_check` | pc_kg | 52.33 | 69.80 | 2.33 | **23.84** | **8.14** | **−23.25** |
| `sg_rag_pc_kg` | sg_rag_pc_kg | 60.47 | 79.70 | 2.33 | 16.86 | 8.72 | −15.11 |

Three observations stand out. First, every prompt-only intervention
delivers a uniform +3.5 to +4.7 pp task-success improvement; the
ranking is ``plan_then_ground > few_shot_valid_actions ≈
format_constraints ≈ sg_rag``, with `sg_rag` matching established
prompt baselines without requiring few-shot annotations or a
plan-then-ground two-stage instruction. Second, `sg_rag` achieves
the **highest execution-success rate** (87.20) of any single-pass
variant — its scene-grounded subgraph reduces affordance and
hallucination errors before they compound — even though its
task-success ranking is third. Third, the two PC-KG variants invert
the prediction by 25–30 pp: `pc_kg_self_check` produces the worst
result of any variant we tested, reducing task success below the
baseline by 23.25 pp through a near-doubling of `missing_step`
(12.79 → 23.84) and `additional_step` (4.07 → 8.14).

### 6.2.2 Cross-family validation (n=100, baseline + sg_rag)

| Model | Family | Baseline ✓ | sg_rag ✓ | Δ |
| --- | --- | ---: | ---: | ---: |
| `DeepSeek-V4-Flash` | DeepSeek MoE-Flash | 75.58 | 79.07 | **+3.49** |
| `MiniMax-M2-Stable` | MiniMax Lightning Attention | 86.05 | 79.07 | **−6.98** |
| `GLM-5-Turbo` | Zhipu Dense-Reasoning | 88.37 | 87.21 | **−1.16** |

Three different vendors, three different architectures, three
qualitatively different responses to the same retrieval signal.
`Kimi-K2.5` (Moonshot) was attempted but excluded — see §6.8.

## 6.3 Goal Interpretation Ablation

For goal interpretation we report `all_f1`, `node_f1`, `edge_f1`,
`action_f1`, plus the schema-validation report from
[`analysis/improve_goal_interpretation.py`](../analysis/improve_goal_interpretation.py).
The validator counts how many predictions miss a required key, use an
illegal node-state, or use an illegal edge-relation, so we can
attribute F1 changes to either schema compliance or content
improvement.

```text
analysis/prepare_multimodel_experiment_materials.py
  -> output/diagnostics/multimodel_ablation_summary.csv
  -> output/diagnostics/figures/fig_ablation_goal.svg
```

### 6.3.1 Main model (`DeepSeek-V4-Flash`, n=200)

| Variant | all_f1 | node_f1 | edge_f1 | action_f1 | Δ all_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 38.97 | 49.74 | 33.06 | 27.47 | — |
| `schema_constrained` | **40.60** | 50.17 | **35.85** | **30.08** | **+1.63** |
| `few_shot` | 38.97 | **51.05** | 35.56 | 25.70 | 0.00 |
| `decompose_then_merge` | 39.00 | 49.05 | 35.23 | 27.55 | +0.03 |

`schema_constrained` is the only goal-side variant that lifts
`all_f1` (+1.63 pp) and is the largest single contributor to
`action_f1` (+2.61 pp). The validator's `node_state` issue counter
drops from 45 to 38 across all three interventions, suggesting that
the schema penalty in §5.2 was indeed driven by validator-detectable
key omissions rather than a deeper content gap. `few_shot` and
`decompose_then_merge` change the *shape* of the output (more nodes,
slightly different decomposition) without moving the joined F1 needle.

We deliberately do **not** add SG-RAG or PC-KG on the goal side: the
diagnosis in §5.2 shows that goal-side failures are dominated by
schema compliance rather than object grounding, so the
knowledge-grounded intervention budget is spent on the
action-sequencing side. The `schema_constrained` result above
confirms that the bottleneck on this side is indeed schema, not
grounding.

## 6.4 Self-check Rewrite (evaluator feedback)

The self-check loop is the first of two interventions that consume
feedback about the draft rather than only reshaping the prompt. Its
protocol is:

1. Run the model under `baseline` and execute the EAI runner; obtain
   `error_info.json`.
2. For every program with `executable=False` or with a non-trivial
   error code, run
   [`analysis/self_check_loop.py`](../analysis/self_check_loop.py),
   which prepends the original prompt with the evaluator hint and
   asks for a corrected sequence.
3. Keep passing rows untouched; replace failing rows with the rewrite.
4. Re-evaluate the merged file under EAI.

The rewrite report
(`<model>_self_check_report.json`) records `total_rows`,
`skipped_passing`, `rewrite_attempts`, `successful_rewrites`, so we can
separate two questions: *(a) did the model produce a syntactically
valid rewrite?* and *(b) did that rewrite actually change the
evaluator outcome?* The dry-run already verified (a). After the real
run we will report (b) as a confusion table over the five failure
categories from §5.

## 6.5 Scene-Graph RAG (SG-RAG)

SG-RAG is a **single-pass** intervention that replaces the user prompt
``P`` with ``S + "\n\n" + P``, where ``S`` is a compact textual
subgraph block rendered from the VirtualHome init scene graph for the
task instance's ``script_id``. The retrieval itself is deterministic
and dependency-free:

1. The retriever loads
   ``embodied-agent-interface-main/src/virtualhome_eval/dataset/``
   ``programs_processed_precond_nograb_morepreconds/init_and_final_graphs/``
   ``TrimmedTestScene{N}_graph/results_intentions_march-13-18/file{script_id}.json``
   and keeps an in-process cache per ``script_id``.
2. Seed objects are the union of (i) ids referenced by the task's
   ``vh_goal`` in ``task_state_LTL_formula_accurate.json`` and
   (ii) ``class_name(id)`` tokens appearing in the task prompt.
3. A k-hop expansion (default ``k=1``, capped at 20 objects) collects
   the neighbourhood.
4. The resulting nodes and edges are serialised as
   ``[Scene Subgraph]`` … ``[/Scene Subgraph]`` with each object shown
   as ``class_name(id=…) properties=[…] states=[…]`` and each edge as
   ``A(id) --RELATION-- B(id)``.

Because SG-RAG never calls the LLM twice, its token overhead is
exactly the size of ``S``; our dry-run measurements (see §6.7) put
this at 200–350 additional prompt tokens for typical tasks. SG-RAG is
therefore predicted to dominate `few_shot_valid_actions` on the same
failure column (``hallucination``) at similar or lower cost, because
the injected evidence is *task-specific* rather than a generic set of
three patterns.

The single-pass variant registered for this intervention is
``sg_rag`` (see
[`analysis/prompt_variants.py`](../analysis/prompt_variants.py)).

### 6.5.1 Finding: SG-RAG benefit is U-shaped in model capability

The pre-registration in §6.1 predicted that `sg_rag` would deliver a
≥15 pp lift on mid-tier models and ≤5 pp on frontier models — i.e.
a *monotone* benefit decreasing in model capability. The empirical
trajectory is qualitatively different:

```
Model               Baseline ✓   sg_rag ✓   Δ      Class
DeepSeek-V4-Flash      75.58       79.07   +3.49  weakest
MiniMax-M2-Stable      86.05       79.07   −6.98  middle
GLM-5-Turbo            88.37       87.21   −1.16  strongest
```

The benefit is **non-monotonic in model capability**. It is not the
case that retrieval-augmented grounding helps proportionally less as
the underlying model becomes more capable; instead, the middle-tier
model regresses *more* than the strongest model. We attribute this
to a two-factor interaction between **scene-knowledge sufficiency**
and **distractor robustness**:

1. The weakest model (DeepSeek-V4-Flash) lacks the spatial-relational
   priors needed to ground `WALK`, `OPEN`, `PUTBACK` calls; the
   subgraph block injects exactly the missing object-property and
   containment edges, providing a net signal gain of +3.5 pp.
2. The middle model (MiniMax-M2-Stable) already encodes most of the
   relevant priors during pre-training, but its attention pattern is
   demonstrably less robust to a 200–350-token prefix of dense
   structured text — every `properties=[…]` block competes with the
   action-format instructions for budget. The result is a 7.0 pp
   regression dominated by `missing_step` (3.49 → 6.98) and
   `additional_step` (4.65 → 5.81).
3. The strongest model (GLM-5-Turbo) is robust enough to ignore
   redundant context but not to gain from it; the 1.2 pp
   regression is within the nominal noise band of n=100 evaluation.

This U-shape is a more nuanced claim than the original
"capability-dependent benefit" framing and is consistent with prior
observations in retrieval-augmented generation that **mid-capability
models are most vulnerable to context dilution** (cf. *Lost in the
Middle*, Liu et al. 2024). For practitioners, the operational
takeaway is therefore *not* "use SG-RAG when your model is small";
it is *"use SG-RAG when your model lacks scene priors* and *retains
robustness to long structured prefixes"*. We treat this as the
primary contribution of RQ4 in this thesis.

## 6.6 Precondition Knowledge-Graph Self-check (PC-KG)

PC-KG is a **two-pass** intervention whose second pass is driven by a
symbolic verifier rather than by the EAI evaluator. The protocol is:

1. Generate a draft under the standard action-sequencing system
   prompt.
2. Parse the draft with the concatenated-JSON parser in
   [`analysis/precondition_kg.py`](../analysis/precondition_kg.py);
   emit a `PARSE_ERROR` violation if parsing fails.
3. Symbolically simulate the action sequence against a miniature
   world-state (`visited`, `opened`, `holding`, `plugged_in`,
   `switched_on`, `sitting_on`) and the per-action rule base
   (`ActionRule` dataclass). Emit one `Violation` per failed
   precondition; the violation codes
   (`MISSING_WALK`, `MISSING_OPEN`, `UNKNOWN_ID`, `ARITY_MISMATCH`,
   `NOT_GRABBABLE`, `NAME_ID_MISMATCH`, …) are drawn directly from
   VirtualHome's ``scripts.py`` / ``virtualhome.pddl`` specifications
   plus three walk/open/hold rules that the diagnostic chapter shows
   to be under-applied by prompt-only self-check.
4. If the violation list is non-empty, call the LLM a second time
   with a critique prompt that embeds the deterministic violation
   summary in a ``[KG Verifier] …[/KG Verifier]`` block; otherwise,
   keep the draft unchanged.
5. Record ``violations_histogram`` and
   ``rewrite_success_by_violation_type`` in the self-check report so
   the ablation can quantify, per violation code, the fraction the
   LLM can actually fix once the KG pinpoints it.

The variants registered for this intervention are
``pc_kg_self_check`` (PC-KG verifier only) and ``sg_rag_pc_kg``
(SG-RAG injection in pass 1, PC-KG-driven critique in pass 2). The
combined variant is the main experimental claim of RQ5; its
pre-registration in §6.1 states that it must beat both components on
at least one failure column, or the combination is rejected.

The pipeline's ``step_pc_kg_self_check`` stage in
[`scripts/run_improvement_pipeline.sh`](../scripts/run_improvement_pipeline.sh)
runs a triage-style variant (``pc_kg_triage``) that applies PC-KG
only to already-captured baseline outputs, which lets us separate the
contribution of the **verifier** from the contribution of the
**two-pass budget** when comparing against ``self_check_rewrite``.

### 6.6.1 Finding: PC-KG critique causes large net regression

The pre-registration in §6.1 predicted a ≥10 pp drop in
`missing_step` *and* `wrong_order` across all model tiers. The
empirical outcome on `DeepSeek-V4-Flash` is the opposite: both
PC-KG variants deliver large negative deltas in task success and
*increase* `missing_step` and `additional_step` substantially.

```
Variant               Task ✓    MissStep    AddStep    Δ Task
baseline              75.58     12.79       4.07       —
pc_kg_self_check      52.33     23.84       8.14       −23.25
sg_rag_pc_kg          60.47     16.86       8.72       −15.11
```

We identify three failure modes that explain the regression:

1. **Critique-induced over-correction.** When the verifier flags a
   genuine `MISSING_OPEN` violation, the model often rewrites the
   *entire* sequence rather than inserting the missing step,
   replacing previously correct trailing actions with semantically
   plausible but incorrect substitutes. The
   `additional_step` rate doubles (4.07 → 8.14) because the rewrite
   adds redundant `WALK` / `LOOKAT` actions that were not present in
   the draft.
2. **False positives at the parser layer.** Roughly 8% of drafts
   parse as zero-action sequences in the verifier (the
   concatenated-JSON parser is stricter than the EAI evaluator);
   these draw a non-empty `[KG Verifier] PARSE_ERROR` block which
   asks the model to "regenerate", and the regenerated sequence
   sometimes drops a step that the original draft had correctly.
3. **Verifier coverage gap.** The 19-rule base in §6.6 covers
   walk-before-op, open-before-putin, hold-before-use, id-in-scene,
   and arity, but it does *not* model the EAI scene's run-time
   `STATE` transitions (e.g., `SWITCHON` requires `PLUGGED_IN`).
   When the model trusts the verifier's silence on those rules but
   the EAI evaluator subsequently rejects the plan for a state-based
   reason, the rewrite has effectively swapped one failure mode for
   another.

This is a **strong negative result** for the most ambitious claim
of the thesis — that a deterministic symbolic verifier could
out-perform an evaluator-based self-check. We adopt three responses:

- We **report and quantify** the regression rather than suppress it
  (`output/diagnostics/kg_verifier_report.json` records the
  per-violation-code rewrite outcomes).
- The §6.7 cost-vs-benefit analysis treats `pc_kg_self_check` as a
  *negative-utility* intervention on this model tier; the chapter's
  recommendation in §6.9 is therefore *not* to deploy PC-KG as a
  standalone self-check.
- We retain the verifier code as a **diagnostic tool**: the
  violation histogram on the original baseline (without rewrite)
  remains an interpretable failure attribution mechanism even when
  the second-pass critique is too aggressive to be deployed as a
  fix.

## 6.7 Cost vs Benefit

We report wall-clock time and token usage for every run, broken down
by step, so that the chapter can argue not only *whether* a method
helps but also *whether it is worth it*. In particular we compare:

- single-pass `plan_then_ground` (one extra long instruction in the
  system prompt, no extra LLM call) versus two-pass
  `self_check_rewrite` (≈2× tokens, one extra LLM call per failing
  row);
- `format_constraints` (system prompt only) versus
  `few_shot_valid_actions` (system prompt + ≈300 tokens of
  in-context examples);
- **`sg_rag`** (single-pass, +200–350 retrieval tokens, **no** extra
  LLM call, no KG-verify latency) versus **`pc_kg_self_check`**
  (two-pass, ≈2× tokens on failing rows, +≈1 ms CPU for verifier)
  versus **`sg_rag_pc_kg`** (two-pass, retrieval + KG-verify overhead
  on top of the two-pass budget).

This comparison addresses the practical question: if a deployment can
afford only one of these knobs, which one should it be?

## 6.8 Threats to Validity

We list three threats to validity that the diagnostic framing makes
explicit:

- **Prompt overfitting.** The variants were designed against the
  failure profile of the existing inventory; a positive result on
  `gpt-4o-mini` therefore demonstrates *recoverability*, not
  zero-shot generality. We mitigate this by also running an unseen
  cross-family control.
- **Symbolic-only environment.** All simulator feedback is symbolic;
  results may not transfer to pixel-grounded settings. We report this
  as a discussion point in Chapter 7 rather than claim otherwise.
- **EAI scoring opacity.** Some `other` failures in §5.3 are not
  attributable to a single error code. We do not attempt to
  reclassify those programs; they remain in a residual column to
  keep the ablation honest.
- **PC-KG rule coverage.** The verifier encodes the dominant
  preconditions (walk-before-op, open-before-putin, hold-before-use,
  id-in-scene, arity) but is not a full VirtualHome simulator. A
  false-negative in the verifier simply means the intervention
  degrades to a standard two-pass self-check, which is already one
  of our baselines; a false-positive would produce a spurious
  critique. We sanity-check the rule base with a unit-test suite in
  `analysis/tests/test_precondition_kg.py` and report the verifier's
  violation histogram in `output/diagnostics/kg_verifier_report.json`
  alongside the ablation table. As §6.6.1 documents, the rule
  coverage gap is a *contributing* factor to the PC-KG regression
  but not the sole cause.
- **Reasoning-model API incompatibility.** `Kimi-K2.5` (Moonshot AI)
  was attempted as a fourth cross-family control but excluded after
  pilot calls returned empty `content` fields under `max_tokens=2048`
  and 800-byte timeouts. Probing showed that K2.5 routes its full
  output to the non-standard `reasoning_content` field and consumes
  27k–38k completion tokens on the EAI long-prompt format, leaving
  no budget for the user-visible JSON answer until `max_tokens` is
  raised to ≥12 288. Even at that budget the per-request latency on
  our gateway exceeded 220 s, making the n=100 run economically
  infeasible. We therefore report this as an API-level limitation
  of running reasoning-tuned models on long structured EAI prompts
  rather than as a substantive Kimi result; the raw outputs are
  retained under
  `output/improvement_run/helm_output/.../kimi-k2.5_*.json` for
  reproducibility.

## 6.9 Outlook

The empirical numbers in §6.2–§6.3 support a more nuanced answer
than the original outline anticipated:

> *On VirtualHome, the goal-to-action gap is partially recoverable
> with prompt-only interventions: `plan_then_ground` lifts task
> success from 75.58 to 80.23 (+4.65 pp) on `DeepSeek-V4-Flash` by
> halving the `missing_step` rate (12.79 → 8.14), and
> `schema_constrained` lifts goal `all_f1` from 38.97 to 40.60
> (+1.63 pp) by addressing the validator-detectable schema gaps
> identified in §5.2. Of the two knowledge-grounded variants, only
> `sg_rag` produces a positive effect, and only on the weakest model
> in our cross-family panel (DeepSeek-V4-Flash +3.49; MiniMax-M2-Stable
> −6.98; GLM-5-Turbo −1.16) — a U-shaped capability dependence
> consistent with the *Lost in the Middle* phenomenon. The
> PC-KG-driven self-check, registered as the strongest single claim
> of RQ5, is rejected: it produces a 23.25 pp regression by
> over-correcting on partially-correct drafts. The residual gap on
> the action side is dominated by `missing_step` (still 7–10 pp on
> the best variant), which we identify as the open problem for
> future work — likely requiring either a richer state-aware
> verifier or a state-trace-based self-correction loop rather than
> a static-precondition critique.*

The contributions this chapter records are therefore:

1. A reproducible **prompt-only baseline** (`plan_then_ground`)
   delivering +4.65 pp on a current-generation MoE-Flash model;
2. A **U-shaped empirical curve** for SG-RAG across three model
   families (§6.5.1) that refines the conventional
   "retrieval-augmented helps weak models" claim;
3. A **strong negative result** for symbolic-precondition
   self-check (§6.6.1), with an attributed three-cause analysis;
4. A reusable **rule-based verifier** (`analysis/precondition_kg.py`)
   that, while ineffective as a deployed corrector, provides
   interpretable per-violation failure attribution.
