# 7. Discussion

The diagnostic framing of Chapters 4–5 and the controlled ablations of
Chapter 6 jointly support a more nuanced reading of "LLMs as embodied
controllers" than either a leaderboard or a single-method paper would
yield. We organise the discussion around three questions implied by the
results.

## 7.1 Recoverable vs. structural failures

The five-category taxonomy of §5.1 separates failures that prompt-level
interventions can attenuate from those that survive every variant we
tested. **Format / parsing** errors are essentially fully recoverable:
`format_constraints` and `few_shot_valid_actions` reduce the parsing
column to ≤1 % on the main model, and the same effect transfers across
both cross-family controls. **Relation / id grounding** errors are
recoverable but capability-dependent: SG-RAG closes a meaningful share
on the weakest model, but the same prefix becomes a distractor on
mid-capability models (§6.5.1). **Missing-step** errors are
*partially* recoverable through `plan_then_ground` (halved on the main
model) but remain the dominant residual after every intervention,
including the best variant — they constitute the structural core of the
goal-to-action gap. **Wrong-order** and **affordance** errors are not
meaningfully reduced by any prompt-only intervention in our panel; the
PC-KG critique was the only mechanism explicitly designed to address
them, and it failed for the reasons documented in §6.6.1.

The practical implication is that a deployment seeking the best
prompt-only outcome should compose `format_constraints` (for parsing
robustness) with `plan_then_ground` (for missing-step reduction), and
should add `sg_rag` only after profiling whether the target model
benefits from or is distracted by structured prefixes.

## 7.2 Cost vs. benefit of self-check loops

A common assumption in the self-correction literature is that adding a
critique pass is approximately free in expected utility, because the
model can choose to reproduce its first draft. Our PC-KG result is a
strong counter-example: a deterministic, well-typed critique signal
does **not** guarantee improvement, even when the verifier rules are
calibrated against the dominant failure column. Two mechanisms drive
the regression. First, the rewrite pass treats the critique as a strong
negative reward and rewrites large portions of an already-correct
sequence, doubling `additional_step` and roughly doubling `missing_step`
in the rewritten outputs. Second, parser-layer false positives — the
verifier flags a draft as malformed when in fact only a JSON wrapper is
slightly off-spec — produce empty rewrites in roughly 8 % of rows. The
two-pass `sg_rag_pc_kg` variant inherits both mechanisms.

This does **not** invalidate self-check as a research direction. It
shows that a critique loop must (i) be calibrated against false
positives, (ii) be allowed to *abstain* rather than always rewrite, and
(iii) be coupled to a verifier whose recall on the dominant failure
column is high enough to outrun the noise it introduces. None of these
were satisfied by our PC-KG protocol; we leave a state-trace-based
verifier-in-the-loop comparison as future work.

## 7.3 Implications for embodied LLM agent design

The U-shaped SG-RAG curve, taken together with the PC-KG negative
result, argues against a "stack everything" approach to grounding. For a
practitioner choosing where to spend a token budget on VirtualHome-like
benchmarks, our results recommend the following ordering: (1) enforce
output schema with format constraints, (2) add a single planning prefix
(`plan_then_ground`) to compress the missing-step column, (3) add
SG-RAG only after profiling that the target model is in the regime
where it helps. Two-pass critique loops should be added last, with
explicit abstention behaviour and a verifier whose precision is
empirically validated against the diagnostic taxonomy of Chapter 5
rather than against a generic notion of "preconditions".

## 7.4 Threats and limitations

§6.8 enumerates the chapter-specific threats. Three deserve repetition
at the discussion layer. **First**, every result is on VirtualHome's
symbolic interface; transfer to pixel-grounded settings is not
established and should not be inferred from our numbers. **Second**,
our cross-family panel has three vendors; the U-shape is a finding on
this panel, not a universal claim about model capability — a fourth
data point in the high-capability regime (e.g. a frontier closed-source
model) would strengthen or refute the curve, and we were unable to add
one because of API gateway constraints (§6.8). **Third**, the PC-KG
verifier covers the dominant preconditions but is not a full VirtualHome
simulator; a richer verifier might yield a different sign — we
explicitly do not claim that *no* symbolic-verifier-in-the-loop scheme
can work on this benchmark, only that ours did not.

# 8. Conclusion and Future Work

This thesis re-framed "LLMs on EAI/VirtualHome" as a diagnostic
question — *conditional on understanding the goal, which errors
persist?* — and used that framing to design and evaluate two
knowledge-grounded recovery methods alongside lightweight
prompt-only baselines.

Empirically, on the main cross-family model
(`DeepSeek-V4-Flash`, n=200), the goal-to-action gap is **partially
recoverable**. The best prompt-only intervention,
`plan_then_ground`, lifts task success from 75.58 to 80.23
(+4.65 pp) by halving `missing_step`, and `schema_constrained`
lifts goal `all_f1` from 38.97 to 40.60 by addressing schema
compliance gaps. The two knowledge-grounded interventions
delivered findings sharper than the pre-registration anticipated:
SG-RAG benefit is **U-shaped in model capability** across our
three-vendor panel, helping only the weakest model and hurting
mid-capability ones; PC-KG self-check produces a 23.25 pp
regression and is rejected as a deployed intervention.

The four contributions are: (i) a reproducible multi-vendor
generation and evaluation pipeline; (ii) a fine-grained diagnosis
that separates recoverable from structural failures across 17
pre-existing model snapshots and 3 newly-evaluated families; (iii)
the **U-shaped empirical curve** for SG-RAG (§6.5.1), refining
the conventional "retrieval-augmented helps weak models" claim;
and (iv) a strong **negative result** for symbolic-precondition
self-check (§6.6.1), accompanied by a reusable rule-based
verifier that remains useful as a diagnostic instrument.

**Future work** falls into three concrete directions. *Cross-domain
extension.* The same diagnostic plus ablation pattern should be
ported to BEHAVIOR (already part of EAI) to test whether the
U-shaped SG-RAG curve and the missing-step structural floor
generalise beyond VirtualHome. *Verifier-in-the-loop redesign.*
The PC-KG regression suggests that future critique loops should
be evaluated under three explicit constraints — calibrated false
positives, an abstention action, and per-failure-column recall —
and the natural next experiment is a state-trace-based verifier
that simulates the draft against a partial VirtualHome environment
rather than checking static preconditions. *Open-weight
reproduction.* Because the U-shape was observed on three closed
APIs, a reproduction on three open-weight families (Llama-3, Qwen-3,
Mistral) at matched scale would settle whether the curve is an
artefact of vendor-specific RLHF or a genuine capability effect.
