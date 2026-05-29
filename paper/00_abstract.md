# Abstract

Large language models (LLMs) increasingly serve as the reasoning layer of
embodied agents, yet a persistent gap separates *understanding a goal*
from *emitting an executable action sequence*. This thesis treats that
**goal-to-action gap** as the central object of study on the Embodied
Agent Interface (EAI) benchmark with VirtualHome. We combine a
fine-grained failure diagnosis of seventeen pre-existing model snapshots
with a controlled ablation on three current-generation cross-family
models (`DeepSeek-V4-Flash`, n=200; `GLM-5-Turbo`, n=100;
`MiniMax-M2-Stable`, n=100) covering five lightweight prompt
interventions (`format_constraints`, `few_shot_valid_actions`,
`plan_then_ground`, `schema_constrained`, `decompose_then_merge`) and
two knowledge-grounded interventions designed against the dominant
failure columns: **Scene-Graph Retrieval-Augmented Grounding (SG-RAG)**,
which injects a task-relevant object subgraph from VirtualHome's init
scene graph into the prompt without an extra LLM call, and a
**Precondition Knowledge-Graph self-check (PC-KG)**, which symbolically
verifies a draft against a 19-rule action base and feeds structured
violation records back as critique.

Three findings stand out. (i) The single best prompt-only intervention,
`plan_then_ground`, lifts action-sequencing task success from 75.58 to
80.23 (+4.65 pp) on the main model, primarily by halving the
`missing_step` rate (12.79 → 8.14). On the goal side,
`schema_constrained` lifts goal `all_f1` from 38.97 to 40.60 (+1.63 pp)
by closing schema-compliance gaps identified in the diagnosis chapter.
(ii) **SG-RAG exhibits a U-shaped capability dependence** rather than
the monotone improvement the outline pre-registered: it helps the
weakest cross-family model (DeepSeek-V4-Flash, +3.49 pp) but hurts the
middle (MiniMax-M2-Stable, −6.98 pp) and slightly hurts the strongest
(GLM-5-Turbo, −1.16 pp), consistent with the *Lost in the Middle*
phenomenon — strong scene priors plus a long structured prefix dilutes
attention rather than grounding it. (iii) The PC-KG self-check, the
most ambitious pre-registered claim, is **rejected**: as a two-pass
intervention it regresses task success by 23.25 pp on the main model,
attributable to critique-induced over-correction on partially-correct
drafts, parser-layer false positives, and a verifier coverage gap on
state transitions. The combined `sg_rag_pc_kg` variant inherits the
PC-KG regression (−15.11 pp) and is also rejected.

The contributions are: (1) a reproducible multi-vendor generation
pipeline with three-family cross-validation; (2) a fine-grained
diagnostic view of the goal-to-action gap that separates recoverable
from structural failures; (3) the U-shaped empirical curve for SG-RAG
across three model families, refining the conventional
"retrieval-augmented helps weak models" claim; and (4) a strong
**negative result** for symbolic-precondition self-check, accompanied
by a reusable rule-based verifier (`analysis/precondition_kg.py`) that
remains useful as a per-violation diagnostic instrument even though it
is ineffective as a deployed corrector. The residual gap on the action
side is dominated by `missing_step` (7–10 pp on the best variant),
which we identify as the open problem for future state-trace-based
correction loops.
