# 1. Introduction

Large language models (LLMs) increasingly act as the reasoning layer of
embodied agents. They translate natural language instructions into
high-level intent, decompose tasks, and emit either action commands or
sub-goals for downstream execution. Despite this growing role, evaluating
whether LLMs are reliable embodied controllers remains difficult. Final
success or failure on a household task captures only the end of the
pipeline; it conflates errors in goal understanding, in plan
generation, in physical preconditions, and in textual formatting. The
Embodied Agent Interface (EAI) benchmark (Li et al., NeurIPS 2024)
addresses this by decomposing decision making into four standardised
modules — goal interpretation, subgoal decomposition, action sequencing,
and transition modeling — and reporting per-error-type diagnostics on
top of overall success rates.

This thesis focuses on the most striking pattern that emerges once such
fine-grained metrics are available: many failures occur **after** the
model has correctly interpreted the goal. A model can list the desired
node states, the spatial relations to enforce, and the actions that
must take place; yet when it produces the corresponding VirtualHome
action sequence, it adds illegal objects, omits a required `WALK`, swaps
two preconditioned actions, or simply violates the JSON output schema.
We refer to this gap as the *goal-to-action gap*. The revised thesis
title — *Bridging the Goal-to-Action Gap: A Diagnostic Study and
Knowledge-Grounded Recovery of LLM Failures in Embodied Planning* — is
meant literally: we treat the gap as the central object of study, and
we test whether grounding LLM outputs against the symbolic structure
already present in VirtualHome (scene graphs and precondition rules)
can close a meaningful fraction of it without environment-specific
fine-tuning.

We pursue five research questions. **RQ1** asks whether goal
interpretation ability and action sequencing ability are correlated
across modern LLMs, or whether they should be reported as distinct
skills. **RQ2** decomposes the gap by failure category, focusing on
cases of high goal-interpretation F1 paired with action sequencing
failure. **RQ3** asks whether failure profiles cluster by model family
(OpenAI, Anthropic, Google, Meta, Mistral, Cohere). **RQ4** asks which
lightweight prompt-only interventions — format-constrained prompting,
few-shot examples grounded in legal VirtualHome traces, self-check
rewriting, plan-then-ground decomposition, and schema-constrained goal
output — shrink the gap, and which failure categories survive these
interventions unchanged. **RQ5** asks whether two knowledge-grounded
interventions — Scene-Graph Retrieval-Augmented Grounding (SG-RAG),
which injects a task-relevant object subgraph into the prompt, and
Precondition Knowledge-Graph Verification (PC-KG), which symbolically
validates a draft against a small rule base before feeding the
violations back as critique — measurably outperform their prompt-only
counterparts on the failure categories that the diagnosis flags as
dominant, and whether their combination (SG-RAG + PC-KG) is strictly
better than either alone.

Our contributions are fourfold. First, we provide a unified diagnostic
view of LLMs on EAI/VirtualHome that separates goal interpretation from
action sequencing and reports fine-grained errors per model. Second, we
build a reproducible multi-vendor generation and evaluation pipeline
that supports controlled ablation between prompt and self-check
variants across three current-generation cross-family models. Third,
we present an empirical study showing that, while format and grounding
errors can be largely eliminated by prompt-level changes, the dominant
residual after every intervention we tested is `missing_step`, which a
plan-then-ground prefix only halves rather than removes. Fourth, we
report two knowledge-grounded interventions whose results are sharper
than the pre-registration anticipated: SG-RAG benefit is **U-shaped in
model capability** across our three-vendor panel — it helps only the
weakest model and acts as a distractor on mid-capability ones — and
the PC-KG-driven two-pass self-check is **a strong negative result**,
producing a 23.25 pp task-success regression that we attribute to
critique-induced over-correction, parser-layer false positives, and
verifier coverage gaps. The combined `sg_rag_pc_kg` variant inherits
the regression and is also rejected. We retain the rule-based PC-KG
verifier as an interpretable per-violation diagnostic instrument even
though it is ineffective as a deployed corrector. The remaining
chapters follow this contribution arc: Chapter 2 reviews the relevant
literature, Chapter 3 details the experimental setup, Chapter 4 reports
the multi-model evaluation, Chapter 5 dissects failure types,
Chapter 6 reports the improvement ablations, and Chapters 7–8 discuss
implications and future work.

# 2. Background and Related Work

**Embodied benchmarking with LLMs.** EAI introduces a unifying
interface for embodied decision making, decomposing tasks into goal
interpretation, subgoal decomposition, action sequencing and transition
modeling. It evaluates 18 LLMs on VirtualHome and BEHAVIOR with
fine-grained error categories such as `parsing`, `hallucination`,
`missing_step`, `wrong_order`, `affordance_error` and `additional_step`.
This level of granularity is what makes the goal-to-action gap visible
in the first place. Concurrent work like EmbodiedBench (Liu et al.,
2025) reaches a complementary conclusion: long-horizon planning is the
dominant bottleneck for current multi-modal LLM agents. Older symbolic
benchmarks (BEHAVIOR-1K, ALFWorld) target narrower slices of the same
problem; we adopt EAI because its module separation is the closest
match to our research questions.

**Action sequencing and grounding.** VirtualHome (Puig et al., 2018)
provides a household simulator with a typed action language whose
arguments are objects identified by `<class_name, id>` pairs. EAI
extends VirtualHome with a textual interface where the model must emit
JSON-encoded action commands; this transforms grounding errors into
observable categories such as wrong object id (`relation_grounding`),
missing precondition steps (`missing_step`), or precondition violations
(`affordance_error`). Several lines of work address grounding through
external tools (PaLM-SayCan, ProgPrompt) or via prompt-level constraints
(Code-as-Policies, ReAct, Plan-and-Act). Our improvement experiments are
in the prompt-level family because the EAI evaluator is symbolic and
we want our findings to transfer across model providers.

**Self-correction and reflection.** Self-Refine (Madaan et al., 2023)
and Reflexion (Shinn et al., 2023) demonstrate that LLMs can revise
their own outputs given verbal feedback. These ideas have been
critically examined: when models lack access to a verifier, naïve
self-critique can degrade rather than improve performance (Huang et
al., 2024). Recent work shows that with sufficient domain structure
(action preconditions, explicit checklists), intrinsic self-critique
can lift planning performance substantially (Valmeekam et al., 2025;
SPIRAL, 2025). Our `self_check_rewrite` and `plan_then_ground`
variants build on this view: we deliberately bake EAI-style checks
(walk-before-act, open-before-putin, name/id consistency) into the
critique prompt, and we treat the EAI evaluator's `error_info.json` as
optional feedback rather than required supervision.

**Reasoning-and-acting prompts.** ReAct (Yao et al., 2022) interleaves
reasoning traces and actions, while Plan-and-Act (2025) shows that
explicitly separating planner and executor roles improves long-horizon
web navigation success rates. We do not aim for a new SOTA framework;
instead, we treat plan-then-ground and self-check rewriting as
controlled interventions to test which fraction of the goal-to-action
gap can be closed *with the same model and the same prompt budget*.

**Knowledge-grounded recovery.** Two strands of recent work motivate
our final intervention family. KGHaluBench (2024) shows that grounding
LLM outputs against a curated knowledge graph meaningfully attenuates
hallucination in breadth-and-depth retrieval settings, arguing for
structured rather than free-form retrieval when the domain itself is
structured. In parallel, embodied-planning studies such as SayPlan
(Rana et al., 2023) and 3D-Scene-Graph-Prompting (2024) have shown
that routing a task-relevant subgraph of the environment into the
prompt reduces id hallucination and affordance errors in VirtualHome-
style simulators. We port both ideas into the EAI/VirtualHome
diagnostic setup. **SG-RAG** retrieves, for each task instance, the
seed objects referenced by the task's `vh_goal` together with their
k-hop neighbours in the init scene graph, and injects them as a
compact ``[Scene Subgraph]`` block before the LLM call; it is
implemented as a stand-alone Python module
([`analysis/scene_graph_rag.py`](../analysis/scene_graph_rag.py))
without any vector store, keeping the retrieval reproducible on the
dry-run provider. **PC-KG** is a small symbolic rule base derived
from VirtualHome's action definitions in
``scripts.py`` and ``virtualhome.pddl``; a miniature verifier
([`analysis/precondition_kg.py`](../analysis/precondition_kg.py))
symbolically simulates a draft action sequence, emits structured
``Violation`` records (e.g. ``MISSING_WALK`` before ``SWITCHON``,
``MISSING_OPEN`` before ``PUTIN``, ``UNKNOWN_ID``), and feeds them as
a deterministic critique block into the self-check pass. These are
the only two interventions in the thesis that couple the model with
external structured knowledge; their combination is registered as the
``sg_rag_pc_kg`` variant and serves as the main test for RQ5. We
flag here, and document in §6.5.1 and §6.6.1, that the empirical
results refine these expectations: the SG-RAG benefit is non-monotone
in model capability, and the PC-KG self-check produces a net
regression on our main model rather than the pre-registered
improvement.

**Position of this thesis.** Most prior multi-LLM evaluations stop at
ranking. The EAI paper itself reports per-error-type tables but does not
study how easy each error type is to fix. We make two complementary
moves. First, we re-frame the leaderboard as a diagnostic question:
*conditional on understanding the goal, which errors persist?*. Second,
we run controlled ablations on the failure types that the diagnostic
flags as dominant. This places the work between large-scale
benchmarking studies (EAI, EmbodiedBench) and methodological papers on
self-correction (Reflexion, Self-Refine, Plan-and-Act): we use the
former's evaluation rigour to test the latter's claims in the embodied
setting.
