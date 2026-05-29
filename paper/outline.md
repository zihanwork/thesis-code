# Thesis Outline

**Working title:** Bridging the Goal-to-Action Gap: A Diagnostic Study
and Knowledge-Grounded Recovery of LLM Failures in Embodied Planning

## Thesis Statement
Large language models often demonstrate strong goal understanding in
embodied tasks yet still fail when generating executable action
sequences. This thesis treats that gap as the central object of study,
diagnoses which failure types dominate across model families on the
Embodied Agent Interface (EAI) benchmark with VirtualHome, and tests
**knowledge-grounded recovery methods** — Scene-Graph Retrieval-Augmented
Grounding (SG-RAG) and Precondition Knowledge-Graph Verification
(PC-KG) — alongside lightweight prompting and self-correction baselines,
showing how much of the gap can be closed without environment-specific
fine-tuning.

## Research Questions
- **RQ1.** Across modern LLMs, do goal interpretation ability and action
  sequencing ability correlate, or are they distinct skills that should
  be reported separately?
- **RQ2.** When a model attains high goal-interpretation F1 but still
  fails action sequencing, which fine-grained error categories
  (`missing_step`, `wrong_order`, `hallucination`, `relation_grounding`,
  `format_or_parsing`) dominate?
- **RQ3.** Are there systematic differences in failure profiles between
  proprietary and open-weight model families (OpenAI, Anthropic, Google,
  Meta, Mistral, Cohere)?
- **RQ4.** Which prompting and self-correction interventions
  (`format_constraints`, `few_shot_valid_actions`, `self_check_rewrite`,
  `plan_then_ground`, `schema_constrained` for goals) measurably lift
  goal F1 or task success rate, and which failures remain robust?
- **RQ5.** Do **knowledge-grounded** interventions — Scene-Graph
  Retrieval-Augmented Grounding (SG-RAG) and Precondition
  Knowledge-Graph Verification (PC-KG) — measurably improve over
  prompt-only baselines on the failure categories the diagnosis flags
  as dominant (`hallucination`, `missing_step`, `wrong_order`), and is
  their combination (SG-RAG + PC-KG) strictly better than either alone?

## Chapter Plan
1. **Introduction** — motivation, gap framing, RQ1–RQ4, contributions.
2. **Background and Related Work**
   - Embodied Agent Interface (Li et al., NeurIPS 2024)
   - VirtualHome simulator and program-level action format
   - LLM planning and reasoning baselines (ReAct, Plan-and-Act)
   - Self-correction (Self-Refine, Reflexion)
   - Comparable benchmarks (EmbodiedBench)
3. **Experimental Setup** — draft in [`paper/02_experimental_setup.md`](02_experimental_setup.md) (~1100 words)
   - Datasets: VirtualHome subset used by EAI; metrics for goal interpretation and action sequencing
   - Models: existing HELM/EAI snapshots (17 models) plus the new generation pipeline (`analysis/generate_outputs.py`)
   - Prompt variants: see `analysis/prompt_variants.py`
   - Evaluation pipeline: `scripts/run_improvement_pipeline.sh`
   - RQ → metric mapping table for downstream chapters
4. **Multi-model Evaluation (RQ1, RQ3)**
   - Use `output/diagnostics/multimodel_existing_inventory.csv`
   - Highlights: top action sequencing models, top goal interpretation
     models, family averages, scatter of goal vs action success
   - Figures: `fig_action_task_success.svg`,
     `fig_goal_interpretation_f1.svg`, `fig_family_average.svg`,
     `fig_goal_vs_action.svg`
5. **Failure Diagnosis (RQ2)** — draft in [`paper/03_failure_diagnosis.md`](03_failure_diagnosis.md) (~1430 words)
   - Five-category failure taxonomy and three regimes (format-, reasoning-, hallucination-bottlenecked)
   - Three real case studies (`Turn on light 125_2`, `Drink 510_1`, `Write an email 996_2`)
   - Joined goal-action analysis from `analysis/link_goal_to_action_failures.py`
   - Source artefacts: `output/diagnostics/failure_case_studies.md`,
     `output/diagnostics/multimodel_failure_profile.csv`,
     `output/diagnostics/failure_pattern_counts.json`
6. **Methods for Improving Success (RQ4, RQ5)** — skeleton in [`paper/06_methods_for_improving_success.md`](06_methods_for_improving_success.md) (~970 words; empirical TBDs to fill once real-provider runs complete)
   - **Pre-registered predictions** from §5.4: each variant is committed to a target failure column and a magnitude direction before the run.
   - Ablation tables and bars produced by `prepare_multimodel_experiment_materials.py` (`fig_ablation_action.svg`, `fig_ablation_goal.svg`).
   - Self-check rewrite protocol with confusion table over the five failure categories.
   - **Knowledge-Grounded Recovery** (RQ5): SG-RAG (scene-subgraph injection via `analysis/scene_graph_rag.py`) and PC-KG (precondition verifier via `analysis/precondition_kg.py`), plus the combined `sg_rag_pc_kg` variant.
   - Cost vs benefit comparison (single-pass vs two-pass; KG/RAG overhead).
   - Threats to validity (prompt overfitting, symbolic-only env, EAI scoring opacity).
7. **Discussion**
   - Recoverable vs structural failures
   - Cost vs benefit of self-check loops
   - Implications for embodied LLM agent design
8. **Conclusion and Future Work**
   - Cross-domain extension (BEHAVIOR)
   - Verifier-in-the-loop comparisons
   - Open-weight reproductions

## Mapping Between Code Artefacts and Paper Claims
| Section | Primary artefacts |
| --- | --- |
| Multi-model Evaluation | [`output/diagnostics/multimodel_experiment_materials.md`](../output/diagnostics/multimodel_experiment_materials.md), [`output/diagnostics/multimodel_existing_inventory.csv`](../output/diagnostics/multimodel_existing_inventory.csv), figures in [`output/diagnostics/figures`](../output/diagnostics/figures) |
| Failure Diagnosis | [`output/diagnostics/failure_case_studies.md`](../output/diagnostics/failure_case_studies.md), [`output/diagnostics/multimodel_failure_profile.csv`](../output/diagnostics/multimodel_failure_profile.csv) |
| Methods for Improving Success | [`output/diagnostics/multimodel_ablation_summary.csv`](../output/diagnostics/multimodel_ablation_summary.csv), [`output/diagnostics/multimodel_prompt_templates.md`](../output/diagnostics/multimodel_prompt_templates.md), [`scripts/run_improvement_pipeline.sh`](../scripts/run_improvement_pipeline.sh) |
| Related Work | [`output/diagnostics/multimodel_related_work.csv`](../output/diagnostics/multimodel_related_work.csv) |

## Contribution Statement (final)
- A unified diagnostic view of LLMs on EAI/VirtualHome that separates
  goal interpretation ability from action sequencing ability and
  reports fine-grained error types per model.
- A reproducible multi-vendor generation and evaluation pipeline that
  supports controlled ablation between prompt and self-check variants,
  with three-family cross-validation
  (`DeepSeek-V4-Flash`/`GLM-5-Turbo`/`MiniMax-M2-Stable`).
- An empirical study of lightweight interventions showing that
  `plan_then_ground` (+4.65 pp) and `schema_constrained` (+1.63 pp
  goal `all_f1`) are the most reliable prompt-only knobs, while
  `missing_step` is the dominant structural residual.
- A **U-shaped empirical curve** for SG-RAG across three model
  families (helps weakest, hurts middle, slightly hurts strongest),
  refining the conventional "retrieval-augmented helps weak models"
  claim.
- A **strong negative result** for symbolic-precondition self-check
  (PC-KG): a 23.25 pp task-success regression on the main model,
  with an attributed three-cause analysis. The rule-based verifier
  (`analysis/precondition_kg.py`) is retained as an interpretable
  per-violation diagnostic instrument.

