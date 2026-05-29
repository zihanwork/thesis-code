# CSC8639 Interim Report

**Project title:** Bridging the Goal-to-Action Gap: Diagnosing and Improving LLM Failures in Embodied Planning  
**Student:** Zihan Wu  
**Programme:** Project and Dissertation in Data Science  
**Benchmark:** Embodied Agent Interface (EAI) on VirtualHome  
**Date:** 2026-05-26

---

## 1. Introduction

Large language models (LLMs) are increasingly used as the reasoning component of embodied agents. In these systems, a model receives a natural-language instruction such as *turn on the light* or *read the book*, and is expected to produce actions that can be executed by an agent in an environment. This is promising because LLMs can interpret language and often form plausible high-level plans. However, embodied planning also requires grounded execution: the model must choose valid actions, use the correct objects, and respect preconditions in the current scene.

This project studies that problem using the Embodied Agent Interface (EAI) benchmark introduced by Li et al. (2024), together with its VirtualHome environment. VirtualHome is a household simulator containing rooms, objects, object identifiers, object states, spatial relations, and a symbolic action language (Puig et al., 2018). For example, a model may need to output actions such as `WALK`, `GRAB`, `OPEN`, `PUTIN`, or `SWITCHON` with object names and numeric identifiers. EAI is especially useful for this project because it separates embodied decision making into modules, including goal interpretation and action sequencing. This makes it possible to distinguish between a model that misunderstands the goal and a model that understands the goal but cannot generate a valid action sequence.

The central motivation for the project is the **goal-to-action gap**. Early analysis shows that a model may correctly identify the intended goal but still produce an incomplete or invalid action sequence. For instance, it may know that a book should be read, but output `READ book` without first walking to and grabbing the book. These failures are not only formatting mistakes; they are failures of grounded action generation.

The project is therefore both diagnostic and constructive. It first analyses where LLMs fail when moving from goals to executable actions. It then tests lightweight prompt and grounding interventions that aim to reduce those failures without fine-tuning the model. The work is relevant to data science because it combines benchmark evaluation, error taxonomy design, multi-model experimental analysis, prompt intervention testing, and reproducible data management.

---

## 2. Aim and Objectives

### Aim

The aim of this project is to investigate the goal-to-action gap in LLM-based embodied planning, and to evaluate whether lightweight prompt and grounding methods can improve action-sequencing success on the EAI/VirtualHome benchmark.

### Objectives

1. **Characterise the goal-to-action gap.** Measure whether goal interpretation and action sequencing behave as distinct skills across modern LLMs.
2. **Diagnose dominant failure modes.** Categorise action-sequencing failures into interpretable types such as parsing errors, hallucinated objects, missing steps, wrong order, and additional steps.
3. **Evaluate prompt-only interventions.** Test whether methods such as format constraints, few-shot examples, and plan-then-ground prompting improve task success.
4. **Evaluate knowledge-grounded interventions.** Test whether scene-graph retrieval (SG-RAG) and precondition-rule checking (PC-KG) reduce grounding and planning errors.
5. **Analyse the limits of more complex prompting.** Compare simple planning prompts against checklist-based and goal-conditioned prompt variants to determine whether added structure improves or harms action sequencing.
6. **Produce a reproducible experimental pipeline.** Ensure that prompts, model outputs, normalised outputs, evaluation summaries, figures, and reports can be traced back to their source files.

These objectives are scientific rather than purely functional. The goal is not only to build a working tool, but also to understand which failures are recoverable, which interventions help, and which assumptions about LLM self-correction do not hold in this benchmark.

---

## 3. Overview of Progress

Substantial progress has been made on both the experimental pipeline and the analysis. The EAI/VirtualHome benchmark has been set up locally, and existing outputs from 17 model snapshots have been organised into a diagnostic inventory. The project focuses mainly on two EAI modules: goal interpretation and action sequencing. Goal interpretation is treated as a control task, while action sequencing is the main target because it contains the clearest gap between understanding and execution.

A reproducible generation and evaluation pipeline has been implemented. The pipeline supports multiple model providers and output variants, normalises generated action sequences into the format expected by EAI, runs the EAI evaluator, and writes summaries into a consistent output structure. This has allowed controlled comparisons between baseline prompting, format-constrained prompting, few-shot prompting, plan-then-ground prompting, scene-graph retrieval, precondition-knowledge-graph checking, and newer goal-conditioned prompt variants.

The first key finding is that **plan-then-ground prompting** is the strongest single prompt-only intervention tested so far. It is not part of the original EAI paper. Rather, it is a lightweight prompt variant designed in this project, inspired by prior plan-and-act and reasoning-then-acting prompting strategies. On `DeepSeek-V4-Flash`, it improves action-sequencing task success from `75.58%` to `80.23%`, a gain of `+4.65` percentage points. Its main effect is a reduction in `missing_step` errors, which fall from `12.79%` to `8.14%`.

The second key finding is that **more complex prompt structures are not automatically better**. Two additional prompt methods were tested. `state_checklist_plan` asks the model to internally check final states and preconditions before producing JSON. It reaches `79.07%`, which improves over baseline but underperforms `plan_then_ground`. `goal_conditioned_scaffold` is inspired by ProgPrompt but adapted to EAI: instead of copying Python-like program prompts, it asks the model to map formal node, edge, and action goals to a minimal action skeleton. It reaches `79.65%`, again improving over baseline but not exceeding `plan_then_ground`. These results suggest that short internal planning is more reliable than longer checklists or heavier scaffolds for this benchmark.

The third key finding concerns knowledge-grounded interventions. SG-RAG helps the weakest model in the three-vendor panel, but hurts or slightly reduces performance on stronger models. This suggests that adding scene-graph context is not always useful; for stronger models, the extra structured prefix may act as distracting context. PC-KG self-checking performs worse than expected. Instead of improving action plans, the rewrite process reduces task success sharply, apparently because the model over-corrects partially correct drafts after receiving violation feedback. This is an important negative result because it questions the assumption that verifier feedback is always beneficial when routed through an LLM rewrite step.

The main work completed so far includes:

- a multi-model EAI/VirtualHome diagnostic inventory;
- a failure taxonomy and case-study analysis of goal-correct but action-failed examples;
- implementation of SG-RAG and PC-KG modules;
- prompt-variant generation and evaluation scripts;
- evaluation of baseline, prompt-only, knowledge-grounded, checklist, and goal-conditioned variants;
- draft thesis chapters covering introduction, experimental setup, failure diagnosis, improvement methods, and discussion.

---

## 4. Project Plan

The remaining project work focuses on strengthening the evidence, writing the dissertation, and preparing final submission materials. The plan below uses a week-by-week Gantt-style schedule. Completed work is marked with `████`; planned work is marked with `░░░░`.

| Task | Apr W3 | Apr W4 | May W1 | May W2 | May W3 | May W4 | Jun W1 | Jun W2 | Jun W3 | Jun W4 | Jul W1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Set up EAI/VirtualHome pipeline | ████ | ████ |  |  |  |  |  |  |  |  |  |
| Build model inventory and diagnostics |  | ████ | ████ |  |  |  |  |  |  |  |  |
| Failure taxonomy and case studies |  |  | ████ | ████ |  |  |  |  |  |  |  |
| Prompt-only intervention experiments |  |  |  | ████ | ████ | ████ |  |  |  |  |  |
| SG-RAG and PC-KG experiments |  |  |  | ████ | ████ |  |  |  |  |  |  |
| Checklist and goal-conditioned prompt ablations |  |  |  |  |  | ████ |  |  |  |  |  |
| Update results and figures |  |  |  |  |  |  | ░░░░ | ░░░░ |  |  |  |
| Dissertation writing and editing |  |  |  | ████ | ████ | ░░░░ | ░░░░ | ░░░░ | ░░░░ | ░░░░ |  |
| Final proofing and submission |  |  |  |  |  |  |  |  |  | ░░░░ | ░░░░ |

The immediate next step is to update the thesis narrative. `plan_then_ground` should be presented as the main prompt-based improvement method. `state_checklist_plan` and `goal_conditioned_scaffold` should be included as additional ablations showing that added prompt complexity does not necessarily improve task success.

---

## 5. Data Management Plan

### 5.1 Data collected and generated

The project uses benchmark data from EAI/VirtualHome, including natural-language task prompts, scene-graph information, gold symbolic goals, and gold action programs. It also generates derived research data: model outputs, normalised action sequences, evaluation summaries, error logs, diagnostic CSV files, figures, and report drafts. No personal or sensitive human-subject data is collected.

### 5.2 Documentation and metadata

Each generated output is stored with a model name, variant name, dataset name, and evaluation type. Important derived files include `summary.json`, `error_info.json`, diagnostic CSV files, and Markdown reports. Scripts are named according to their role, such as `generate_outputs.py`, `normalize_action_outputs.py`, `scene_graph_rag.py`, and `precondition_kg.py`. This file naming convention supports traceability from reported results back to raw outputs.

### 5.3 Storage and backup

Working files are stored locally in the project workspace under structured folders such as `analysis/`, `paper/`, `scripts/`, and `output/`. Large generated outputs are kept under `output/improvement_run/` and `output/diagnostics/`. Important written material is also exported to Word format when needed for supervisor review. The project should be backed up regularly to a secure cloud or university-supported storage service. API keys and private credentials must not be stored in the repository or report outputs.

### 5.4 Data quality and reproducibility

Data quality is maintained through deterministic settings where possible, including `temperature=0` for model generation and fixed evaluation scripts. Smoke tests using a dry-run provider are used to verify that the pipeline works before expensive model calls are made. Generated summaries are checked against original `summary.json` files before being included in reports. The same benchmark, prompt files, and normalisation scripts are used across variants to ensure fair comparison.

### 5.5 Sharing and preservation

The final dissertation can share source code, derived summaries, figures, and non-sensitive configuration files. Raw benchmark data should be shared only in accordance with the EAI/VirtualHome licence terms. Model outputs may be shared if allowed by provider terms and if no private information is included. The final preserved package should include: analysis scripts, prompt variants, summary CSV files, figures, report chapters, and instructions for reproducing the main tables.

### 5.6 Ethical and legal considerations

The project does not involve human participants or personal data. The main ethical considerations concern correct citation of benchmark datasets, responsible use of model provider outputs, and transparent reporting of negative results. The PC-KG result is reported as a negative finding rather than hidden, which improves scientific transparency. Any reused benchmark material, figures, or examples must be cited properly and not presented as original data collection.

---

## References

Li, M., Zhao, S., Wang, Q., Wang, K., Zhou, Y., Srivastava, S., Gokmen, C., Lee, T., Li, L. E., Zhang, R., Liu, W., Liang, P., Li, F.-F., Mao, J. and Wu, J. (2024). *Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making*. NeurIPS Datasets and Benchmarks.

Puig, X., Ra, K. K., Boben, M., Li, J., Wang, T., Fidler, S. and Torralba, A. (2018). VirtualHome: Simulating Household Activities via Programs. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv preprint arXiv:2210.03629.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Welleck, S., Majumder, B. P., Gupta, S., Yazdanbakhsh, A. and Clark, P. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *Advances in Neural Information Processing Systems*.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. and Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems*.

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X. and Zhou, D. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. *International Conference on Learning Representations*.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F. and Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*.

Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J. and Garg, A. (2023). ProgPrompt: Generating Situated Robot Task Plans using Large Language Models. *Autonomous Robots*.
