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

The project is therefore both diagnostic and constructive. It first analyses where LLMs fail when moving from goals to executable actions. It then tests lightweight prompt interventions, and lays out a concrete plan for moving from prompt-only methods to a knowledge-grounded planning framework backed by a real vector database and graph database. The work is relevant to data science because it combines benchmark evaluation, error taxonomy design, multi-model experimental analysis, prompt intervention testing, knowledge-base engineering, and reproducible data management.

---

## 2. Aim and Objectives

### Aim

The aim of this project is to investigate the goal-to-action gap in LLM-based embodied planning, and to evaluate (i) whether lightweight prompt methods can improve action-sequencing success on the EAI/VirtualHome benchmark, and (ii) whether grounding LLM planning in a persistent external knowledge base (vector store + graph database) can further close that gap.

### Objectives

1. **Characterise the goal-to-action gap.** Measure whether goal interpretation and action sequencing behave as distinct skills across modern LLMs.
2. **Diagnose dominant failure modes.** Categorise action-sequencing failures into interpretable types such as parsing errors, hallucinated objects, missing steps, wrong order, and additional steps.
3. **Evaluate prompt-only interventions.** Test whether methods such as format constraints, few-shot examples, and plan-then-ground prompting improve task success.
4. **Analyse the limits of more complex prompting.** Compare simple planning prompts against checklist-based, goal-conditioned, and bidirectional-causal prompt variants to determine whether added prompt structure improves or harms action sequencing.
5. **Design a persistent knowledge-grounded planning framework with an iterative harness.** Specify and implement (i) an external knowledge-base infrastructure (vector store + graph database) capable of supporting scene-aware retrieval, action-precondition reasoning, and lookup of past failure cases, and (ii) an evaluation harness that automatically feeds every generated bad case back into the knowledge base, enabling a closed-loop iterative-improvement cycle. This is positioned as the next-stage method following the prompt-only experiments.
6. **Produce a reproducible experimental pipeline.** Ensure that prompts, model outputs, normalised outputs, evaluation summaries, figures, and reports can be traced back to their source files.

These objectives are scientific rather than purely functional. The goal is not only to build a working tool, but also to understand which failures are recoverable, which prompt-level interventions help, and what kind of external knowledge infrastructure is required to push past the prompt-only ceiling.

---

## 3. Overview of Progress

### 3.1 Pipeline and diagnostics

Substantial progress has been made on both the experimental pipeline and the analysis. The EAI/VirtualHome benchmark has been set up locally, and existing outputs from 17 model snapshots have been organised into a diagnostic inventory. The project focuses mainly on two EAI modules: goal interpretation and action sequencing. Goal interpretation is treated as a control task, while action sequencing is the main target because it contains the clearest gap between understanding and execution.

A reproducible generation and evaluation pipeline has been implemented. The pipeline supports multiple model providers and output variants, normalises generated action sequences into the format expected by EAI, runs the EAI evaluator, and writes summaries into a consistent output structure. This has allowed controlled comparisons between baseline prompting, format-constrained prompting, few-shot prompting, plan-then-ground prompting, and several newer goal-conditioned and bidirectional prompt variants.

### 3.2 Findings from prompt-only experiments

The first key finding is that **plan-then-ground prompting** is the strongest single prompt-only intervention tested so far. It is not part of the original EAI paper. Rather, it is a lightweight prompt variant designed in this project, inspired by prior plan-and-act and reasoning-then-acting prompting strategies. On `DeepSeek-V4-Flash`, it improves action-sequencing task success from `75.58%` to `80.23%`, a gain of `+4.65` percentage points. Its main effect is a reduction in `missing_step` errors, which fall from `12.79%` to `8.14%`.

The second key finding is that **more complex prompt structures are not automatically better**. Three additional prompt methods were tested. `state_checklist_plan` asks the model to internally check final states and preconditions before producing JSON; it reaches `79.07%`, which improves over baseline but underperforms `plan_then_ground`. `goal_conditioned_scaffold` is inspired by ProgPrompt but adapted to EAI: instead of copying Python-like program prompts, it asks the model to map formal node, edge, and action goals to a minimal action skeleton; it reaches `79.65%`, again improving over baseline but not exceeding the main method. `bidirectional_causal_planning` combines goal-back-tracking with forward sequencing; it raises the relation-goal score sharply (`64.41%` → `80.00%`) but causes missing steps to rise (`12.79%` → `17.05%`) and overall task success to fall (`74.43%`). These results suggest that short internal planning is more reliable than longer checklists, scaffolds, or two-pass causal reasoning for this benchmark.

### 3.3 Honest accounting of earlier "RAG" and "KG" modules

Two earlier modules in the codebase were originally labelled `Scene Graph RAG` and `Precondition KG`. After re-examination, those implementations do not meet the standard definitions of retrieval-augmented generation or a knowledge graph. The "RAG" module performed keyword-and-id matching against scene-graph JSON files and serialised a small subgraph as a prompt prefix; it had no embedding model, no vector store, and no persistent index. The "KG" module was a hand-coded dictionary of action rules together with an in-process symbolic verifier; it had no graph database, no persistent storage, and no ability to query historical failure cases. These earlier modules are therefore best described as **structured prompt injection** and a **rule-based verifier**, not as RAG or KG. Following supervisor feedback, they have been removed from the experimental method line, and a real persistent knowledge-base infrastructure has been designed as the next-stage method (Section 3.4).

### 3.4 Designed next-stage method: persistent knowledge-grounded planning framework

To genuinely move beyond prompt-only methods, the next stage of the project is to ground the planner in a persistent external knowledge base. The full engineering scheme has been designed and committed to the project repository (under `analysis/kb/`); bootstrap of the live services is the immediate next task. The design has the following components.

**Vector store (RAG).** A persistent Chroma database stores two collections. The `scene_objects` collection holds one document per `(file_id, node_id)` pair across all 518 VirtualHome scenes (~150 000 vectors at ~287 nodes per scene), embedded with the local `BAAI/bge-small-en-v1.5` model. The `failure_cases` collection holds one document per past LLM failure recorded in `output/diagnostics/`, indexed by `failure_type` and `model`. Retrieval combines semantic seeding (BGE embeddings of the task prompt) with deterministic id-extraction, replacing the earlier keyword-only matching.

**Graph database (KG).** A Neo4j 5 database (run via Docker) stores a three-layer graph. The first layer is the **scene-instance layer** where each `Scene` node connects to its `Object` nodes via `CONTAINS`, and `Object`–`RELATION`–`Object` edges encode the original VirtualHome relations such as `INSIDE`, `ON`, `CLOSE`, and `FACING`. The second layer is the **rule-schema layer** where each `Action` node connects to `Property`, `Precondition`, and `Effect` nodes derived from the 22 VirtualHome action rules. The third layer is the **failure-case layer** where each `FailureCase` node is linked to the `Scene` in which it occurred and to the `Action` whose preconditions it violated. Cypher constraints on `(Scene.file_id)`, `(Object.file_id, Object.node_id)`, and `(Action.name)` keep the graph keyed and idempotent across re-builds.

**Drop-in retriever and verifier.** Two new classes, `PersistentSceneGraphRetriever` and `PersistentPreconditionKG`, expose the same public interface as the earlier modules so that all downstream callers (planner, evaluator, reports) work without modification. The persistent backend is selected at runtime via the environment variable `KB_BACKEND=persistent`; if the backend is unreachable the system logs a warning and falls back to the legacy in-memory implementation, which preserves reproducibility on offline evaluation machines.

**Planning agent v2.** With the persistent infrastructure in place, the planning agent will: (i) embed the task prompt and query Chroma for the most relevant `scene_objects` in the current scene, (ii) expand them via Neo4j `RELATION*0..k` Cypher traversal to obtain a compact subgraph, (iii) invoke the LLM under a `plan_then_ground` prompt to produce a draft, (iv) verify the draft against the rule-schema layer in Neo4j, and (v) when violations occur, query the failure-case layer for similar past failures and inject them as few-shot exemplars into a conservative local-repair prompt. Steps (iv) and (v) are the genuinely new capability that the earlier in-memory verifier could not provide, because they require a queryable history of failures linked to actions and scenes.

**Iterative evaluation harness.** A standalone harness (`analysis/kb/harness.py`) will close the loop between the planning agent and the knowledge base. Each iteration runs the agent on the EAI action-sequencing test split, parses the per-task evaluator summary, identifies every failed task as a `BadCase`, and writes the bad cases back into both the Chroma `failure_cases` collection and the Neo4j `(:FailureCase)` layer (tagged with an `iteration_id`). On the next iteration, the agent retrieves these freshly added failures when it encounters semantically similar tasks, and uses them as few-shot exemplars in the repair step. The harness reports per-iteration metrics (task success, number of newly fixed cases, number of newly failed cases, cumulative success delta) and provides a convergence test that stops once successive iterations yield gains below a configurable threshold. The original benchmark gold sequences are never modified; only LLM-generated drafts and their associated violation codes flow into the knowledge base, which keeps the EAI evaluator's ground truth clean.

This iterative-harness design supports three planned experiments. **E1 (static KB)** builds the knowledge base once from the existing `output/diagnostics/` failures and runs a single evaluation, isolating the contribution of the persistent vector store and graph database. **E2 (iterative KB)** runs the full closed loop for several iterations, isolating the contribution of bad-case feedback. **E3 (convergence study)** tracks task success, repaired-failure count, and newly-introduced-failure count per iteration, producing a convergence curve and identifying the point at which marginal gain falls below a chosen threshold. Together, E1–E3 turn the project's evaluation from a one-shot benchmark sweep into a reproducible, auditable diagnostic-then-repair cycle.

### 3.5 Work completed and work outstanding

Completed:

- a multi-model EAI/VirtualHome diagnostic inventory;
- a failure taxonomy and case-study analysis of goal-correct but action-failed examples;
- prompt-variant generation, normalisation, and evaluation scripts;
- evaluation of baseline, format-constrained, few-shot, plan-then-ground, checklist, goal-conditioned, and bidirectional prompt variants;
- design and implementation of the persistent RAG + KG infrastructure and an iterative evaluation-harness skeleton (build scripts, drop-in retriever and verifier, schema definition, harness module, Docker bootstrap, dependency manifest), all committed to the project repository;
- draft thesis chapters covering introduction, experimental setup, failure diagnosis, prompt methods, and discussion.

Outstanding:

- bootstrap the live Chroma index and Neo4j graph (install Docker, install `chromadb` / `sentence-transformers` / `neo4j-driver`, run `scripts/build_knowledge_base.sh`);
- replace the harness `run_iteration` stub with real `generate_outputs.py` + EAI evaluator calls and verify a single closed-loop pass on the dry-run provider;
- implement the failure-case retrieval and few-shot injection step inside the v2 planning agent;
- run experiments E1 (static KB), E2 (iterative KB), and E3 (convergence study) and update the result tables;
- write the persistent RAG / KG / harness section into the methodology chapter and update the discussion accordingly.

---

## 4. Project Plan

The remaining project work focuses on bringing up the persistent knowledge base, running the v2 planning agent, writing the dissertation, and preparing final submission materials. The plan below uses a week-by-week Gantt-style schedule. Completed work is marked with `████`; planned work is marked with `░░░░`.

| Task | Apr W3 | Apr W4 | May W1 | May W2 | May W3 | May W4 | Jun W1 | Jun W2 | Jun W3 | Jun W4 | Jul W1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Set up EAI/VirtualHome pipeline | ████ | ████ |  |  |  |  |  |  |  |  |  |
| Build model inventory and diagnostics |  | ████ | ████ |  |  |  |  |  |  |  |  |
| Failure taxonomy and case studies |  |  | ████ | ████ |  |  |  |  |  |  |  |
| Prompt-only intervention experiments |  |  |  | ████ | ████ | ████ |  |  |  |  |  |
| Checklist / goal-conditioned / bidirectional prompt ablations |  |  |  |  |  | ████ |  |  |  |  |  |
| Design persistent RAG + KG infrastructure |  |  |  |  |  | ████ |  |  |  |  |  |
| Bootstrap Chroma and Neo4j; implement v2 agent and harness |  |  |  |  |  |  | ░░░░ | ░░░░ |  |  |  |
| Run E1 / E2 / E3 (static, iterative, convergence) experiments |  |  |  |  |  |  |  | ░░░░ | ░░░░ |  |  |
| Dissertation writing and editing |  |  |  | ████ | ████ | ░░░░ | ░░░░ | ░░░░ | ░░░░ | ░░░░ |  |
| Final proofing and submission |  |  |  |  |  |  |  |  |  | ░░░░ | ░░░░ |

The immediate next steps are: (i) install Docker and the knowledge-base dependencies, (ii) run `scripts/build_knowledge_base.sh` to populate the Chroma vector store and the Neo4j graph, (iii) wire the failure-case retrieval step into the v2 planning agent, and (iv) update the methodology chapter to describe the persistent RAG and KG architecture explicitly.

---

## 5. Data Management Plan

### 5.1 Data collected and generated

The project uses benchmark data from EAI/VirtualHome, including natural-language task prompts, scene-graph information, gold symbolic goals, and gold action programs. It also generates derived research data: model outputs, normalised action sequences, evaluation summaries, error logs, diagnostic CSV files, figures, and report drafts. The next stage will additionally generate a persistent Chroma index and a Neo4j graph; both are derived artefacts that can be regenerated from the source dataset using committed build scripts. No personal or sensitive human-subject data is collected.

### 5.2 Documentation and metadata

Each generated output is stored with a model name, variant name, dataset name, and evaluation type. Important derived files include `summary.json`, `error_info.json`, diagnostic CSV files, and Markdown reports. Scripts are named according to their role, such as `generate_outputs.py`, `normalize_action_outputs.py`, and `build_kg_planning_agent.py`. The persistent knowledge-base layer is encapsulated in `analysis/kb/`, with explicit build scripts (`build_vector_store.py`, `build_graph_db.py`) and a Cypher schema file (`schema.cypher`). This file naming convention supports traceability from reported results back to raw outputs and to the knowledge-base build commands.

### 5.3 Storage and backup

Working files are stored locally in the project workspace under structured folders such as `analysis/`, `paper/`, `scripts/`, and `output/`. Large generated outputs are kept under `output/improvement_run/` and `output/diagnostics/`. Knowledge-base artefacts (Chroma index, Neo4j data volume, embedding model cache) are kept under `data/kb/` and are excluded from version control because they are deterministically rebuildable. The full project, excluding regenerable artefacts and the upstream EAI dataset, is mirrored to a private GitHub repository for off-machine backup. Important written material is also exported to Word format when needed for supervisor review. API keys and private credentials are not stored in the repository or report outputs.

### 5.4 Data quality and reproducibility

Data quality is maintained through deterministic settings where possible, including `temperature=0` for model generation and fixed evaluation scripts. Smoke tests using a dry-run provider are used to verify that the pipeline works before expensive model calls are made. Generated summaries are checked against original `summary.json` files before being included in reports. The persistent knowledge base is built with idempotent operations (Chroma `upsert` and Neo4j `MERGE`), so re-running the build script does not produce duplicate records. The system can also fall back to the legacy in-memory implementation when the persistent backend is unavailable, which preserves reproducibility on offline evaluation environments.

### 5.5 Sharing and preservation

The final dissertation can share source code, derived summaries, figures, the Cypher schema, and non-sensitive configuration files. Raw benchmark data should be shared only in accordance with the EAI/VirtualHome licence terms. Model outputs may be shared if allowed by provider terms and if no private information is included. The final preserved package should include: analysis scripts, knowledge-base build scripts, prompt variants, summary CSV files, figures, report chapters, and instructions for reproducing the main tables and the persistent RAG / KG.

### 5.6 Ethical and legal considerations

The project does not involve human participants or personal data. The main ethical considerations concern correct citation of benchmark datasets, responsible use of model provider outputs, and transparent reporting of negative results. The earlier "RAG" and "KG" modules have been openly re-classified as structured prompt injection and a rule-based verifier, rather than presented as full RAG / KG; this honest re-framing is itself an example of transparent reporting. Any reused benchmark material, figures, or examples must be cited properly and not presented as original data collection.

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

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*.

Xiao, S., Liu, Z., Zhang, P. and Muennighoff, N. (2024). C-Pack: Packed Resources For General Chinese Embeddings. *SIGIR*.
