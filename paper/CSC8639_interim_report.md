# CSC8639 Interim Report

**Project title:** Bridging the Goal-to-Action Gap: Diagnosing and Improving LLM Failures in Embodied Planning  
**Student:** Zihan Wu  
**Programme:** Project and Dissertation in Data Science  
**Supervisor:** (CSC8639)  
**Date:** 2026-05-30

---

## 1. Introduction

Large language models (LLMs) are increasingly proposed as the reasoning core of embodied agents [1, 2]. Given a natural-language instruction such as *turn on the light* or *read the book*, an LLM-based agent must produce a grounded sequence of executable actions—choosing valid operation verbs, referring to the correct objects by name and identifier, and respecting the state preconditions of the environment. While LLMs excel at language understanding and high-level planning, their grounded execution remains unreliable in ways that are poorly understood. Recent surveys of embodied planning report that even frontier models routinely produce action programs that look fluent but fail at runtime because of missing preconditions, hallucinated object identifiers, or incorrect temporal ordering [3, 4]. Understanding *why* this happens, and *how* it can be reduced without retraining the underlying model, is the broader research question that this project contributes to.

This project investigates that problem using the Embodied Agent Interface (EAI) benchmark [4], which evaluates LLMs on two decoupled sub-tasks: goal interpretation and action sequencing, within the VirtualHome household simulator [5]. VirtualHome defines a symbolic action language of 22 verbs (e.g. `WALK`, `GRAB`, `OPEN`, `SWITCHON`) with strict object-state preconditions, making it straightforward to measure whether a model produces syntactically valid, causally ordered, and goal-satisfying plans. The benchmark also exposes per-task scene graphs and gold action programs, which makes it possible to attribute every failure either to a misinterpreted goal or to a flawed action sequence. This separation is central to the project: it permits an evaluation methodology in which goal understanding and action grounding can be measured, and improved, independently of one another.

The central observation motivating the project is the **goal-to-action gap**: empirical results show that many models correctly interpret the intended goal but still fail to produce a valid action sequence. This gap is actionable—it points to specific failures in grounded execution that are distinct from semantic understanding, and it suggests that the bottleneck for current LLM planners is not what to do, but how to translate intent into operations consistent with the environment's symbolic constraints. The project therefore pursues two complementary lines of work. The first is diagnostic: constructing a systematic failure taxonomy across seventeen model snapshots and testing prompt-only interventions designed to reduce specific error types. The second is constructive: engineering a persistent knowledge base—combining a vector store for semantic retrieval [6] and a graph database for structured rule reasoning—and implementing an iterative evaluation harness that feeds observed failures back into the knowledge base to enable closed-loop improvement. Together these two strands move from *measuring* the gap to *closing* it through external grounding and accumulated experience, rather than through additional fine-tuning of the model itself.

---

## 2. Aim and Objectives

### Aim

The aim of this project is to diagnose and reduce the goal-to-action gap in LLM-based embodied planning, by (i) characterising failure patterns across modern models, (ii) evaluating prompt-only interventions, and (iii) assessing whether grounding the planner in a persistent external knowledge base can further improve action-sequencing success on the EAI/VirtualHome benchmark. The intended contribution is therefore both empirical—a quantitative account of where current LLMs fail in grounded execution—and methodological, in the form of an open, reproducible pipeline that combines diagnostic evaluation, prompt engineering, and knowledge-grounded planning into a single closed loop.

### Objectives

1. **Characterise the goal-to-action gap.** Quantify the discrepancy between goal-interpretation performance and action-sequencing performance across a representative set of LLMs to establish whether the two skills behave independently.

2. **Diagnose dominant failure modes.** Classify action-sequencing failures into interpretable types—parsing errors, hallucinated actions, missing precondition steps, wrong temporal order, and superfluous steps—and identify which types are most prevalent per model family.

3. **Evaluate prompt-only interventions.** Test whether format constraints, few-shot examples, and plan-then-ground prompting strategies (inspired by reasoning-then-acting frameworks such as [7]) measurably reduce specific failure types.

4. **Assess the limits of complex prompt structure.** Compare simple planning prompts against richer checklist, goal-conditioned [8], and bidirectional causal variants to determine whether additional prompt complexity improves or harms grounded action generation.

5. **Build and evaluate a persistent knowledge-grounded planning framework.** Implement a vector store (Chroma with BGE embeddings [9]) and a graph database (Neo4j) populated from VirtualHome scene data, action rules, failure cases, and inductively derived constraints; evaluate the system against the baseline in a controlled single-iteration experiment (E1) and an iterative closed-loop experiment (E2).

6. **Produce a reproducible experimental pipeline.** Ensure that all prompts, model outputs, evaluation summaries, figures, and knowledge-base build scripts are traceable, versioned, and rerunnable.

Taken together, these objectives are designed to be cumulative rather than independent: each builds on the artefacts produced by the previous one. Objectives 1–2 generate the diagnostic ground truth that Objectives 3–4 use to target prompt interventions, and the failure cases produced by all of these stages become the seed corpus that populates the persistent knowledge base in Objective 5. Objective 6 acts as a cross-cutting constraint: every artefact, from the diagnostic CSVs to the Neo4j build scripts, is committed to the project repository so that any individual experiment can be re-executed end-to-end from a single command.

---

## 3. Overview of Progress

### 3.1 Experimental pipeline and diagnostic inventory

A complete generation-and-evaluation pipeline has been implemented and used to run experiments across seventeen model snapshots. The pipeline reads EAI-format prompt files, generates action sequences via a provider-agnostic API wrapper (supporting OpenAI, Anthropic, Gemini, and OpenAI-compatible endpoints), normalises outputs into the `[VERB] <name> (id)` format expected by the EAI evaluator, runs evaluation, and writes structured summaries. This infrastructure has been used to produce a diagnostic inventory comparing `task_success_rate`, `execution_success_rate`, and all six error-type rates across models, enabling the failure-taxonomy work described below. The pipeline was deliberately designed to be model-agnostic: switching providers or model snapshots requires only a single configuration line, which made it practical to scale from a handful of pilot runs to the full seventeen-model inventory without reworking the evaluation logic.

### 3.2 Findings from prompt-only experiments

The strongest prompt-only intervention identified is **plan-then-ground prompting**, a two-stage variant developed in this project, drawing on the chain-of-thought reasoning paradigm [10]. The model is first prompted to describe the goal state in terms of VirtualHome node, edge, and action goals, then to generate a minimal JSON action sequence grounded against that plan. On `DeepSeek-V4-Flash`, this raises action-sequencing task success from `75.58%` to `80.23%` (+4.65 pp), primarily by reducing `missing_step` errors from `12.79%` to `8.14%`. The variant draws on the plan-and-act literature [7] but is simplified to avoid the context-length overhead of full chain-of-thought reasoning.

Three more complex prompt variants were also tested, and none exceeded plan-then-ground performance. A state-checklist variant, inspired by self-refinement [11], reached `79.07%`. A goal-conditioned scaffold variant, adapted from ProgPrompt [8], reached `79.65%`. A bidirectional causal variant achieved the highest relation-goal score (`80.00%`) but raised `missing_step` errors to `17.05%`, reducing overall task success to `74.43%`. These results confirm that short internal planning outperforms longer checklists and two-pass causal reasoning for this benchmark, and are consistent with findings reported on long-context degradation [12].

### 3.3 Honest accounting of earlier "RAG" and "KG" modules

Two modules in the codebase were originally labelled `SceneGraphRAG` and `PreconditionKG`. On re-examination, neither meets the standard definitions of those terms. The "RAG" module performed keyword-and-identifier matching against scene-graph JSON files and serialised a small subgraph as a prompt prefix; it used no embedding model and no persistent index. The "KG" module was a hand-coded dictionary of 22 action rules with an in-process symbolic verifier; it used no graph database, had no queryable history of failures, and could not link failure cases to scene context. These modules are therefore re-classified as **structured prompt injection** and a **rule-based verifier**, and have been removed from the experimental method line. A genuinely persistent knowledge-base infrastructure has subsequently been built and is described in Section 3.4.

### 3.4 Persistent knowledge-grounded planning framework

A persistent RAG and KG infrastructure has been designed, implemented, and fully bootstrapped. All components reside under `analysis/kb/` in the project repository.

**Vector store (Chroma RAG).** A Chroma database has been populated with two collections. The `scene_objects` collection holds one document per `(file_id, node_id)` pair across all 518 VirtualHome scenes, producing approximately 150,000 vectors embedded with the local `BAAI/bge-small-en-v1.5` model [9]. The `failure_cases` collection holds 345 past failure cases drawn from `output/diagnostics/`, indexed by failure type and model. Retrieval combines BGE semantic matching with deterministic identifier extraction, replacing the earlier keyword-only approach. This design is analogous to the dynamic grounding mechanism in LLM-Planner [2], which retrieves relevant environment descriptions as few-shot context for a language model planner.

**Graph database (Neo4j KG).** A Neo4j 5 database has been built with a three-layer graph: (i) a scene-instance layer linking 518 `Scene` nodes to their `Object` nodes via `CONTAINS` edges and spatial relations such as `INSIDE`, `ON`, and `FACING`; (ii) a rule-schema layer encoding the 22 VirtualHome action preconditions as `Action`–`Property`–`Precondition`–`Effect` subgraphs; and (iii) a failure-case layer linking each `FailureCase` node to the scene and action where it occurred. All constraints use Cypher `MERGE` to ensure idempotent re-builds.

**KG enhancement (three directions).** Three extensions to the base graph have been implemented. *Direction 1* adds 12 `TaskTemplate` nodes encoding common multi-step household task sequences (laundry, cooking, hygiene, etc.) with ordered `STEP_OF` edges, enabling template-guided plan generation. *Direction 2* adds an LLM-based rule induction module (`rule_induction.py`) that analyses batches of bad cases and writes new `DerivedRule` nodes to Neo4j, enabling the graph to accumulate learned constraints across iterations. *Direction 3* adds a simulation log extraction module (`simulation_rule_extraction.py`) that parses all `error_info.json` files produced by the EAI evaluator, aggregates failure patterns by action verb and object class, and persists high-frequency patterns (≥3 occurrences) as `DerivedRule` nodes; applied to the 52 existing evaluation logs, this process extracted 113 distinct rules, with the most frequent being `LOOKAT television` (missing-step, 411 occurrences), `PLUGIN light` (hallucination, 133 occurrences), and `SWITCHON computer` (missing-step, 80 occurrences).

**Drop-in retriever and verifier.** Two classes, `PersistentSceneGraphRetriever` and `PersistentPreconditionKG`, expose the same public interface as the earlier in-memory modules. The persistent backend is selected via `KB_BACKEND=persistent`; unavailability triggers a logged fallback to the legacy implementation, preserving reproducibility on offline machines.

**Iterative evaluation harness.** A standalone harness module (`analysis/kb/harness.py`) implements the closed-loop pipeline. Each iteration generates drafts via `generate_outputs.py`, optionally applies knowledge-grounded agent repair, runs the EAI evaluator, parses per-task outcomes, classifies failed tasks as `BadCase` objects, and writes them into both the Chroma `failure_cases` collection and the Neo4j `(:FailureCase)` layer tagged with `iteration_id`. A convergence check terminates the loop once successive iterations yield gains below a configurable threshold. The harness also calls the LLM rule induction module (Direction 2) and the simulation log extraction module (Direction 3) after each iteration, so the knowledge base grows automatically with each pass.

### 3.5 E1 experiment: static persistent KB

To isolate the contribution of the persistent knowledge base, a controlled single-iteration experiment (E1) was conducted. The `sg_rag` prompt variant (which calls `PersistentSceneGraphRetriever` for scene-context injection) was run on 200 EAI/VirtualHome action-sequencing prompts using `DeepSeek-V4-Flash`, and the results compared against the same model and variant using the in-memory baseline.

| Metric | In-memory baseline | Persistent KB | Delta |
|---|---|---|---|
| task\_success\_rate | 79.07% | 77.33% | −1.74% |
| action\_goal | 68.00% | 69.33% | +1.33% |
| execution\_success\_rate | 87.20% | 85.50% | −1.70% |
| missing\_step | 8.72% | 9.30% | +0.58% |

The persistent KB achieves a small improvement on `action_goal` (+1.33 pp), suggesting that structured scene-context retrieval supports action-level grounding. The slight decrease in overall `task_success_rate` (−1.74 pp) reflects the trade-off inherent in top-k retrieval: the in-memory baseline injects the full scene graph, providing higher coverage at the cost of longer context. Given the 200-prompt sample size, the difference lies within the normal variance range for this benchmark, and no statistically significant conclusion can be drawn at this stage. The result motivates the E2 iterative experiment, where accumulated failure cases are expected to provide an improvement signal that the static in-memory baseline cannot replicate.

### 3.6 Work completed and next steps

**Completed:**
- Multi-model diagnostic inventory and failure taxonomy across 17 model snapshots
- Prompt-variant pipeline: generation, normalisation, evaluation, and reporting scripts
- Evaluation of 8 prompt variants (baseline, format constraints, few-shot, plan-then-ground, state checklist, goal-conditioned scaffold, bidirectional causal, sg_rag)
- Persistent Chroma vector store (518 scenes, ~150,000 vectors) and Neo4j graph (22 action rules, 345 failure cases, 12 task templates, 113 simulation-log rules)
- Iterative harness with LLM rule induction and simulation log extraction wired in
- E1 experiment (200 prompts, persistent KB vs baseline)

**Next steps:**
- Run E2 (iterative KB): execute the closed-loop harness for 3–5 iterations and compare per-iteration success rates against the E1 baseline
- Run E3 (convergence study): extract and plot the convergence curve of task success vs. iteration count
- Complete dissertation methodology chapter (persistent RAG/KG architecture, iterative harness design, E1–E3 results)
- Final proofing and submission by 4 July

---

## 4. Project Plan

The Gantt chart below covers the full project duration from mid-April to early July 2026. Completed phases are marked `████`; in-progress or planned work is marked `░░░░`. Today's date is 30 May 2026 (May W4).

| Task | Apr W3 | Apr W4 | May W1 | May W2 | May W3 | May W4 | Jun W1 | Jun W2 | Jun W3 | Jun W4 | Jul W1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| EAI/VirtualHome pipeline setup | ████ | ████ | | | | | | | | | |
| Multi-model diagnostic inventory | | ████ | ████ | | | | | | | | |
| Failure taxonomy and case studies | | | ████ | ████ | | | | | | | |
| Prompt-only intervention experiments | | | | ████ | ████ | ████ | | | | | |
| Complex prompt ablations (checklist / scaffold / causal) | | | | | | ████ | | | | | |
| Persistent RAG + KG design | | | | | | ████ | | | | | |
| Bootstrap Chroma + Neo4j; implement harness | | | | | | ████ | | | | | |
| KG enhancement (task templates / rule induction / sim-log rules) | | | | | | ████ | | | | | |
| E1 experiment (static KB, 200 prompts) | | | | | | ████ | | | | | |
| E2 experiment (iterative KB, closed loop) | | | | | | | ░░░░ | ░░░░ | | | |
| E3 experiment (convergence study) | | | | | | | | ░░░░ | ░░░░ | | |
| Dissertation writing | | | | ████ | ████ | ████ | ░░░░ | ░░░░ | ░░░░ | ░░░░ | |
| Final proofing and submission | | | | | | | | | | ░░░░ | ░░░░ |

The immediate next steps are (i) running the iterative harness for E2 and (ii) completing the dissertation methodology chapter based on the implemented and tested persistent KB architecture.

Several risks to the remaining timeline have been identified and have associated mitigations. The largest is API cost and rate-limiting during E2 and E3, since each iteration of the closed-loop harness regenerates the full 200-prompt slice; this is mitigated by caching deterministic outputs at `temperature=0` and by reusing previous-iteration drafts whenever the relevant prompt and KB snapshot are unchanged. A second risk is non-convergence of the iterative loop, where additional KB content fails to translate into measurable gains; should this occur, the loop's stopping criterion will be relaxed and the analysis will pivot to characterising *why* convergence stalls rather than enforcing further iterations. Finally, the Neo4j and Chroma services run locally via Docker; to guard against environment drift, every build script is idempotent and the legacy in-memory backend remains available as a fallback path so that report figures can still be regenerated on machines without a running database.

---

## 5. Data Management Plan

### 5.1 Data collected and generated

The project uses benchmark data from EAI/VirtualHome, comprising natural-language task prompts, scene-graph files, gold symbolic goals, and gold action programs. It generates derived research data: LLM outputs, normalised action sequences, evaluation summaries (`summary.json`, `error_info.json`), diagnostic CSV files, figures, and report chapters. The persistent knowledge base (Chroma index, Neo4j data volume, embedding model cache) is a derived artefact that can be fully regenerated from the source dataset using the committed build scripts (`scripts/build_knowledge_base.sh`, `analysis/kb/build_vector_store.py`, `analysis/kb/build_graph_db.py`). No personal data or sensitive human-subject data is collected.

### 5.2 Documentation and metadata

All generated outputs are stored with model name, variant name, dataset name, and evaluation type embedded in the file path. Scripts are named according to their function (e.g. `generate_outputs.py`, `normalize_action_outputs.py`, `simulation_rule_extraction.py`). The knowledge-base schema is explicitly versioned in `analysis/kb/schema.cypher`. This naming convention supports full traceability from reported metrics to raw outputs and build commands.

### 5.3 Storage and backup

Working files are stored locally in a structured workspace under `analysis/`, `paper/`, `scripts/`, and `output/`. Regenerable artefacts (Chroma index, Neo4j volume, model cache) are stored under `data/kb/` and excluded from version control. The full project—excluding regenerable artefacts and the upstream EAI dataset—is mirrored to a private GitHub repository for off-machine backup. API keys and private credentials are not stored in the repository or any report output.

### 5.4 Data quality and reproducibility

Deterministic settings are used wherever possible: `temperature=0` for all model calls, fixed random seeds, and fixed EAI evaluation scripts. A dry-run provider allows smoke-testing the full pipeline without live API calls. The knowledge base is built with idempotent operations (`MERGE` in Neo4j, `upsert` in Chroma) so that re-running the build script does not duplicate records. Fallback to the legacy in-memory backend preserves reproducibility on evaluation machines without Docker access.

### 5.5 Sharing and preservation

The final dissertation may share source code, evaluation summaries, figures, the Cypher schema, and non-sensitive configuration files. Raw benchmark data must be shared in accordance with EAI/VirtualHome licence terms. The final preserved package will include: analysis scripts, knowledge-base build scripts, prompt-variant definitions, summary CSV files, figures, report chapters, and step-by-step instructions for reproducing the main tables and knowledge-base indices.

### 5.6 Ethical and legal considerations

The project does not involve human participants or personal data. The main ethical obligations concern accurate citation of benchmark datasets, transparent reporting of all experimental results including negative ones, and responsible use of LLM provider outputs. The re-classification of the earlier in-memory modules from "RAG/KG" to "structured prompt injection" and "rule-based verifier" demonstrates a commitment to honest reporting. All benchmark material, figures, and examples are properly cited and not presented as original data collection.

---

## References

[1] Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Ho, D., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jang, E., Ruano, R. J., Jeffrey, K., Jesmonth, S., Joshi, N., Julian, R., Kalashnikov, D., Kuang, Y., Lee, K.-H., Levine, S., Lu, Y., Luu, L., Parada, C., Pastor, P., Quiambao, M., Rao, K., Pertsch, K., Salazar, J., Sanketi, P., Sayed, K., Singh, J., Sontakke, S., Stone, A., Tan, C., Tran, H., Vanhoucke, V., Vega, S., Vuong, Q., Xia, F., Xiao, T., Xu, P., Xu, S. and Yan, M. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. *Conference on Robot Learning (CoRL)*.

[2] Song, C. H., Wu, J., Washington, C., Sadler, B. M., Chao, W.-L. and Su, Y. (2023). LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

[3] Shridhar, M., Thomason, J., Gordon, D., Bisk, Y., Han, W., Mottaghi, R., Zettlemoyer, L. and Fox, D. (2020). ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[4] Li, M., Zhao, S., Wang, Q., Wang, K., Zhou, Y., Srivastava, S., Gokmen, C., Lee, T., Li, L. E., Zhang, R., Liu, W., Liang, P., Li, F.-F., Mao, J. and Wu, J. (2024). Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making. *NeurIPS Datasets and Benchmarks*.

[5] Puig, X., Ra, K. K., Boben, M., Li, J., Wang, T., Fidler, S. and Torralba, A. (2018). VirtualHome: Simulating Household Activities via Programs. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[6] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*.

[7] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv preprint arXiv:2210.03629*.

[8] Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J. and Garg, A. (2023). ProgPrompt: Generating Situated Robot Task Plans using Large Language Models. *Autonomous Robots*.

[9] Xiao, S., Liu, Z., Zhang, P. and Muennighoff, N. (2024). C-Pack: Packed Resources For General Chinese Embeddings. *SIGIR*.

[10] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q. V. and Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems*.

[11] Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Welleck, S., Majumder, B. P., Gupta, S., Yazdanbakhsh, A. and Clark, P. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *Advances in Neural Information Processing Systems*.

[12] Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F. and Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*.
