# From Goal Understanding to Action Failure
## Meeting Update Report (Filled Version)

### 1. Basic Information
- Tentative Thesis Title: From Goal Understanding to Action Failure: A Diagnostic Study of LLMs for Embodied Decision Making
- Meeting Type: Hybrid meeting
- Date: 4.20
- Presenter: wu zihan
- Current Stage: Baseline reproduction completed, downstream diagnosis started

### 2. This Week's Core Update
#### 2.1 One-sentence update
I have reproduced the EAI goal interpretation baseline, completed one downstream action sequencing run, and started diagnosing why goal-level understanding does not transfer to executable actions.

#### 2.2 What I focused on
- Read EAI framework and re-check module interfaces
- Run `goal_interpretation` and organize multi-model baseline outputs
- Run `action_sequencing` and collect downstream logs
- Build a tiny diagnostic pipeline linking goal-level and action-level outcomes

#### 2.3 What has been completed
- Environment setup and CLI reproduction completed (`conda + Python 3.8 + eai-eval`)
- Goal interpretation evaluation completed and ranked
- Action sequencing evaluation completed for VirtualHome with official HELM responses
- Diagnostic scripts completed: run / summarize / link-and-diagnose

#### 2.4 What is still in progress
- Distinguish formatting/parsing failures from true planning-reasoning failures
- Add a cleaner per-model case alignment to avoid unknown-model cases
- Expand diagnosis beyond one downstream module

### 3. Research Direction
#### 3.1 Main research question
When LLMs appear to understand the goal, why do they still fail to generate executable or successful action plans?

#### 3.2 Motivation
- Overall success rate alone is insufficient for failure diagnosis
- Embodied pipelines contain multiple failure points after goal understanding
- Need module-level and case-level diagnosis

#### 3.3 Current scope
- Benchmark/framework: EAI
- Simulator: VirtualHome
- Main module: Goal Interpretation
- Downstream module: Action Sequencing
- Diagnostic focus: relation grounding, implicit preconditions, planning/executability

#### 3.4 Expected contribution
Provide a diagnostic bridge from goal understanding to downstream action failure, rather than only reporting benchmark-level success rates.

### 4. Benchmark / Codebase Progress
#### 4.1 Environment and code setup
- Repository used: https://github.com/embodied-agent-interface/embodied-agent-interface?tab=readme-ov-file
- Main scripts: `scripts/run_action_sequencing_eval.sh`, `analysis/summarize_downstream.py`, `analysis/link_goal_to_action_failures.py`
- Output structure: `output/virtualhome/evaluate_results/{goal_interpretation,action_sequencing}/<model>/summary.json`
- Diagnostics output: `output/diagnostics/`

#### 4.2 Modules located
- Goal Interpretation: completed
- Action Sequencing: completed
- Subgoal Decomposition: located, not yet diagnosed
- Transition Modeling: located, not yet diagnosed

#### 4.3 Current run status
- Goal Interpretation / VirtualHome / 18 models / completed
- Action Sequencing / VirtualHome / 18 models / completed (with severe formatting/parsing issues)

### 5. Baseline Results
#### 5.1 Goal Interpretation baseline summary (Top-5 by all_f1)
- Rank 1: o1-preview-2024-09-12 | all_f1=42.7462, all_precision=31.7547, all_recall=65.3750
- Rank 2: cohere-command-r | all_f1=36.6883, all_precision=27.3608, all_recall=55.6650
- Rank 3: gpt-4o-2024-05-13 | all_f1=36.5318, all_precision=26.4067, all_recall=59.2500
- Rank 4: gemini-1.5-pro-preview-0409 | all_f1=36.2242, all_precision=33.5886, all_recall=39.3086
- Rank 5: gpt-4-turbo-2024-04-09 | all_f1=33.2433, all_precision=24.0379, all_recall=53.8750

#### 5.2 Key observations from goal interpretation
- Many models show recall > precision, implying over-generation
- Node/edge/action metrics are imbalanced across models
- A few models have abnormal zero outputs and need input-format checks

#### 5.3 Downstream module baseline summary (Action Sequencing)
- Models evaluated: 18; models with task_success_rate = 0: 18
- claude-3-5-sonnet-20240620: task_success_rate=0.0, execution_success_rate=0.0, grammar.parsing=100.0
- claude-3-haiku-20240307: task_success_rate=0.0, execution_success_rate=0.0, grammar.parsing=100.0
- claude-3-opus-20240229: task_success_rate=0.0, execution_success_rate=0.0, grammar.parsing=100.0

#### 5.4 Key observations from downstream module
- Current downstream failures are dominated by format/parsing issues
- Many logs indicate: `Action WALK does not follow name_id format`
- This means immediate failures may happen before deep planning logic is evaluated

### 6. Very Small Diagnostic Slicing
#### 6.1 Diagnostic question
Which cases look correct at the goal level, but still fail at the action level?

#### 6.2 Case summary (Top-3)
- Case 1:
  - Task: Pet cat
  - file_id: 203_2
  - Goal indicator (goal_f1): 0.8
  - Goal interpretation judgment: relatively acceptable at case-level metric
  - Downstream action outcome: no executable prediction
  - Failure type: format_or_parsing
  - Diagnostic note: 2026-04-21 00:10:32 - virtualhome_eval.simulation.evolving_graph.eval_utils -   Action WALK does not follow name_id format | , file 203_2 prediction has no prediction
- Case 2:
  - Task: Drink
  - file_id: 156_1
  - Goal indicator (goal_f1): 0.8
  - Goal interpretation judgment: relatively acceptable at case-level metric
  - Downstream action outcome: no executable prediction
  - Failure type: format_or_parsing
  - Diagnostic note: 2026-04-21 00:10:34 - virtualhome_eval.simulation.evolving_graph.eval_utils -   Action WALK does not follow name_id format | , file 156_1 prediction has no prediction
- Case 3:
  - Task: Relax on sofa
  - file_id: 137_1
  - Goal indicator (goal_f1): 0.8
  - Goal interpretation judgment: relatively acceptable at case-level metric
  - Downstream action outcome: no executable prediction
  - Failure type: format_or_parsing
  - Diagnostic note: 2026-04-21 00:10:37 - virtualhome_eval.simulation.evolving_graph.eval_utils -   Action WALK does not follow name_id format | , file 137_1 prediction has no prediction

#### 6.3 Emerging failure patterns
- format_or_parsing: 305
- other: 37

### 7. Current Hypothesis
#### 7.1 Working hypothesis
LLM failures in embodied decision making may come not only from goal misunderstanding but also from relation grounding, implicit preconditions, and executable action formatting/planning.

#### 7.2 Evidence I currently have
- EAI emphasizes module-level diagnosis rather than overall success only
- Current runs show a clear gap between goal-level scores and action-level executability
- Parsing-format failures are frequent in action_sequencing logs

#### 7.3 Evidence I still need
- Controlled experiments separating format errors from planning errors
- More cases showing high goal quality but downstream runtime failure categories
- Extension to another module (subgoal decomposition or transition modeling)

### 8. Questions to Discuss in the Meeting
1. Should the thesis scope stay focused on `goal_interpretation + action_sequencing` for depth first?
2. For current all-zero downstream results, should we first normalize output action format before deeper diagnosis?
3. Is VirtualHome enough for phase-1, with BEHAVIOR moved to phase-2 validation?
4. Is the best framing a diagnostic thesis rather than full benchmark reproduction?

