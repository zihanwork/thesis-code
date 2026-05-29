# Goal-to-Action Diagnostic Cases

- goal threshold: 0.5
- top_n: 3
- joined cases: 342
- selected diagnostic cases: 3

## Cases

### Case 1
- Task: Turn on light
- file_id: 125_2
- Goal F1: 1.0
- Task success: None
- Execution success: None
- Failure type: relation_grounding
- Evidence: , file 125_2 has hallucination error

### Case 2
- Task: Drink
- file_id: 510_1
- Goal F1: 0.8
- Task success: None
- Execution success: None
- Failure type: relation_grounding
- Evidence: 2026-04-22 14:40:29 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - Encounter error: MISSING_STEP | 2026-04-22 14:40:29 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - failed_error_code=1 | 2026-04-22 14:40:29 - virtualhome_eval.simulation.evolvi

### Case 3
- Task: Read book
- file_id: 163_1
- Goal F1: 0.8
- Task success: None
- Execution success: None
- Failure type: relation_grounding
- Evidence: 2026-04-22 14:40:32 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - Encounter error: MISSING_STEP | 2026-04-22 14:40:32 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - failed_error_code=1 | 2026-04-22 14:40:32 - virtualhome_eval.simulation.evolvi

## Failure Pattern Counts

- relation_grounding: 59
- other: 40
- planning_order: 39
- format_or_parsing: 1
