# Failure Case Studies

- selected cases: 5 (top-5 by goal F1)
- joined cases scanned: 342

These cases highlight failures where the model appeared to understand the
goal (high goal F1) but its action sequence still failed. Each block lists
the task, the failure category, the predicted action sequence taken from
`error_info.json`, and (when available) the gold action sequence from the
VirtualHome programs.

## Case 1: Turn on light (125_2)

- model: `gpt-4o-2024-05-13`
- goal F1: 1.0000
- task success: None
- execution success: None
- failure type: `relation_grounding` — Wrong object id or spatial relation prevented execution.
- evaluator hint: hallucination error

Predicted action sequence:

```text
{"WALK":["home_office","319"]}
{"PLUGIN":["light","411"]}
{"SWITCHON":["light","411"]}
```

Ground-truth action sequence (if available):

```text
(unavailable)
```

Evaluator log excerpt:

```text
, file 125_2 has hallucination error
```

## Case 2: Drink (510_1)

- model: `gpt-4o-2024-05-13`
- goal F1: 0.8000
- task success: None
- execution success: None
- failure type: `relation_grounding` — Wrong object id or spatial relation prevented execution.
- evaluator hint: missing_step

Predicted action sequence:

```text
[WALK] <dining_room> (201)
[GRAB] <drinking_glass> (1001)
[DRINK] <drinking_glass> (1001)
[DRINK] <drinking_glass> (1001)
```

Ground-truth action sequence (if available):

```text
(unavailable)
```

Evaluator log excerpt:

```text
2026-04-22 14:40:29 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - Encounter error: MISSING_STEP | 2026-04-22 14:40:29 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - failed_error_code=1 | 2026-04-22 14:40:29 - virtualhome_eval.simulation.evolving_graph.eval_utils - GOAL FAIL! Not found: {'from_id': 65, 'relation_type': 'HOLDS_RH', 'to_id': 1001}
```

## Case 3: Read book (163_1)

- model: `gpt-4o-2024-05-13`
- goal F1: 0.8000
- task success: None
- execution success: None
- failure type: `relation_grounding` — Wrong object id or spatial relation prevented execution.
- evaluator hint: missing_step

Predicted action sequence:

```text
[WALK] <home_office> (319)
[GRAB] <novel> (1000)
[READ] <novel> (1000)
```

Ground-truth action sequence (if available):

```text
(unavailable)
```

Evaluator log excerpt:

```text
2026-04-22 14:40:32 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - Encounter error: MISSING_STEP | 2026-04-22 14:40:32 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - failed_error_code=1 | 2026-04-22 14:40:32 - virtualhome_eval.simulation.evolving_graph.eval_utils - GOAL FAIL! Not found: {'from_id': 65, 'relation_type': 'HOLDS_RH', 'to_id': 1000}
```

## Case 4: Watch TV (548_2)

- model: `gpt-4o-2024-05-13`
- goal F1: 0.7500
- task success: None
- execution success: None
- failure type: `relation_grounding` — Wrong object id or spatial relation prevented execution.
- evaluator hint: missing_step

Predicted action sequence:

```text
[WALK] <television> (248)
[SWITCHON] <television> (248)
[LOOKAT] <television> (248)
```

Ground-truth action sequence (if available):

```text
(unavailable)
```

Evaluator log excerpt:

```text
2026-04-22 14:40:26 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - Encounter error: MISSING_STEP | 2026-04-22 14:40:26 - virtualhome_eval.evaluation.action_sequencing.scripts.evaluate_results - failed_error_code=1 | 2026-04-22 14:40:26 - virtualhome_eval.simulation.evolving_graph.eval_utils - GOAL FAIL! Not found: {'from_id': 65, 'relation_type': 'FACING', 'to_id': 248}
```

## Case 5: Write an email (996_2)

- model: `gpt-4o-2024-05-13`
- goal F1: 0.6667
- task success: None
- execution success: None
- failure type: `other` — Failure pattern not yet categorised.
- evaluator hint: n/a

Predicted action sequence:

```text
[WALK] <computer> (1001)
[TYPE] <computer> (1001)
[TYPE] <computer> (1001)
```

Ground-truth action sequence (if available):

```text
(unavailable)
```

Evaluator log excerpt:

```text
2026-04-22 14:40:07 - virtualhome_eval.simulation.evolving_graph.eval_utils - GOAL FAIL! Not matched: {'id': 1001, 'class_name': 'computer', 'state': 'ON'}
```

