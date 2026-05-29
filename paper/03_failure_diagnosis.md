# 5. Failure Diagnosis

The previous chapter showed that goal-interpretation F1 and
action-sequencing task success rate are correlated only weakly across the
seventeen models in our inventory: the best goal-interpretation model
(`o1-preview`, all-F1 0.4275) is not the best action sequencer
(`mistral-large-2402`, task success 0.7639). RQ1 therefore answers
"distinct skills". The natural follow-up, RQ2, is to look inside the
*goal-to-action gap*: when a model has correctly parsed the desired node
states, edge relations, and action goals, what kind of mistake does its
action sequence still make?

This chapter combines three sources of evidence to answer that question.
First, the per-task evaluator logs from the EAI runner, which record the
fine-grained error code raised by the simulator for every program (e.g.
`MISSING_STEP`, `WRONG_ORDER`, `ADDITIONAL_STEP`). Second, the
per-model `error_info.json` summaries that the runner emits alongside
its overall scores. Third, the joined view we build in
`analysis/link_goal_to_action_failures.py`, which aligns the goal-log
view of each task (`task`, `file_id`, `goal_f1`) with the action-log
view (`task_success`, `failure_type`, `raw_failure_text`) so that we
can scan all 342 task instances at once.

## 5.1 Failure Categories

We collapse EAI's raw error codes into five categories that are useful
both as paper labels and as targets for prompt interventions:

- `format_or_parsing` — output cannot be parsed into a legal action
  list (wrong JSON shape, free-form prose, missing brackets).
- `hallucination` — at least one referenced object name or numeric id
  does not exist in the scene graph supplied by the prompt.
- `relation_grounding` — objects exist, but the predicted action
  manipulates the wrong instance, attaches the wrong room, or violates
  a `HOLDS_RH` / `FACING` relation needed by the goal.
- `missing_step` — a required precondition action (`WALK`, `OPEN`,
  `GRAB`) is omitted, so a later action becomes non-executable.
- `wrong_order` — the right actions are present but in a sequence that
  fails preconditions (e.g. `PUTIN` before `OPEN`).

Mapping these categories onto the per-model failure profile in
`output/diagnostics/multimodel_failure_profile.csv` exposes three
distinct failure regimes, summarised in Table 5.1 (extracted from that
CSV; figures shown as percentage of all 305 EAI programs).

| Regime | Representative models | Dominant failure | Notes |
| --- | --- | --- | --- |
| Format-bottlenecked | `mixtral-8x22b-instruct-v0.1` (100% parsing), `gemini-1.0-pro` (76.07%), `llama-3-70b-chat` (37.70%), `gpt-3.5-turbo` (35.08%) | `format_or_parsing` | The model never gets a chance to be wrong about physics — it cannot emit a legal action list at all. |
| Reasoning-bottlenecked | `gpt-4o-2024-05-13` (25.25% missing_step), `gpt-4-turbo` (32.13%), `claude-3-haiku` (43.28%), `gemini-1.5-flash` (29.51%) | `missing_step` | Format is fine; the model omits a required precondition action, typically a `WALK` or a second `GRAB` when both hands are involved. |
| Hallucination-bottlenecked | `llama-3-8b-chat` (41.31% hallucination), `cohere-command-r` (28.52%), `claude-3-opus` (14.10%) | `hallucination` | The model invents object ids that are not in the prompt. This is the only regime where action_f1 can collapse even when goal F1 is moderate. |

Two patterns deserve emphasis. First, the strongest action sequencers
(`mistral-large`, `claude-3-5-sonnet`, `o1-preview`) are *not*
free of failure — they simply concentrate failure into `missing_step`
plus a small `additional_step` tail, while keeping `parsing` and
`hallucination` near zero. Second, the `wrong_order` channel is small
across the board (≤2.6% for every non-Cohere model). This argues
against the popular hypothesis that LLMs primarily fail at long-range
ordering. They fail mostly by *omission*, which is a different failure
mode and a different intervention target.

## 5.2 High Goal F1, Failed Action Sequence

To localise RQ2, we filter the 342 joined cases to those with
`goal_f1 ≥ 0.66` and `task_success = False`. The five top-ranked cases
(by goal F1, then by goal coverage) are saved in full in
`output/diagnostics/failure_case_studies.md`. We discuss three of them
here, chosen so each illustrates a distinct failure mechanism.

### Case A — Hallucinated precondition (`Turn on light`, `125_2`)

`gpt-4o-2024-05-13` correctly produces every node, edge and action goal
for `Turn on light` (goal F1 = 1.000). Its action sequence is, however,

```text
{"WALK":["home_office","319"]}
{"PLUGIN":["light","411"]}
{"SWITCHON":["light","411"]}
```

The simulator flags `hallucination error`. Inspecting the scene graph
for `125_2` confirms that `light(411)` exists, but it is *not* a
plug-in lamp; no `PLUGIN` action is admissible on it. The model has
imported a precondition pattern from a different scene (where a floor
lamp does need to be plugged in) and applied it on top of an otherwise
correct goal. This is the canonical failure mode predicted by RQ2: the
goal is right, but the model invents an extra precondition that
breaks executability. Format constraints cannot fix this; only a
verifier or a few-shot example with the legal action set can.

### Case B — Missing step (`Drink`, `510_1`)

For `Drink`, the goal interpretation is mostly right (goal F1 = 0.800).
The action sequence is

```text
[WALK] <dining_room> (201)
[GRAB] <drinking_glass> (1001)
[DRINK] <drinking_glass> (1001)
[DRINK] <drinking_glass> (1001)
```

The evaluator returns `MISSING_STEP` and `GOAL FAIL! Not found:
{from_id: 65, relation_type: HOLDS_RH, to_id: 1001}`. The character
node `65` never enters a `HOLDS_RH` relation with the glass because
there is no second-hand grasp action: VirtualHome's `DRINK` requires
the actor to be holding the target with a specific hand state, which
is not entered by `[GRAB]` alone in this scene. The model has
duplicated `[DRINK]` instead of inserting the missing precondition.
This is the dominant regime for `gpt-4o` (25.25% of programs) and for
the entire frontier-OpenAI / late-Gemini cluster.

### Case C — Goal not achieved despite executable plan (`Write an email`, `996_2`)

```text
[WALK] <computer> (1001)
[TYPE] <computer> (1001)
[TYPE] <computer> (1001)
```

This sequence is *executable* — the simulator does not raise an
error code. But the final state graph still does not contain
`{id: 1001, class_name: computer, state: ON}`, so `GOAL FAIL!` is
raised. The model omitted `[SWITCHON]` because in everyday language
"writing an email" implies the computer is already on; the goal
interpretation correctly listed the `ON` node-state, but the action
sequence relied on a commonsense default rather than enforcing it.
This case is interesting because it cannot be diagnosed from
`error_info.json` alone (no error code is raised); only the joined
view, which carries the goal-F1 signal across, surfaces it.

## 5.3 Distribution of Goal-correct Failures

Aggregating the joined view at the case level
(`output/diagnostics/failure_pattern_counts.json`):

```text
relation_grounding   59
planning_order       39
other                40
format_or_parsing     1
```

Among the 139 cases where the goal was at least partially correct but
the action sequence failed, **42% are relation-grounding errors** in
the broad sense (Case A and Case C above), **28% are
planning-order / missing-step errors** (Case B), and the residual
~30% are heterogeneous low-frequency failures including timeouts and
ambiguous task wording. Crucially, only one of these 139 cases is a
pure parsing failure — the remaining `format_or_parsing` mass
documented in Table 5.1 lives in models whose goal F1 is already low,
which is a different population.

This redistribution sharpens the answer to RQ2. *Conditional on the
model getting the goal right*, the goal-to-action gap is dominated by
two failure modes: hallucinating an extra precondition or relation
(Case A / C) and omitting a required precondition action (Case B).
Both are local edits to the action sequence, not deep planning
failures.

## 5.4 Implications for Interventions

The diagnostic above directly motivates the intervention taxonomy in
Chapter 6:

- `format_constraints` and `schema_constrained` (for goals) target the
  *format-bottlenecked* regime in Table 5.1. They are expected to
  collapse the parsing column for `gpt-3.5-turbo`, `gemini-1.0-pro`
  and the open-weight Llama / Mixtral cluster, but not to move
  frontier OpenAI / Anthropic numbers.
- `few_shot_valid_actions` targets the hallucination column, since
  every demonstration is itself a legal VirtualHome trace.
- `self_check_rewrite` and `plan_then_ground` target the missing-step
  column, because both inject a second pass that can re-check
  preconditions before emitting the final sequence.

We deliberately separate these expectations from the empirical
ablation (Section 6.2) so that the chapter can report, for each
failure category, whether the prediction held. A negative result —
for example, a `format_constraints` variant that improves parsing but
*also* introduces wrong-order errors — is itself a contribution,
since it shows where the goal-to-action gap is recoverable by
prompting and where it is not.

## 5.5 Chapter Summary

- The dominant failure mode across mid-tier and frontier models is
  `missing_step`, not `wrong_order`; long-range ordering is a small
  contributor.
- Open-weight and older proprietary models live in a separate
  *format-bottlenecked* regime; they should not be lumped into the
  same column as `gpt-4o` when reporting overall task success.
- When a model already understands the goal, ~70% of remaining
  failures are either a hallucinated precondition or an omitted
  precondition action; both are local interventions away from being
  fixable.
- These three observations set the agenda for Chapter 6: rather than
  expanding model count, we test whether targeted prompting can move
  these specific columns of the failure profile.
