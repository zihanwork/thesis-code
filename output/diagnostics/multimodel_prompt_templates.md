# Multi-model Prompt and Ablation Templates

这些模板用于下一轮成功率提升实验。所有模板都应保持相同样本、相同 temperature、相同 max token，并输出到独立 `<model>_<variant>_outputs.json`，方便和 baseline 做消融对比。

## Variant 1: Baseline

使用 EAI 原始 `llm_prompt`，仅加最小 system message，作为可比基线。

```text
You output ONLY the VirtualHome action sequence requested by the user.
Do not include explanations, markdown, or extra text.
```

## Variant 2: Format Constraints

目标是降低 `parsing` 和 `predicate_argument_number`。

```text
You output ONLY a compact JSON action sequence for VirtualHome.
Format: concatenate one or more JSON objects with no separator.
Example: {"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}
Rules:
- Action names must be uppercase.
- Each value must be a JSON array of strings.
- Parameters must alternate object_name and numeric id.
- One-object actions use 2 strings; two-object actions use 4 strings.
- STANDUP uses an empty array [].
- Do not wrap the answer in markdown.
```

## Variant 3: Few-shot Valid Actions

目标是减少 hallucinated action 和参数数量错误。示例应来自合法轨迹或 oracle 转换结果，避免引入不可执行动作。

```text
Example 1:
Task: Turn on light
Answer: {"WALK":["floor_lamp","1000"]}{"SWITCHON":["floor_lamp","1000"]}

Example 2:
Task: Sit on chair
Answer: {"WALK":["chair","245"]}{"SIT":["chair","245"]}

Now solve the user's task with the same output-only format.
```

## Variant 4: Self-check Rewrite

目标是降低 `missing_step`、`wrong_order` 和 `additional_step`。实现时建议两次调用：第一次生成草稿，第二次只输出修正版。

```text
Review the draft action sequence against these checks:
1. Every action is a valid VirtualHome action from the prompt.
2. Every object uses an object name and numeric id available in the prompt.
3. Preconditions are satisfied before each action.
4. The sequence contains necessary steps but no redundant repetitions.
5. The final output still follows the compact JSON object concatenation format.

Return ONLY the corrected final action sequence.
```

## Variant 5: Plan Then Ground

目标是改善长程任务的动作顺序和 relation grounding。实现时第一步生成高层计划，第二步把计划压缩成 EAI 合法动作格式；只保存第二步输出供评测。

```text
First identify the high-level steps needed to satisfy the task.
Then convert those steps into executable VirtualHome actions using only objects and ids from the prompt.
Final answer must contain ONLY the compact JSON action sequence.
Do not include the high-level plan in the final answer.
```

## Variant Naming

- `<model>_baseline_outputs.json`
- `<model>_format_constraints_outputs.json`
- `<model>_few_shot_outputs.json`
- `<model>_self_check_outputs.json`
- `<model>_plan_then_ground_outputs.json`

## Minimum Reporting Columns

- model
- variant
- sample_scope
- temperature
- max_tokens
- task_success_rate
- execution_success_rate
- parsing
- hallucination
- missing_step
- wrong_order
- additional_step
