# 导师汇报材料：E2 闭环知识库实验进展

> 项目：Bridging the Goal-to-Action Gap: Diagnosing and Improving LLM Failures in Embodied Planning  
> 汇报日期：2026-06-12  
> 汇报重点：使用 `gpt-5.5` 和持久化知识库完成 E2 closed-loop 实验，评估 failure write-back 是否带来持续改进

---

## 1. 本阶段目标

前一阶段已经完成了 E1 实验，即静态 persistent KB 与 in-memory baseline 的对比。E1 的结果是 mixed result：persistent KB 在 `action_goal` 上有小幅提升，但整体 `task_success_rate` 略低于 in-memory baseline。因此，本阶段不再简单验证“persistent KB 是否一次性提升成功率”，而是进一步测试：

> 当失败案例被写回知识库后，系统是否能在后续迭代中减少重复失败、提高可恢复性，并形成稳定的闭环改进趋势。

这对应论文中的 E2 实验：iterative KB / closed-loop failure write-back。

---

## 2. 本阶段完成的工程改造

为了跑 E2，先完成了三项必要工程工作。

### 2.1 接入百度 One API 与 `gpt-5.5`

原 harness 支持 `openai_compatible` provider，但没有把 One API 的 `base_url` 传入生成脚本。因此修改了 `analysis/kb/harness.py`：

- 新增环境变量 `KB_BASE_URL`
- 当 `KB_PROVIDER=openai_compatible` 时，将 `KB_BASE_URL` 传给 `analysis/generate_outputs.py`
- 若缺少 `KB_BASE_URL`，harness 会 warning 并 soft-fail，避免误连默认 endpoint

最终确认 One API 可用模型列表中存在实际模型 ID：

```text
gpt-5.5
```

随后用最小 chat completion 测试确认：

```text
OK: OK
```

说明 base URL、token、模型名和 OpenAI-compatible 调用链路均已打通。

### 2.2 在 harness 中加入自动 normalize

`gpt-5.5` 生成的动作输出虽然非空，但 EAI evaluator 需要特定目录结构和动作格式。因此修改了 `analysis/kb/harness.py` 的 `_run_eai_eval()`：

- raw 输出复制到：

```text
<iter>/eval/llm_response_raw/virtualhome/action_sequencing/
```

- 自动调用：

```bash
analysis/normalize_action_outputs.py
```

- normalize 后输出到：

```text
<iter>/eval/llm_response/virtualhome/action_sequencing/
```

- EAI evaluator 使用 normalized root 作为 `LLM_RESPONSE_PATH`

验证结果：`gpt-5.5` smoke 中 3/3 rows 可被 normalize。

### 2.3 从 `error_info.json` 构造逐任务 rows

EAI 的 `summary.json` 是聚合指标，不包含逐任务 rows。例如：

```json
{
  "goal_evaluation": {"task_success_rate": 78.4884},
  "trajectory_evaluation": {"execution_success_rate": 86.6}
}
```

但 E2 需要统计：

- fixed cases
- regressed cases
- bad cases
- failure type 分布
- per-iteration pass/fail

因此扩展了 `parse_summary()`：当 `summary.json` 没有 rows 时，自动读取同目录 `error_info.json`，并构造逐任务 row。

在 iter101 小规模验证中，解析结果为：

```text
rows=6
task_success_rate=0.6667
num_failed=2
bad_cases=2
failure_types={'passed': 4, 'hallucination error': 1, 'missing_step': 1}
```

说明逐任务失败案例已能被 harness 正确收集，用于后续 fixed/regressed 与 KB write-back 分析。

---

## 3. E2 实验设置

### 3.1 模型与方法

| 项目 | 设置 |
|---|---|
| Model | `gpt-5.5` via 百度 One API |
| Provider | OpenAI-compatible endpoint |
| Variant | `plan_then_ground` |
| Dataset | EAI / VirtualHome action sequencing |
| Prompt count | 200 prompts |
| Iterations | 3 rounds |
| Iteration IDs | 201, 202, 203 |
| Repair module | KG planning agent enabled |
| Evaluation | EAI evaluator after normalization |

### 3.2 实验流程

每一轮执行以下步骤：

1. 使用 `gpt-5.5` 生成 200 条 action-sequencing outputs
2. 调用 KG planning agent 对输出做知识约束修复
3. 将 repaired outputs normalize 成 EAI 可评测格式
4. 调用 EAI evaluator 得到 `summary.json` 和 `error_info.json`
5. 从 `error_info.json` 抽取 bad cases
6. 比较相邻轮次中的 fixed / regressed cases

---

## 4. E2 三轮结果

### 4.1 聚合指标

| Iteration | EAI task success | Execution success | State goal | Relation goal | Action goal | Missing step | Hallucination | Parsing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iter201 | 78.49% | 86.6% | 91.67% | 61.02% | 65.33% | 9.30% | 2.33% | 0.58% |
| iter202 | 79.07% | 86.6% | 91.67% | 61.02% | 65.33% | 8.72% | 2.33% | 0.58% |
| iter203 | 78.49% | 86.0% | 90.28% | 61.02% | 64.00% | 9.30% | 2.33% | 0.58% |

### 4.2 Row-level closed-loop 统计

| Iteration | Parsed rows | Row-level success | Failed rows | Bad cases | Main failure types |
|---|---:|---:|---:|---:|---|
| iter201 | 170 | 86.47% | 23 | 23 | missing_step 16, hallucination 4, parameter 2, parsing 1 |
| iter202 | 170 | 86.47% | 23 | 23 | missing_step 15, hallucination 4, parameter 3, parsing 1 |
| iter203 | 170 | 85.88% | 24 | 24 | missing_step 16, hallucination 4, parameter 3, parsing 1 |

### 4.3 Fixed / regressed cases

| Transition | Fixed cases | Regressed cases | Row-level delta |
|---|---|---|---:|
| iter201 → iter202 | `415_1` | `181_1` | 0.00 |
| iter202 → iter203 | `269_2`, `509_2` | `1027_2`, `244_2`, `813_2` | −0.59 pp |

---

## 5. 主要观察

### 5.1 闭环没有形成单调提升

E2 的核心发现是：failure write-back 和 KG repair 确实会改变局部样本的结果，但没有形成稳定的单调提升曲线。

- iter201 → iter202：EAI task success 从 78.49% 提升到 79.07%，提升约 +0.58 pp
- iter202 → iter203：又回落到 78.49%
- row-level success 从 86.47% 保持一轮后下降到 85.88%

因此，当前结果不支持“闭环知识库持续提升整体成功率”的强结论。

### 5.2 系统具备局部恢复能力，但也引入 regression

相邻轮次中均出现 fixed cases，说明闭环机制确实能修复部分任务：

- iter201 → iter202 修复了 `415_1`
- iter202 → iter203 修复了 `269_2` 和 `509_2`

但同时也出现 regressed cases：

- iter201 → iter202 退化了 `181_1`
- iter202 → iter203 退化了 `1027_2`, `244_2`, `813_2`

这说明 KB write-back / repair 不是纯收益机制，而是在部分任务上提供修复信号，同时可能因为检索噪声、规则过泛化或上下文干扰造成退化。

### 5.3 主导失败类型仍是 `missing_step`

三轮中最主要失败类型始终是 `missing_step`：

- iter201: 16
- iter202: 15
- iter203: 16

这与之前的诊断结论一致：LLM 在 VirtualHome action sequencing 中的主要问题不是完全不理解任务，而是在把目标转成可执行动作序列时漏掉必要前置步骤。

### 5.4 `gpt-5.5` 的格式稳定性较好

三轮中 parsing error 都维持在很低水平：

```text
0.5814%
```

说明对于强模型，格式问题已经不是主要瓶颈；更关键的问题是 grounded execution 中的 precondition / missing-step / object-grounding 错误。

---

## 6. 和 E1 的关系

E1 是 static persistent KB 实验，结果为：

| Metric | In-memory baseline | Persistent KB | Delta |
|---|---:|---:|---:|
| task_success_rate | 79.07% | 77.33% | −1.74 pp |
| action_goal | 68.00% | 69.33% | +1.33 pp |
| execution_success_rate | 87.20% | 85.50% | −1.70 pp |
| missing_step | 8.72% | 9.30% | +0.58 pp |

E1 说明 static persistent KB 并没有直接提升整体成功率，但在 action grounding 上有一定信号。

E2 进一步说明：即使引入 failure write-back 和闭环迭代，也没有形成稳定持续提升，但可以观察到 fixed/regressed 的局部变化。

因此，E1 + E2 可以形成一个比较诚实且有研究价值的结论：

> Persistent KB provides useful grounding and recovery signals, but naive retrieval/write-back closed loops do not guarantee monotonic improvement. The remaining bottleneck is not merely access to more knowledge, but controlling when and how retrieved failure cases should influence plan repair.

---

## 7. 可以写进论文的结论

建议论文中不要写成“persistent KB 显著提升成功率”，而应写成以下更稳健的结论：

1. **Goal-to-action gap 仍然存在。** 即使使用更强模型 `gpt-5.5`，action sequencing 中仍然存在 missing-step 和 grounded execution 失败。
2. **Persistent KB 能提供 grounding / repair signal。** E1 中 action_goal 有小幅提升；E2 中出现 fixed cases。
3. **闭环写回不是单调收益。** E2 三轮中第 2 轮略升，第 3 轮回落，说明 naive failure write-back 可能引入 regression。
4. **主要失败仍集中在 missing_step。** 后续方法应重点处理前置动作补全和条件约束选择，而不是继续堆叠更多上下文。
5. **负结果本身有价值。** 该结果说明 embodied planning 中的 external memory / KG 不能简单地通过“加入更多失败案例”获得稳定提升，需要更精细的 retrieval filtering、confidence gating 和 repair selection。

---

## 8. 明天汇报时建议强调的三句话

1. 我已经把项目从静态 KB 扩展到了真正的 closed-loop E2：每轮生成、修复、评测、解析失败，再比较 fixed/regressed。
2. E2 的结果不是单调提升：第 2 轮有小幅改善，但第 3 轮回落，说明 failure write-back 有局部修复能力但也会带来 regression。
3. 这个结果改变了论文叙事：重点不是宣称 KB 一定提升，而是说明 embodied planning 中 naive memory/retrieval loop 的局限，并提出后续需要 retrieval filtering 和 repair gating。

---

## 9. 下一步计划

### 9.1 短期：把 E2 写进论文结果章

需要补充：

- E2 实验设置
- 三轮指标表
- fixed/regressed case 表
- failure type 分布表
- 对 non-monotonic trend 的解释

### 9.2 中期：补一张 E3 收敛/趋势图

建议画三条曲线：

- EAI task success vs iteration
- row-level success vs iteration
- missing_step count vs iteration

这可以作为 E3 convergence study 的核心图。

### 9.3 后续改进方向

如果还要继续做方法改进，不建议简单增加第 4、5 轮，而应改闭环策略：

- retrieval filtering：只检索与当前任务高度相似的 failure cases
- confidence gating：只有当 KG repair 高置信时才修改原输出
- regression check：修复后先用 symbolic verifier 过滤可能退化的动作
- failure-type-specific repair：针对 missing_step、hallucination、parameter error 使用不同修复策略

---

## 10. 关键文件索引

### E2 汇总

```text
output/harness/e2main_gpt55_summary.json
```

### E2 三轮 summary

```text
output/harness/iter201/eval/virtualhome/evaluate_results/action_sequencing/iter201_repaired/summary.json
output/harness/iter202/eval/virtualhome/evaluate_results/action_sequencing/iter202_repaired/summary.json
output/harness/iter203/eval/virtualhome/evaluate_results/action_sequencing/iter203_repaired/summary.json
```

### E2 三轮 error info

```text
output/harness/iter201/eval/virtualhome/evaluate_results/action_sequencing/iter201_repaired/error_info.json
output/harness/iter202/eval/virtualhome/evaluate_results/action_sequencing/iter202_repaired/error_info.json
output/harness/iter203/eval/virtualhome/evaluate_results/action_sequencing/iter203_repaired/error_info.json
```

### Harness 修改

```text
analysis/kb/harness.py
```

### 论文 interim report

```text
paper/CSC8639_interim_report.md
```
