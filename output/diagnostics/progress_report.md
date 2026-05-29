# 进度汇报：诊断与提升 EAI/VirtualHome 中的 LLM 能力

## 项目定位
论文题目：**From Goal Understanding to Action Failure: A Diagnostic Study
of Large Language Models for Embodied Decision Making**

我们不再做单纯的模型排行榜，而是聚焦于：模型在“理解目标”和“生成可执行动作序列”
之间的失败链。下面分别给出根据老师四条建议完成的进度、论文主线、下周计划。

---

## 1. 老师建议 → 已完成进度

### 建议 1：测试更多 AI 模型，不只限于 GPT-4
- 整理仓库中已有 OpenAI、Claude、Gemini、Llama、Mistral、Cohere、o1 等 18 个模型
  的评测结果，去重后 37 条记录，覆盖两个评测模块。
- 设计了新增模型的统一生成与评测矩阵，包括家族、样本范围、生成参数、输出目录、
  评测命令。
- 实现统一多厂商生成层，可在同一脚本下切换 OpenAI、OpenAI-compatible、Anthropic、
  Gemini，以及离线 dry-run 模式。
- 相关代码与材料：
  - [`analysis/generate_outputs.py`](../../analysis/generate_outputs.py)
  - [`analysis/prompt_variants.py`](../../analysis/prompt_variants.py)
  - [`output/diagnostics/multimodel_existing_inventory.csv`](multimodel_existing_inventory.csv)
  - [`output/diagnostics/multimodel_new_model_matrix.csv`](multimodel_new_model_matrix.csv)
  - [`output/diagnostics/multimodel_experiment_materials.md`](multimodel_experiment_materials.md)

### 建议 2：补充查找实验相关资料
- 整理 EAI、VirtualHome、Plan-and-Act、EmbodiedBench、Self-Refine、Reflexion、ReAct
  等核心文献，并在论文骨架中明确每篇的引用位置。
- 相关代码与材料：
  - [`output/diagnostics/multimodel_related_work.csv`](multimodel_related_work.csv)
  - [`paper/outline.md`](../../paper/outline.md)
  - [`paper/01_introduction_and_related_work.md`](../../paper/01_introduction_and_related_work.md)

### 建议 3：研究提升模型成功率的方法
- 设计五种 action sequencing 提升变体（baseline、format_constraints、few_shot、
  self_check_rewrite、plan_then_ground）和四种 goal interpretation 提升变体
  （baseline、schema_constrained、few_shot、decompose_then_merge）。
- 实现失败驱动的 critique-rewrite 循环：仅对 baseline 中失败的样本触发，写出
  `<model>_self_check_outputs.json`，与 baseline 直接对比。
- 提供完整的一键流程脚本，从 prompt 生成到评测、规范化、汇总、画图。
- 相关代码与材料：
  - [`analysis/improve_goal_interpretation.py`](../../analysis/improve_goal_interpretation.py)
  - [`analysis/self_check_loop.py`](../../analysis/self_check_loop.py)
  - [`scripts/run_improvement_pipeline.sh`](../../scripts/run_improvement_pipeline.sh)
  - [`output/diagnostics/multimodel_success_improvement_plan.csv`](multimodel_success_improvement_plan.csv)
  - [`output/diagnostics/multimodel_prompt_templates.md`](multimodel_prompt_templates.md)
  - [`output/diagnostics/multimodel_ablation_summary.csv`](multimodel_ablation_summary.csv)
  - [`output/diagnostics/multimodel_ablation_summary.md`](multimodel_ablation_summary.md)

### 建议 4：在展示材料中加入更详细的描述和图表
- 现有 5 张主图：模型 task success、execution success、goal F1、失败类型分布、
  家族均值；新增 2 张消融与诊断图：goal vs action 散点图、ablation task success 柱状图。
- 失败案例 case study Markdown，可直接放进 Failure Diagnosis 章节。
- 相关材料：
  - [`output/diagnostics/figures/fig_goal_vs_action.svg`](figures/fig_goal_vs_action.svg)
  - [`output/diagnostics/figures/fig_ablation_action.svg`](figures/fig_ablation_action.svg)
  - [`output/diagnostics/figures/fig_failure_profile.svg`](figures/fig_failure_profile.svg)
  - [`output/diagnostics/figures/fig_family_average.svg`](figures/fig_family_average.svg)
  - [`output/diagnostics/figures/fig_action_task_success.svg`](figures/fig_action_task_success.svg)
  - [`output/diagnostics/figures/fig_action_execution_success.svg`](figures/fig_action_execution_success.svg)
  - [`output/diagnostics/figures/fig_goal_interpretation_f1.svg`](figures/fig_goal_interpretation_f1.svg)
  - [`output/diagnostics/failure_case_studies.md`](failure_case_studies.md)

---

## 2. 论文主线、结构与研究问题
完整骨架在 [`paper/outline.md`](../../paper/outline.md)；摘要如下。

### 主线
LLM 在 embodied decision making 中的失败常常发生在“理解目标之后”。本文以 EAI/
VirtualHome 为评测平台，先诊断从 goal understanding 到 action failure 的断点，再
研究哪些 prompt 与自检方法能显著缩小这一断点。

### 研究问题
- **RQ1** 不同 LLM 的 goal interpretation 能力与 action sequencing 能力是否一致？
- **RQ2** 当 goal F1 较高时，action sequencing 失败主要来自哪些细粒度错误？
- **RQ3** 不同模型家族的失败模式是否系统不同？
- **RQ4** 哪些 prompt 与 self-check 方法能提升 goal F1 与 action 成功率？

### 章节结构
1. Introduction
2. Background and Related Work
3. Experimental Setup
4. Multi-model Evaluation（RQ1, RQ3）
5. Failure Diagnosis（RQ2，含 case study）
6. Methods for Improving Success（RQ4，消融实验）
7. Discussion
8. Conclusion and Future Work

### 论文开头草稿
完整文本在 [`paper/01_introduction_and_related_work.md`](../../paper/01_introduction_and_related_work.md)，长度 ≈958 词。

### 第三章 Experimental Setup 初稿
完整文本在 [`paper/02_experimental_setup.md`](../../paper/02_experimental_setup.md)，长度 ≈1100 词。固定数据集（VirtualHome / EAI 305 任务）、指标（goal F1 四项、action 7 类细粒度错误）、模型清单（17 个 baseline 加新跑的主模型），并把 RQ 与指标的对应关系做成一张表，作为后续章节的引用基准。

### 第五章 Failure Diagnosis 初稿
完整文本在 [`paper/03_failure_diagnosis.md`](../../paper/03_failure_diagnosis.md)，长度 ≈1430 词。本章直接复用真实 EAI 评测产物：
- `output/diagnostics/multimodel_failure_profile.csv` 派生出三种失败 regime（format / reasoning / hallucination-bottlenecked）。
- `output/diagnostics/failure_case_studies.md` 中的 5 个真实案例选取了 3 个最有代表性的写进章节（`Turn on light 125_2`、`Drink 510_1`、`Write an email 996_2`）。
- `output/diagnostics/failure_pattern_counts.json` 提供了 “goal 已对、action 仍败” 子样本上的失败分布（relation_grounding 59 / planning_order 39 / other 40 / format 1）。
- 章节末给出与第六章 Methods 的明确对接：每个 prompt 变体对应攻击哪一个失败列。

### 第六章 Methods for Improving Success 骨架
完整文本在 [`paper/06_methods_for_improving_success.md`](../../paper/06_methods_for_improving_success.md)，长度 ≈970 词。这是骨架（empirical 数字保留 TBD），但已经写完：
- §6.1 把第五章给出的 6 条预测固化成预注册表（`format_constraints` → 攻 parsing、`few_shot_valid_actions` → 攻 hallucination、`plan_then_ground` / `self_check_rewrite` → 攻 missing_step、`schema_constrained` / `decompose_then_merge` → 攻 goal F1 子项）。
- §6.2/6.3 给出 ablation 表的列定义与计划评测的 (model, variant) 单元格。
- §6.4 描述 self-check 协议与产生 confusion table 的方法。
- §6.5 增加 cost vs benefit（单 pass vs 双 pass、tokens、wall-clock）。
- §6.6 列出三类 threats to validity，回应"prompt 过拟合"风险。
- §6.7 给出最终结论模板，等数字一回填即可读。

---

## 3. 端到端流水线验证（dry-run smoke run）

为证明流水线可以端到端跑通，已在 `dry_run` 模式下用合成 helm prompts 跑了两轮：

- 第一轮：`baseline + format_constraints + few_shot_valid_actions + plan_then_ground`
  四个 action 变体，加上四个 goal interpretation 变体。每个变体写出
  `smoke-model_<variant>_outputs.json`，并被
  [`analysis/normalize_action_outputs.py`](../../analysis/normalize_action_outputs.py)
  规范化为 EAI 可解析的 name/id 格式。
- 第二轮：开启 `RUN_SELF_CHECK=1`，提供合成的 `error_info.json`（标注两条
  失败行：missing_step 与 wrong_order）。`smoke-model_self_check_outputs.json`
  对失败行触发了 critique-rewrite，对通过行保持原样。
  自检报告 `smoke-model_self_check_report.json` 显示
  `successful_rewrites=2/2`、`skipped_passing=1`。

可在如下目录看到这次的实际产物：

- 原始变体输出：[`output/improvement_run/helm_output/virtualhome/action_sequencing`](../improvement_run/helm_output/virtualhome/action_sequencing)
- 规范化输出：[`output/improvement_run/helm_output_norm/virtualhome/action_sequencing`](../improvement_run/helm_output_norm/virtualhome/action_sequencing)
- goal 输出与 schema 校验报告：[`output/improvement_run/helm_output/virtualhome/goal_interpretation`](../improvement_run/helm_output/virtualhome/goal_interpretation)

要把流水线接到真实 provider，只需切换环境变量：

```bash
PROVIDER=openai \
API_MODEL=gpt-4o-mini \
MODEL_NAME=gpt-4o-mini \
RUN_EVAL=1 \
MAX_PROMPTS=0 \
bash scripts/run_improvement_pipeline.sh
```

## 4. 真实失败案例已落盘

[`output/diagnostics/failure_case_studies.md`](failure_case_studies.md)
现在使用真实 EAI 评测日志（`logs/action_sequencing_eval_20260422_144000.log` 与
`logs/goal_interpretation_eval_20260422_143944.log`）和 `gpt-4o-2024-05-13` 的
`error_info.json` 生成。文件中给出了 5 个 goal-F1 高但 action 仍失败的案例（如
`Turn on light` 125_2、`Drink` 510_1、`Read book` 163_1），可以直接放进论文
Failure Diagnosis 章节。

## 5. 当前结果速览
| 指标 | 值 |
| --- | --- |
| 已整理多模型结果（去重后）| 37 行（17 个模型 × 2 个评测模块） |
| 设计的提升变体数 | 5（action）+ 4（goal）|
| 自动化生成的图表 | 7 张 SVG（含 ablation 与 goal vs action）|
| 论文骨架与开头初稿 | 完成（包含 RQ1–RQ4 与 contribution statement）|

下面三个数据点最值得在汇报时强调：

- 在已有 baseline 中，goal F1 最高的 `o1-preview` 在 action sequencing 上并不
  最优，最优是 `mistral-large-2402`（task success 76.39%）。这说明 goal 理解
  与 action 执行确实是不同能力。
- 失败类型分布显示 `missing_step` 普遍偏高（如 `gpt-4o` 25.25%、`gpt-4-turbo`
  32.13%、`gemini-1.5-flash` 29.51%），是后续 self-check 和 plan-then-ground 的
  主要攻击对象。
- `parsing` 和 `hallucination` 在多个模型上仍然存在（如 `gemini-1.0-pro` parsing
  76.07%、`llama-3-8b-chat` hallucination 41.31%），适合用 format_constraints
  与 schema_constrained 变体优先治理。

---

## 6. 下周进度计划

按以下顺序推进，每条都有明确的产出：

- D1：与老师确认论文主线“diagnostic + improvement”被采纳，研究问题定 RQ1–RQ4。
- D2：dry-run 已经在本周跑通（见第 3 节），下周直接切到真实 provider，对一个
  主模型（建议 `gpt-4o-mini`）跑 baseline 与 format_constraints。
- D3：对同一主模型跑 goal_interpretation 的 schema_constrained 变体，记录
  `all_f1` 是否提升；将结果写进 `multimodel_ablation_summary.md`。
- D4：基于 baseline 结果跑 self_check_loop，统计 missing_step 与 wrong_order
  的下降幅度。
- D5：再加一个对照模型（建议 Claude 或 Gemini 中的一个），跑 baseline 与一种
  提升变体，作为跨家族对照。
- D6：从 [`failure_case_studies.md`](failure_case_studies.md) 选 3 个典型案例
  写进 Failure Diagnosis 章节初稿。
- D7：撰写 Methods for Improving Success 章节初稿，使用 `fig_ablation_action.svg`
  与 `fig_ablation_goal.svg`。

---

## 7. 后续研究方向
- 把 self-check 反馈来源换成 EAI evaluator 的真实错误信息，与纯模型自检比较。
- 在 BEHAVIOR 数据集复现一组实验，作为跨域泛化证据。
- 加入开源模型对比，验证方法是否依赖闭源模型的强格式遵循能力。
- 把失败类型与任务长度、目标数量做相关性分析，建立任务复杂度与失败类型之间
  的描述性结论。

## 8. 给老师的一句话
> 我已经完成多模型结果整理与诊断框架，并实现了一条可复现的 prompt + self-check
> 改进管线。下一步重点是用这条管线在自己的实验中提升 goal F1 与 action 成功率，
> 而不是单纯增加模型数量。
