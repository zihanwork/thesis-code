# 完整进展报告 · From Goal Understanding to Action Failure

> 项目：硕士论文 *From Goal Understanding to Action Failure: A Diagnostic
> Study of Large Language Models for Embodied Decision Making*
> 编写时间：2026-04-30
> 工作根目录：`/Users/hanson/Documents/Newcastle/thesis/code`

本报告把目前所有改动按"代码 / 数据产出 / 论文写作 / 复现流程"四个维度
聚合，所有路径都是仓库内相对路径，可以直接点开。

---

## 0. 老师 4 条意见 → 当前进展速查

| # | 老师意见 | 完成情况 | 主要落点 |
| --- | --- | --- | --- |
| 1 | 测试更多 AI 模型，不只限于 GPT-4 | 已整理 17 个模型的真实评测；并搭好可一键切换 OpenAI / Anthropic / Gemini 的多厂商生成层 | [`output/diagnostics/multimodel_existing_inventory.csv`](multimodel_existing_inventory.csv)、[`analysis/generate_outputs.py`](../../analysis/generate_outputs.py) |
| 2 | 补充查找实验相关资料 | EAI / VirtualHome / Plan-and-Act / Self-Refine / Reflexion / ReAct / EmbodiedBench 7 项资料整理成可引用条目 | [`multimodel_experiment_materials.md`](multimodel_experiment_materials.md) 的 *Related Work Notes* 段 |
| 3 | 研究提升模型成功率的方法 | 设计了 5 个 action 变体 + 4 个 goal 变体 + 1 个 self-check rewrite，全部实现并 dry-run 跑通 | [`analysis/prompt_variants.py`](../../analysis/prompt_variants.py)、[`analysis/self_check_loop.py`](../../analysis/self_check_loop.py)、[`paper/06_methods_for_improving_success.md`](../../paper/06_methods_for_improving_success.md) |
| 4 | 在展示材料中加入更详细的描述和图表 | 8 张 SVG（task success、execution、F1、family avg、failure profile、goal vs action、ablation × 2）+ 失败案例 case study + 多模型实验材料 md | [`output/diagnostics/figures/`](figures/)、[`failure_case_studies.md`](failure_case_studies.md) |

---

## 1. 论文骨架与章节进度

> 字数累计 ≈ **4,460 词**（不含 outline 与本报告）

| 文件 | 章节 | 字数 | 状态 |
| --- | --- | ---: | --- |
| [`paper/outline.md`](../../paper/outline.md) | 总骨架 | — | 已含 RQ1–RQ4、章节结构、代码 → 论文章节映射、Contribution Statement |
| [`paper/01_introduction_and_related_work.md`](../../paper/01_introduction_and_related_work.md) | 第 1 章 Introduction + 第 2 章 Background and Related Work | ~958 | 初稿完成 |
| [`paper/02_experimental_setup.md`](../../paper/02_experimental_setup.md) | 第 3 章 Experimental Setup | ~1,099 | 初稿完成（RQ → metric 对应表已落） |
| [`paper/03_failure_diagnosis.md`](../../paper/03_failure_diagnosis.md) | 第 5 章 Failure Diagnosis | ~1,433 | 初稿完成（含 3 个真实 case：`Turn on light 125_2`、`Drink 510_1`、`Write an email 996_2`） |
| [`paper/06_methods_for_improving_success.md`](../../paper/06_methods_for_improving_success.md) | 第 6 章 Methods for Improving Success | ~970 | 骨架完成；含 §6.1 **Pre-registered Predictions** 表，empirical 数字保留 TBD 待真实运行回填 |

**待写章节**（建议按顺序）：

- 第 4 章 **Multi-model Evaluation (RQ1, RQ3)** — 数据齐全，可立即写。
- 第 7 章 Discussion / 第 8 章 Conclusion — 等第 6 章数字回填后再写。

### 论文主线（一句话）

> LLM 在 EAI/VirtualHome 上有显著的 **goal-to-action gap**：goal 已对、action 仍败的失败约占 70%，其中绝大多数是
> *hallucinated precondition* 或 *omitted precondition step* 这两种局部错误，
> 因此可以用轻量级 prompt + self-check 干预定向治理。

---

## 2. 代码改动清单

总计 **2,824 行**新增 / 修改代码（`wc -l`）。

### 2.1 新增脚本

| 文件 | 行数 | 作用 |
| --- | ---: | --- |
| [`analysis/prompt_variants.py`](../../analysis/prompt_variants.py) | 250 | 集中定义 `PromptVariant` 数据类与 5 个 action 变体（`baseline`、`format_constraints`、`few_shot_valid_actions`、`plan_then_ground`、`self_check_rewrite`）和 4 个 goal 变体（`baseline`、`schema_constrained`、`few_shot`、`decompose_then_merge`）。 |
| [`analysis/generate_outputs.py`](../../analysis/generate_outputs.py) | 304 | 统一多厂商生成层：支持 OpenAI、Anthropic、Gemini、OpenAI-compatible、`dry_run`。处理单 pass / 双 pass、prompt 截断、休眠、文件命名规范。 |
| [`analysis/improve_goal_interpretation.py`](../../analysis/improve_goal_interpretation.py) | 181 | 专门跑 goal interpretation 变体，对输出做 schema 验证（node states、edge relations、action keys），输出 `<model>_<variant>_validation.json`。 |
| [`analysis/self_check_loop.py`](../../analysis/self_check_loop.py) | 201 | 失败驱动的 critique-rewrite：读取 baseline outputs + EAI `error_info.json`，对失败行触发重写，passing 行原样保留，并产出 `<model>_self_check_report.json`。 |
| [`scripts/run_improvement_pipeline.sh`](../../scripts/run_improvement_pipeline.sh) | 191 | 端到端流水线（生成 prompts → 多变体推理 → self-check → 规范化 → EAI 评测 → 刷新材料和图）。所有路径与 provider 都用环境变量切换；`dry_run` smoke 已验证可跑通。 |

### 2.2 修改脚本

| 文件 | 行数 | 关键改动 |
| --- | ---: | --- |
| [`analysis/prepare_multimodel_experiment_materials.py`](../../analysis/prepare_multimodel_experiment_materials.py) | 1,193 | 新增 `parse_variant`、`collect_ablation_results`、`goal_to_action_pairs`；新增 `write_scatter_svg`（fig_goal_vs_action）、`write_ablation_svg`（fig_ablation_action）、`write_goal_ablation_svg`（fig_ablation_goal）；输出 `multimodel_ablation_summary.csv/.md`、`multimodel_goal_vs_action.csv`。 |
| [`analysis/link_goal_to_action_failures.py`](../../analysis/link_goal_to_action_failures.py) | 504 | 新增 `FAILURE_TYPE_EXPLANATION`、`write_case_study_md`、`_stringify_actions`、`_truncate`；新增 CLI `--case-study-top-n`、`--error-info-json`、`--gold-actions-json`；修复 `model` 字段回退（goal log 缺 `Model name is` 时回落到 action log）。 |

### 2.3 既有脚本（被流水线调用，未改）

| 文件 | 作用 |
| --- | --- |
| [`analysis/normalize_action_outputs.py`](../../analysis/normalize_action_outputs.py) | 把 LLM 自由文本动作转成 EAI 可解析的 `name/id` 格式 |
| [`scripts/run_action_sequencing_eval.sh`](../../scripts/run_action_sequencing_eval.sh) | 调用 `conda run -n eai-eval eai-eval` 跑 action sequencing 评测 |
| [`scripts/run_action_sequencing_eval_normalized.sh`](../../scripts/run_action_sequencing_eval_normalized.sh) | 在 normalised 输出上跑 EAI 评测（针对自由文本输出的兼容版本） |
| [`scripts/run_action_sequencing_generate_prompts.sh`](../../scripts/run_action_sequencing_generate_prompts.sh) | 单独触发 EAI prompt 生成步骤 |
| [`scripts/run_legal_action_pipeline.sh`](../../scripts/run_legal_action_pipeline.sh) | 早期管线，被 `run_improvement_pipeline.sh` 取代但保留参考 |

---

## 3. 数据产出与图表

### 3.1 多模型实验材料

| 文件 | 内容 |
| --- | --- |
| [`multimodel_experiment_materials.md`](multimodel_experiment_materials.md) | 主报告：模型清单、Top-N 表、失败剖面、家族平均、Related Work Notes、消融占位 |
| [`multimodel_existing_inventory.csv`](multimodel_existing_inventory.csv) / [`.json`](multimodel_existing_inventory.json) | 17 模型 × 2 模块 的全量行（去重后） |
| [`multimodel_failure_profile.csv`](multimodel_failure_profile.csv) | 每模型 7 类失败的百分比 |
| [`multimodel_family_averages.csv`](multimodel_family_averages.csv) | 6 个家族的平均 task success / all_f1 |
| [`multimodel_goal_vs_action.csv`](multimodel_goal_vs_action.csv) | scatter 数据（goal F1 vs action task success） |
| [`multimodel_ablation_summary.csv`](multimodel_ablation_summary.csv) / [`.md`](multimodel_ablation_summary.md) | 消融结果汇总（等真实运行回填） |
| [`multimodel_figure_plan.csv`](multimodel_figure_plan.csv) | 8 张图与所用数据列的索引 |

### 3.2 失败诊断

| 文件 | 内容 |
| --- | --- |
| [`failure_case_studies.md`](failure_case_studies.md) | **真实** 5 个 goal-correct-but-action-fail 案例（model 字段已正确显示 `gpt-4o-2024-05-13`） |
| [`goal_action_diagnostics.md`](goal_action_diagnostics.md) | goal 与 action 失败联动的 markdown 诊断 |
| [`goal_action_joined_cases.csv`](goal_action_joined_cases.csv) | 342 个 joined cases 的明细行 |
| [`goal_correct_but_action_fail_top_cases.json`](goal_correct_but_action_fail_top_cases.json) | top-N 高 goal F1 仍失败的案例 |
| [`failure_pattern_counts.json`](failure_pattern_counts.json) | `relation_grounding 59 / planning_order 39 / other 40 / format 1` 的分布 |

### 3.3 图表（[`figures/`](figures/)）

| 文件 | 用于 |
| --- | --- |
| [`fig_action_task_success.svg`](figures/fig_action_task_success.svg) | 第 4 章：各模型 action task success 排序 |
| [`fig_action_execution_success.svg`](figures/fig_action_execution_success.svg) | 第 4 章：execution success 排序 |
| [`fig_goal_interpretation_f1.svg`](figures/fig_goal_interpretation_f1.svg) | 第 4 章：goal F1 排序 |
| [`fig_family_average.svg`](figures/fig_family_average.svg) | 第 4 章：家族平均 |
| [`fig_failure_profile.svg`](figures/fig_failure_profile.svg) | 第 5 章：失败剖面热图 |
| [`fig_goal_vs_action.svg`](figures/fig_goal_vs_action.svg) | 第 4/5 章：goal F1 vs action success scatter |
| [`fig_ablation_action.svg`](figures/fig_ablation_action.svg) | 第 6 章：action 变体消融柱状图（待数据） |
| [`fig_ablation_goal.svg`](figures/fig_ablation_goal.svg) | 第 6 章：goal 变体消融柱状图（待数据） |

### 3.4 dry-run smoke 流水线产物

证明 `scripts/run_improvement_pipeline.sh` 可以端到端执行（包含
self-check 步骤），所有产物在
[`output/improvement_run/`](../improvement_run/)：

```text
output/improvement_run/
├── prompts/virtualhome/generate_prompts/
│   ├── action_sequencing/helm_prompt.json    (合成 3 任务)
│   └── goal_interpretation/helm_prompt.json  (合成 2 任务)
├── helm_output/virtualhome/
│   ├── action_sequencing/
│   │   ├── smoke-model_outputs.json                     # baseline
│   │   ├── smoke-model_format_constraints_outputs.json
│   │   ├── smoke-model_few_shot_outputs.json
│   │   ├── smoke-model_plan_then_ground_outputs.json
│   │   ├── smoke-model_self_check_outputs.json
│   │   └── smoke-model_self_check_report.json           # 2/2 rewrite, 1 skip
│   └── goal_interpretation/
│       ├── smoke-model_outputs.json + 3 个变体
│       └── 4 个 *_validation.json (schema 校验报告)
└── helm_output_norm/virtualhome/action_sequencing/      # 规范化为 EAI 可读格式
    └── 5 个 *_outputs.json
```

---

## 4. 复现命令（一份脚本搞定）

### 4.1 dry-run（已验证可跑通，无需 API key）

```bash
PROVIDER=dry_run \
API_MODEL=dummy \
MODEL_NAME=smoke-model \
RUN_EVAL=0 \
RUN_SELF_CHECK=1 \
MAX_PROMPTS=3 \
SLEEP=0 \
ACTION_VARIANTS="baseline format_constraints few_shot_valid_actions plan_then_ground" \
GOAL_VARIANTS="baseline schema_constrained few_shot decompose_then_merge" \
OUTPUT_ROOT=output/improvement_run \
PROMPTS_DIR=output/improvement_run/prompts \
RESPONSES_DIR=output/improvement_run/helm_output \
NORMALISED_ROOT=output/improvement_run/helm_output_norm \
EVAL_OUTPUT_DIR=output/improvement_run \
bash scripts/run_improvement_pipeline.sh
```

### 4.2 真实运行（待补 API key）

```bash
export OPENAI_API_KEY=sk-...

PROVIDER=openai \
API_MODEL=gpt-4o-mini \
MODEL_NAME=gpt-4o-mini \
RUN_EVAL=1 \
RUN_SELF_CHECK=1 \
MAX_PROMPTS=0 \
bash scripts/run_improvement_pipeline.sh
```

跑完后 [`multimodel_ablation_summary.md`](multimodel_ablation_summary.md)
与 [`fig_ablation_action.svg`](figures/fig_ablation_action.svg) /
[`fig_ablation_goal.svg`](figures/fig_ablation_goal.svg) 会自动填入数据，
第 6 章 paper 的 `TBD` 也可直接替换。

### 4.3 单独刷新失败案例

```bash
python3 analysis/link_goal_to_action_failures.py \
  --goal-log logs/goal_interpretation_eval_20260422_143944.log \
  --action-log logs/action_sequencing_eval_20260422_144000.log \
  --output-dir output/diagnostics \
  --case-study-top-n 5 \
  --error-info-json output_norm_all/virtualhome/evaluate_results/action_sequencing/gpt-4o-2024-05-13/error_info.json
```

---

## 5. 关键发现（可直接拿去汇报）

来自真实数据，不依赖未来实验：

1. **goal 与 action 是不同能力。** 17 个模型中，goal F1 最高的
   `o1-preview-2024-09-12`（all-F1 42.75）在 action sequencing 上只排第 9
   （task success 60.0）；最强的 action 模型是 `mistral-large-2402`
   （task success 76.39）。
2. **失败模式分三种 regime：**
   - *format-bottlenecked*（`mixtral-8x22b` 100% / `gemini-1.0-pro` 76% /
     `llama-3-70b` 38% / `gpt-3.5-turbo` 35% 全部卡在 parsing）。
   - *reasoning-bottlenecked*（`gpt-4o` 25% / `gpt-4-turbo` 32% /
     `claude-3-haiku` 43% / `gemini-1.5-flash` 30% 主要是 missing_step）。
   - *hallucination-bottlenecked*（`llama-3-8b` 41% / `cohere-command-r` 29%
     / `claude-3-opus` 14% 主要是 hallucination）。
3. **`wrong_order` 几乎不是主因**（除 Cohere 系列，所有模型 ≤ 2.6%）。
   论文里的常见说法"LLM 在长程顺序上失败"在 EAI/VirtualHome 上 *并不成立*。
4. **当 goal 已对、action 仍败时**（139 个案例），
   - 42% 是 relation_grounding（虚构前置或抓错对象 id）
   - 28% 是 planning_order / missing_step（漏一个前置动作）
   - 仅 1 个是 pure parsing failure
   ⇒ goal-to-action gap 是 *局部* 的可恢复错误，不是深层规划失败。
5. **方法预测**（已写入第 6 章 §6.1，等回填验证）：
   - `format_constraints` 攻 parsing；
   - `few_shot_valid_actions` 攻 hallucination；
   - `plan_then_ground` / `self_check_rewrite` 攻 missing_step；
   - `schema_constrained` / `decompose_then_merge` 攻 goal F1 子项。

---

## 6. 待办与等待项

| 项 | 阻塞原因 | 一旦解锁怎么做 |
| --- | --- | --- |
| 第 6 章 ablation 数字回填 | 缺 API key | `export OPENAI_API_KEY=…` 后跑 §4.2 命令；约 305×N 任务，预计 30–60 分钟 |
| 跨家族对照（Anthropic 或 Gemini） | 缺 API key | 同上，加 `ANTHROPIC_API_KEY` 或 `GEMINI_API_KEY` |
| 第 4 章 Multi-model Evaluation 初稿 | 无阻塞 | 数据齐全，可随时撰写 |
| 第 7、8 章 Discussion / Conclusion | 等第 6 章回填 | 数字到位后一气呵成 |
| 老师确认论文主线 (D1) | 等老师反馈 | 收到反馈后微调 §5.4 / §6.1 的 attack 列定义 |

---

## 7. 与上一版 [`progress_report.md`](progress_report.md) 的关系

`progress_report.md` 是 *面向老师的简版* 报告（已包含 dry-run smoke 与第三章
第六章初稿信息）。本文件 `full_progress_report.md` 是 *面向自己复盘* 的全量
版本，给出每一个文件的链接、行数、用途，便于做"任意一项产出 → 对应代码"的
反查；两者互补，无须二选一。

---

## 8. 给老师的一句话

> 我已经把整套实验从"只跑 GPT-4 一个模型"扩展为：(1) 17 个模型的真实诊断，
> (2) 一条可一键复现的多厂商生成 + 自检评测流水线，(3) 在真实失败数据上
> 提出可被 ablation 验证的 6 条 prompt 干预预测。下周只要 API key 就绪，
> 就能把第 6 章 ablation 表填完，整本论文就只剩 Discussion / Conclusion。
