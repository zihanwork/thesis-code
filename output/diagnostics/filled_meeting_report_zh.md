# 从目标理解到动作失败
## 组会更新报告（补全版）

### 1. 基本信息
- 暂定论文题目：从目标理解到动作失败：面向具身决策的大语言模型诊断研究
- 会议形式：线上线下混合
- 日期：4.20
- 汇报人：wu zihan
- 当前阶段：已完成基线复现，正在开展下游诊断

### 2. 本周核心更新
#### 2.1 一句话更新
本周已完成 EAI 中 goal_interpretation 基线复现，并完成 action_sequencing 下游运行，开始诊断“目标看似正确但动作不可执行”的原因。

#### 2.2 本周重点工作
- 回顾 EAI 框架并复核模块接口
- 跑通 `goal_interpretation` 并整理多模型结果
- 跑通 `action_sequencing` 并收集日志
- 构建 goal 到 action 的小规模诊断链路

#### 2.3 已完成内容
- 环境与 CLI 复现完成（`conda + Python 3.8 + eai-eval`）
- Goal Interpretation 评测完成并完成排序
- Action Sequencing（VirtualHome）评测完成
- 诊断脚本已完成：运行、汇总、跨模块对齐

#### 2.4 正在进行内容
- 区分“格式/解析失败”与“真实规划推理失败”
- 优化 case 与模型的一一映射，减少 unknown-model 记录
- 扩展到更多下游模块进行对比

### 3. 研究方向
#### 3.1 研究问题
当 LLM 在目标层面看起来理解正确时，为什么仍会在动作层面产生不可执行或失败的计划？

#### 3.2 研究动机
- 仅看总体成功率无法定位错误发生在哪一层
- 具身决策是流水线问题，目标正确不代表执行成功
- 需要模块级 + 案例级诊断

#### 3.3 当前范围
- 基准框架：EAI
- 仿真器：VirtualHome
- 主模块：Goal Interpretation
- 下游模块：Action Sequencing
- 诊断重点：关系落地、隐式前提、可执行性与规划

#### 3.4 预期贡献
在基准复现之外，提供从“目标理解”到“动作失败”的可诊断分析链路。

### 4. Benchmark / Codebase 进展
#### 4.1 环境与代码进展
- 参考仓库：https://github.com/embodied-agent-interface/embodied-agent-interface?tab=readme-ov-file
- 关键脚本：`scripts/run_action_sequencing_eval.sh`、`analysis/summarize_downstream.py`、`analysis/link_goal_to_action_failures.py`
- 输出结构：`output/virtualhome/evaluate_results/{goal_interpretation,action_sequencing}/<model>/summary.json`
- 诊断产物：`output/diagnostics/`

#### 4.2 已定位模块
- Goal Interpretation：已完成
- Action Sequencing：已完成
- Subgoal Decomposition：已定位，待诊断
- Transition Modeling：已定位，待诊断

#### 4.3 当前运行状态
- Goal Interpretation / VirtualHome / 18 模型 / 已完成
- Action Sequencing / VirtualHome / 18 模型 / 已完成（但存在明显格式与解析问题）

### 5. 基线结果
#### 5.1 Goal Interpretation 基线摘要（按 all_f1 前5）
- 第1名：o1-preview-2024-09-12 | all_f1=42.7462，all_precision=31.7547，all_recall=65.3750
- 第2名：cohere-command-r | all_f1=36.6883，all_precision=27.3608，all_recall=55.6650
- 第3名：gpt-4o-2024-05-13 | all_f1=36.5318，all_precision=26.4067，all_recall=59.2500
- 第4名：gemini-1.5-pro-preview-0409 | all_f1=36.2242，all_precision=33.5886，all_recall=39.3086
- 第5名：gpt-4-turbo-2024-04-09 | all_f1=33.2433，all_precision=24.0379，all_recall=53.8750

#### 5.2 Goal Interpretation 关键观察
- 多数模型出现 recall 高于 precision，表现为“多生成”倾向
- node/edge/action 三类指标不均衡，说明能力分布差异明显
- 存在异常全 0 结果，需继续排查输入输出配置

#### 5.3 下游模块基线摘要（Action Sequencing）
- 评测模型数：18；task_success_rate 为 0 的模型数：18
- claude-3-5-sonnet-20240620：task_success_rate=0.0，execution_success_rate=0.0，grammar.parsing=100.0
- claude-3-haiku-20240307：task_success_rate=0.0，execution_success_rate=0.0，grammar.parsing=100.0
- claude-3-opus-20240229：task_success_rate=0.0，execution_success_rate=0.0，grammar.parsing=100.0

#### 5.4 下游模块关键观察
- 当前失败主要集中在格式与解析层
- 日志高频出现：`Action WALK does not follow name_id format`
- 说明很多样本在规划深层评估前已因格式问题失败

### 6. 小规模诊断切片
#### 6.1 诊断问题
哪些样本在 goal 层面看起来不错，但在 action 层面仍失败？

#### 6.2 案例摘要（Top-3）
- Case 1:
  - Task: Pet cat
  - file_id: 203_2
  - Goal 指标（goal_f1）: 0.8
  - Goal 判断：在 case-level 指标上相对可接受
  - Action 结果：无可执行预测
  - 失败类型：format_or_parsing
  - 诊断备注：2026-04-21 00:10:32 - virtualhome_eval.simulation.evolving_graph.eval_utils -   Action WALK does not follow name_id format | , file 203_2 prediction has no prediction
- Case 2:
  - Task: Drink
  - file_id: 156_1
  - Goal 指标（goal_f1）: 0.8
  - Goal 判断：在 case-level 指标上相对可接受
  - Action 结果：无可执行预测
  - 失败类型：format_or_parsing
  - 诊断备注：2026-04-21 00:10:34 - virtualhome_eval.simulation.evolving_graph.eval_utils -   Action WALK does not follow name_id format | , file 156_1 prediction has no prediction
- Case 3:
  - Task: Relax on sofa
  - file_id: 137_1
  - Goal 指标（goal_f1）: 0.8
  - Goal 判断：在 case-level 指标上相对可接受
  - Action 结果：无可执行预测
  - 失败类型：format_or_parsing
  - 诊断备注：2026-04-21 00:10:37 - virtualhome_eval.simulation.evolving_graph.eval_utils -   Action WALK does not follow name_id format | , file 137_1 prediction has no prediction

#### 6.3 初步失败模式
- format_or_parsing: 305
- other: 37

### 7. 当前假设
#### 7.1 工作假设
具身决策失败不仅来自目标理解错误，也来自关系落地、隐式前提以及动作格式/可执行性问题。

#### 7.2 目前已有证据
- EAI 框架强调模块化诊断而非只看总成功率
- 当前结果显示 goal 分数与 action 可执行性存在明显断层
- action 日志显示大量 parsing/format 失败

#### 7.3 仍需补充证据
- 设计实验区分“格式失败”与“规划推理失败”
- 增加高 goal 质量但 runtime 失败的案例
- 扩展到 subgoal decomposition 或 transition modeling 进行对照

### 8. 组会讨论问题
1. 论文阶段一是否先聚焦 `goal_interpretation + action_sequencing`？
2. 当前下游全 0 结果下，是否应先做动作格式标准化，再进入深层诊断？
3. 第一阶段是否只用 VirtualHome，后续再纳入 BEHAVIOR？
4. 论文定位是否以“诊断分析”优先，而非“全量基准复现”？

