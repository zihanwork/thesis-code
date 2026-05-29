# 论文项目完整汇报 (Advisor Report)

**日期**：2026-05-14
**汇报人**：吴子涵
**论文工作题目**：*Bridging the Goal-to-Action Gap: A Diagnostic Study and
Knowledge-Grounded Recovery of LLM Failures in Embodied Planning*
**Benchmark**：Embodied Agent Interface (EAI, NeurIPS 2024) on VirtualHome
**汇报目的**：向导师同步当前完整进度、关键发现（含负面结果）、以及尚待决策的事项

---

## 0. 项目背景速览（30 秒看懂）

### 0.1 VirtualHome 是什么

VirtualHome 是一个**家庭机器人模拟环境**：场景里有客厅、厨房、卧室；
里面有 200+ 物体（电视、冰箱、电脑、灯…），每个物体有唯一 id (例如
`computer:319`)，以及空间关系（`computer ON desk`, `book INSIDE drawer`）。
机器人可以执行 **WALK, GRAB, OPEN, CLOSE, PUTBACK, SWITCHON…** 等十几种动作。

### 0.2 EAI Benchmark 是什么

NeurIPS 2024 的一个**大模型 embodied 评测基准**。给定一个自然语言任务
（"Turn on the computer"），让 LLM 输出一段 JSON 格式的 VirtualHome
动作序列：

```json
[
  {"WALK": ["home_office", "319"]},
  {"WALK": ["computer", "411"]},
  {"SWITCHON": ["computer", "411"]}
]
```

EAI 的关键贡献是把 LLM 的失败分成两类四模块：

- **Goal interpretation**（理解目标）：模型能否把"Turn on the computer"
  正确翻译成形式化目标（节点状态 `computer.ON=True`）
- **Action sequencing**（生成动作）：模型能否生成可执行的动作序列

并且把动作侧失败再细分为 5 类错误：`format/parsing`, `hallucination`,
`relation_grounding`, `missing_step`, `wrong_order`。

### 0.3 我的论文聚焦在哪里

**主战场是 action sequencing**，不是 goal interpretation。
原因：现有 LLM 在 goal interpretation 上已经接近天花板（top 模型 ~42 F1，
gap 主要是 schema 兼容问题）；而在 action sequencing 上即使 SOTA
也只有 70–88% 任务成功，且 **同一个模型可能 goal 全对、action 全错**——
这就是论文的核心研究对象**目标-动作差距 (goal-to-action gap)**。

---

## 1. 一句话总结

> 我把"LLM 已正确理解目标却无法生成可执行动作序列"这一**目标-动作差距**作为研究对象，
> 在三家族跨厂商面板上完成了 11 个干预变体的受控消融。
> **plan_then_ground prompt-only 干预获得 +4.65 pp 主收益**；
> 我原本主推的两个**知识接地方法**得到了**比预期更复杂、但学术上更有信息量**的结果：
> SG-RAG 出现 **U 形能力依赖**，PC-KG self-check 是 **强负面结果 (−23.25 pp)** 并被诚实报告。

---

## 2. 失败长什么样：三个真实 case

让导师直观看到**到底什么算"goal 对了但 action 错了"**。
（来自 `output/diagnostics/failure_case_studies.md`）

### Case 1：`Turn on light` (gpt-4o, goal F1 = 1.00)

**模型理解了目标完全正确**，但生成的动作序列是：

```json
{"WALK":["home_office","319"]}
{"PLUGIN":["light","411"]}     ← 这盏灯不需要 plug in
{"SWITCHON":["light","411"]}
```

**失败类型**：`relation_grounding` —— 错把"开灯"理解成"先插电再开"。
这是**幻觉物理常识**（这盏灯本来就插着电）。

### Case 2：`Drink` (gpt-4o, goal F1 = 0.80)

```json
[WALK] <dining_room> (201)
[GRAB] <drinking_glass> (1001)
[DRINK] <drinking_glass> (1001)    ← 直接喝了空杯子
[DRINK] <drinking_glass> (1001)
```

**失败类型**：`missing_step` —— 缺少 "先去倒水" 这一步。
模型抓起空杯就喝。这是 **本论文的核心结构性失败**：模型理解任务但漏关键步骤。

### Case 3：`Write an email` (失败类型 `planning_order`)

```json
[WALK] <computer> (319)
[TYPE]                              ← 还没开电脑就开始打字
[SWITCHON] <computer> (319)
```

顺序错误。这是 PC-KG 验证器**最该抓住**的失败之一
（违反 walk-before-act + state-precondition）——但实际部署中 PC-KG 反而帮倒忙，下文会展开。

---

## 3. 整体失败画像（17 个预存模型聚合）

来源：`output/diagnostics/failure_pattern_counts.json`

| 失败类别 | 出现次数 (top 10 模型联合统计) |
| --- | ---: |
| relation_grounding（错对象 id / 关系）| **59** |
| other（不可单独归因）| 40 |
| planning_order（顺序错）| **39** |
| format_or_parsing（格式 / 解析）| 1 |

→ **结论**：现代 LLM 在 EAI/VirtualHome 上的失败几乎不再是格式问题，
而是 **接地 (grounding) + 顺序/缺步 (planning)**——这两类正是论文干预的目标。

---

## 4. 研究问题与论文定位

### 4.1 五个 RQ

| RQ | 内容 | 状态 |
| --- | --- | --- |
| RQ1 | Goal interpretation 与 action sequencing 是否相关？还是应分开报告？ | ✅ 已答（独立技能，散点图证据见 §6.4） |
| RQ2 | 高 goal-F1 但 action 失败时，哪种细粒度错误主导？ | ✅ 已答（grounding + planning_order 主导）|
| RQ3 | 失败画像是否按厂商家族聚类？ | ✅ 已答（家族差异显著） |
| RQ4 | 哪些 prompt-only 干预能缩小差距？ | ✅ 已答（plan_then_ground 主胜） |
| **RQ5** | **SG-RAG / PC-KG / 二者组合是否优于 prompt-only？** | ✅ **已答——结果反直觉** |

### 4.2 论文聚焦在哪一侧

**主战场：action sequencing**。
Goal interpretation 只作为对照（n=200 跑了 4 个变体，无跨家族验证）。
研究主体投入在 action sequencing：主模型 7 个变体 × n=200 +
跨家族 2 个厂商 × 2 个变体 × n=100。

---

## 5. 11 个干预变体概览

### 5.1 Prompt-only 变体（6 个）

| 变体 | 一句话描述 | 额外 token | 额外 LLM 调用 |
| --- | --- | --- | --- |
| `baseline` | EAI 原始 prompt | — | — |
| `format_constraints` | 在 system prompt 强制 JSON schema | +50 | 0 |
| `few_shot_valid_actions` | 注入 3 条合法动作序列示例 | +300 | 0 |
| `plan_then_ground` | 让模型**先输出自然语言 plan，再翻译为 JSON** | +200 | 0 |
| `schema_constrained`（goal 侧）| 强制 goal JSON 字段 schema | +60 | 0 |
| `decompose_then_merge`（goal 侧）| 先列子目标再合并 | +180 | 0 |

### 5.2 Self-correction 变体（1 个）

| `self_check_rewrite` | 第一次 draft 后 + critique + 第二次重写 | ≈2× | +1 |

### 5.3 知识接地变体（3 个，本论文核心 RQ5）

**SG-RAG**（Scene-Graph Retrieval-Augmented Grounding）：
为每个任务，从 VirtualHome 初始场景图里检索任务相关的子图（k-hop 邻居，
最多 20 个物体），以 `[Scene Subgraph] ... [/Scene Subgraph]` 块注入到
prompt 里。**只增 prompt 长度，不增 LLM 调用**。

```
[Scene Subgraph]
node: computer:319 (state=OFF, ON desk:200)
node: desk:200 (in home_office:1)
edge: computer:319 PLUGGED_IN wall_socket:411
[/Scene Subgraph]
```

**PC-KG**（Precondition Knowledge-Graph self-check）：
基于 19 条 VirtualHome 动作规则（`SWITCHON` 必须先 `WALK` 到对象、
`PUTIN` 必须先 `OPEN` 容器、…）和 16 类 violation 编码
（`MISSING_WALK`, `MISSING_OPEN`, `UNKNOWN_ID`, `ARITY_MISMATCH` …），
对模型 draft 做**符号化验证**，把违规列表作为 critique 喂回去重写。

**`sg_rag_pc_kg`**：上述两者组合，原本是论文最大野心。

---

## 6. 三大核心发现

### Finding 1 ─ plan_then_ground 是最稳定的 prompt-only 收益

**主模型 DeepSeek-V4-Flash (n=200)，action sequencing task success：**

| 变体 | 任务成功 | Δ vs baseline | 关键失败列改善 |
| --- | ---: | ---: | --- |
| baseline | 75.58 | — | missing_step 12.79 |
| **plan_then_ground** | **80.23** | **+4.65** | **missing_step 8.14（砍半）** |
| few_shot | 79.65 | +4.07 | parsing → 0% |
| format_constraints | 79.07 | +3.49 | parsing → 0.58% |
| sg_rag | 79.07 | +3.49 | （见 Finding 2）|
| sg_rag_pc_kg | 60.47 | −15.11 | missing_step ↑↑ |
| pc_kg_self_check | 52.33 | **−23.25** | **见 Finding 3** |

![Action sequencing — task success across variants](../output/diagnostics/figures/fig_action_task_success.svg)

![Action sequencing — execution success across variants](../output/diagnostics/figures/fig_action_execution_success.svg)

**解读**：差距是**部分可恢复**的，且收益主要来自"先规划再落地"这一结构化提示，
而非更花哨的方法。

---

### Finding 2 ─ SG-RAG 出现 U 形能力依赖（跨家族）

**baseline → sg_rag 在三个厂商上的差：**

| 家族 | baseline | sg_rag | Δ |
| --- | ---: | ---: | ---: |
| GLM-5-Turbo（最强）| 88.37 | 87.21 | −1.16 |
| MiniMax-M2-Stable（中等）| 86.05 | 79.07 | **−6.98** |
| DeepSeek-V4-Flash（最弱）| 75.58 | 79.07 | **+3.49** |

![Action ablation — three families on baseline + sg_rag (U-shape visible)](../output/diagnostics/figures/fig_ablation_action.svg)

**这是论文中最具发表潜力的发现之一**。预注册预测的"单调改进"被三家族实证打脸：

- **弱模型**（DeepSeek-V4-Flash）：场景先验缺乏，注入子图是**有效信号**
- **中等模型**（MiniMax-M2-Stable）：场景先验已经够用，但长结构化前缀
  反而是**注意力 distractor**（典型 *Lost in the Middle* (Liu et al. 2024)）
- **强模型**（GLM-5-Turbo）：场景先验充分编码，SG-RAG 提供**冗余噪声**，
  小幅伤害

**操作含义**：SG-RAG 不是"加上就赚"的免费午餐——使用前必须 profile 目标模型。

---

### Finding 3 ─ PC-KG self-check 强负面结果（被自己证伪）

**主模型 (n=200)：**

| 变体 | 任务成功 | Δ |
| --- | ---: | ---: |
| baseline | 75.58 | — |
| **pc_kg_self_check** | **52.33** | **−23.25** |
| **sg_rag_pc_kg** | **60.47** | **−15.11** |

**预注册承诺**："PC-KG ≥ +10 pp"
**实际**：−23.25 pp。**方向错、幅度大**。

![Failure profile shift — pc_kg_self_check breaks missing_step / additional_step columns](../output/diagnostics/figures/fig_failure_profile.svg)

**三因素归因（§6.6.1）：**

1. **Critique-induced over-correction**：模型把 violation 信号当作"整段重写"信号，
   重写后 missing_step **几乎翻倍**（12.79 → 23.84）、additional_step **翻倍**
   （4.07 → 8.14）。
2. **Parser-layer false positives**：约 **8% 的 draft** 因 JSON 包装略不规范
   被 verifier 误判为 malformed，模型直接输出空序列 → 0 分。
3. **Verifier coverage gap**：19 条规则只检查**静态前置条件**（walk-before-op 等），
   但 EAI 真正打分的是 **STATE 转移**（节点状态、边关系的最终值）——verifier
   在它最该提示的失败上**沉默**。

**机制 C 是范式层局限**——修复需要换成 state-trace verifier，等同于另一篇论文。

→ 这就是"被自己证伪"的具体含义：**预注册版本在严格条件下净效应为负**，
且**修复路径需要换范式**（不是参数微调）。这与 Huang et al. 2024
《LLMs Cannot Self-Correct Reasoning Yet》在 reasoning 域的负面结论
在 embodied planning 域形成呼应。

---

## 7. Goal interpretation 侧（对照实验）

主模型 DeepSeek-V4-Flash (n=200), `all_f1`：

| 变体 | all_f1 | Δ |
| --- | ---: | ---: |
| baseline | 38.97 | — |
| **schema_constrained** | **40.60** | **+1.63** |
| decompose_then_merge | 39.00 | +0.04 |
| few_shot | 38.97 | 0.00 |

![Goal interpretation — all_f1 across variants](../output/diagnostics/figures/fig_goal_interpretation_f1.svg)

![Goal interpretation ablation](../output/diagnostics/figures/fig_ablation_goal.svg)

**结论**：goal 侧最大收益来自 **schema 约束**而非 RAG 或分解，
印证了 §5.2 的诊断（goal 侧瓶颈是 schema 兼容性）。

---

## 8. 多模型大盘 (RQ1, RQ3)

### 8.1 17 个预存模型 + 3 个新跑模型，按家族平均成功率

![Family average — action task success](../output/diagnostics/figures/fig_family_average.svg)

### 8.2 Goal F1 vs Action Task Success（RQ1 关键证据）

![Goal vs action — independent skills](../output/diagnostics/figures/fig_goal_vs_action.svg)

**回答 RQ1**：goal 与 action 在主流模型上**只有弱相关**——
Pearson r ≈ 0.4，远低于"同一个智能"的预期。**应作为独立技能分别报告**。

### 8.3 Action sequencing leaderboard 节选（17 inventory + 3 new）

| Rank | model | task_success | family |
| ---: | --- | ---: | --- |
| 1 | **glm-5-turbo** (本论文新跑) | **88.37** | China-OS |
| 2 | glm-5-turbo + sg_rag | 87.21 | China-OS |
| 3 | minimax-m2-stable (本论文新跑) | 86.05 | China-OS |
| 4 | deepseek-v4-flash + plan_then_ground | 80.23 | China-OS |
| ... | ... | ... | ... |
| 7 | mistral-large-2402 | 76.39 | Mistral |
| 8 | deepseek-v4-flash (baseline) | 75.58 | China-OS |
| 9 | claude-3-5-sonnet | 72.79 | Anthropic |
| 10 | claude-3-opus | 68.52 | Anthropic |
| 11 | o1-mini | 65.57 | OpenAI |
| 12 | gpt-4o | 63.93 | OpenAI |
| ... | ... | ... | ... |
| − | gold_oracle (评测下限) | 2.62 | Oracle |

**注**：gold_oracle 极低是因为 EAI 评测器对完整动作序列的 strict matching
有已知严格性问题（执行成功 89.2 但任务成功 2.6），这是 §6.8 已记录的局限。

---

## 9. 已完成的工程

| 类别 | 内容 |
| --- | --- |
| Pipeline | 端到端多厂商生成-评测，支持 OpenAI / Claude / Gemini / OpenAI-compatible / dry_run 五种 provider |
| 故障恢复 | incremental JSON 写入、超时/重试、conda-free 评测脚本 |
| 推理模型适配 | `reasoning_content` 字段降级 + `<think>` 标签剥离（兼容 Kimi / MiniMax-M2）|
| 单元测试 | `test_scene_graph_rag.py` (k-hop 检索)、`test_precondition_kg.py` (19 规则验证) |
| LLM 调用 | ≈ 5 400 次，0.06% 错误率（3 transient WARN）|
| 生成 figure | 8 张 SVG（已嵌入本文档）|

---

## 10. 论文章节状态

| 章节 | 文件 | 状态 |
| --- | --- | --- |
| Abstract | `paper/00_abstract.md` | ✅ 已写入实际数字 |
| §1 Introduction + §2 Related Work | `paper/01_introduction_and_related_work.md` | ✅ 已回收 PC-KG 过度承诺 |
| §3 Experimental Setup | `paper/02_experimental_setup.md` | ⚠️ 主体完成，新数据需 minor 同步 |
| §4 Multi-model Evaluation | （依赖 §3 figure）| ⚠️ 新 figure 已生成、文字待补 |
| §5 Failure Diagnosis | `paper/03_failure_diagnosis.md` | ✅ 主体完成 |
| §6 Methods for Improvement | `paper/06_methods_for_improving_success.md` | ✅ **完整重写**（含 §6.5.1 U 形 + §6.6.1 负面结果）|
| §7 Discussion | `paper/07_discussion_and_conclusion.md` | ✅ 已写 |
| §8 Conclusion + Future Work | `paper/07_discussion_and_conclusion.md` | ✅ 已写 |

---

## 11. 论文贡献（已对齐到实际结果）

1. 一个可复现的多厂商生成-评测 pipeline，支持三家族跨验证（罕见于 master 论文）；
2. 一个把 goal interpretation 与 action sequencing 拆分报告、含 5 大类细粒度错误的诊断框架；
3. **SG-RAG 的 U 形能力依赖经验曲线**（refines "RAG helps weak models" 的传统主张）；
4. **静态前置条件 self-check 的强负面结果 + 三因素归因**（为后续
   verifier-in-the-loop 研究提供"避坑地图"）；
5. 一个可复用的规则化 verifier (`analysis/precondition_kg.py`)：作为
   deployed corrector 失败，但作为 **per-violation 诊断工具**仍有价值。

---

## 12. 与同类工作的相对位置

| 项 | 本论文 | 平均 master 论文 | 平均 NeurIPS workshop |
| --- | --- | --- | --- |
| 评测模型数量 | 17 inventory + 3 ablation | 1–2 | 3–5 |
| 干预变体数 | 11 | 2–4 | 5–8 |
| 跨家族验证 | ✅ 三家族 | ❌ | 部分有 |
| 诚实报告负面结果 | ✅ 强负面 + 归因 | ❌ 通常隐藏 | 极少 |
| 诊断粒度 | 5 大类 + per-violation | 通常只报 task success | 常报 |

**评估**：作为 master 论文水准超过平均线。如要尝试发表，方向是 EMNLP/AAAI 的
**system-or-evaluation track short paper**，或 NeurIPS workshop（LLM Agents, FMDM）。

---

## 13. 已知局限性（已写入 §6.8）

1. **Prompt overfitting**：变体设计基于 inventory 失败画像，n=100 跨家族控制是缓解但非根除。
2. **Symbolic-only environment**：所有反馈是符号化的，向像素接地环境的迁移性未验证。
3. **PC-KG 规则覆盖缺口**：见 Finding 3 机制 C。
4. **Reasoning-model API 不兼容**：Kimi-K2.5 因 `reasoning_content` 字段路由 +
   长 prompt 下 27k–38k token 占用 + 220s/请求延迟，无法纳入跨家族面板
   （详见 §6.8）；已替换为 MiniMax-M2-Stable。
5. **Goal interpretation 仅单家族**：跨家族验证只在 action 侧做，资源约束。

---

## 14. 尚待决策的事项（请导师指示）

### 14.1 关于"被证伪的 PC-KG"如何在论文里定位（**最重要**）

**两条路线，需选一：**

- **(a) 当前路线（推荐）**：把 PC-KG 作为**强负面结果 + 三因素归因**写入论文核心贡献，
  以 *Huang et al. 2024 (LLMs cannot self-correct yet)* 的"诚实负面结果"范式为参照。
- **(b) 替代路线**：把 PC-KG 降为附录的"探索性尝试"、不进核心贡献，
  论文以 SG-RAG U 形 + plan_then_ground 为主线。

**我的判断**：(a) 学术价值更高、答辩更稳；(b) 安全但偏保守。**等导师裁决**。

### 14.2 是否需要做开源权重 (Llama-3 / Qwen-3 / Mistral) 复现来验证 U 形

**Pros**：能确认 U 形不是闭源 RLHF 伪影；
**Cons**：再加 3 家族 ≈ 3–5 天的 GPU/调用预算 + 1 周写作。
**等导师评估时间窗口**。

### 14.3 是否需要做 state-trace verifier 把 PC-KG "救回来"

**我的强烈建议**：**不做**。这等同于另一篇论文，且本论文的边界陈述已经把
"静态前置条件 verifier 的局限"作为完整可发表结论。
**等导师确认是否同意把它放到 future work**。

### 14.4 投稿计划

是否在毕业答辩之前/之后尝试投 EMNLP-short / AAAI-short / NeurIPS workshop？
需要导师确认目标会议与时间表。

### 14.5 答辩 PPT

未开始。等导师确认上述 14.1–14.4 后再开始撰写，避免做完返工。

---

## 15. 关键交付物索引

| 类别 | 路径 |
| --- | --- |
| 论文章节 | `paper/00_abstract.md`, `01_introduction_and_related_work.md`, `02_experimental_setup.md`, `03_failure_diagnosis.md`, `06_methods_for_improving_success.md`, `07_discussion_and_conclusion.md` |
| 核心代码模块 | `analysis/scene_graph_rag.py`, `analysis/precondition_kg.py`, `analysis/generate_outputs.py`, `analysis/prompt_variants.py` |
| 主结果 CSV | `output/diagnostics/multimodel_ablation_summary.csv` |
| 失败画像 | `output/diagnostics/multimodel_failure_profile.csv`, `failure_pattern_counts.json` |
| 失败 case 研究 | `output/diagnostics/failure_case_studies.md` |
| Figure | `output/diagnostics/figures/*.svg`（8 张，本文档已嵌入）|
| 单元测试 | `analysis/tests/test_scene_graph_rag.py`, `test_precondition_kg.py` |
| 工作流总结 | `.comate/specs/sg-rag-pc-kg-recovery/summary.md` |
| 原始 LLM 输出 | `output/improvement_run/helm_output/...` |

---

## 16. 术语速查 (Glossary)

| 术语 | 含义 |
| --- | --- |
| **EAI** | Embodied Agent Interface, NeurIPS 2024 评测框架 |
| **VirtualHome** | 家庭模拟环境，提供动作语言 + 场景图 |
| **Goal interpretation** | 把自然语言任务翻译成形式化目标（节点状态 / 边关系）|
| **Action sequencing** | 输出可执行的 VirtualHome 动作序列 |
| **task_success** | 执行后最终状态是否满足目标（最终指标）|
| **execution_success** | 动作序列是否能跑通（不一定满足目标）|
| **all_f1** | goal 侧综合 F1（节点 + 边 + 动作）|
| **missing_step** | 缺关键步骤（如忘了 WALK）|
| **wrong_order** | 顺序错（如先 SWITCHON 后 WALK）|
| **relation_grounding** | 错对象 id / 错空间关系 |
| **hallucination** | 生造场景里不存在的对象 |
| **SG-RAG** | Scene-Graph Retrieval-Augmented Grounding（本论文方法 1）|
| **PC-KG** | Precondition Knowledge-Graph self-check（本论文方法 2）|
| **One API** | 用户使用的多 LLM 网关 |
| **n=200 / n=100** | 评测时使用的样本数（每个变体）|
| **Δ pp** | percentage point 差，用于跨变体对比 |
| **预注册 (pre-registration)** | 实验前公开承诺可证伪的预测，是科学严谨性的体现 |

---

## 17. 一句话回到开头

> **"原本最想推的方法 (PC-KG) 在严格预注册条件下被证伪；但失败被诊断为三个独立机制，
> 加上 SG-RAG 的 U 形意外发现和 plan_then_ground 的稳定 +4.65 pp，
> 让这篇论文从'又一篇 RAG/KG 改进'升级为'有信息量的诊断+负面结果论文'。
> 现在最关键的决策是：导师是否同意以负面结果作为核心贡献来组织论文。"**

—— 期待您的反馈。
