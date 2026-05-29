# 本周汇报：从 Prompt 方法到 Knowledge-Grounded Planning Framework

**日期**：2026-05-26  
**汇报人**：吴子涵  
**论文题目**：*Bridging the Goal-to-Action Gap: A Diagnostic Study and Knowledge-Grounded Recovery of LLM Failures in Embodied Planning*  
**实验环境**：EAI / VirtualHome action sequencing  

---

## 1. 本周主要结论

本周的重点是继续寻找一个真正能提升 EAI / VirtualHome 动作规划成功率的方法。我先继续测试了几种 prompt 方法，确认了在 prompt-only 这个层面上，再加复杂结构的提升空间已经比较有限。因此本周后半部分把方向调整为：**从 prompt-only 方法，转向规划一个真正基于外部知识库的 knowledge-grounded planning framework**。

需要诚实地说明一点：之前在代码里写过的 `Scene Graph RAG` 和 `Precondition KG` 模块，本质上只是**在 prompt 前面拼接结构化信息 + 在内存里硬编码动作规则**，没有向量检索、没有图数据库、没有持久化。这种实现严格来说不属于 RAG / KG，更接近 "structured prompt injection + rule-based verifier"，本周已经把它从论文方法主线中去除，只保留 prompt-only 的实验作为当前的实证结果。

目前最稳定的结果仍然是 `plan_then_ground`，它把 DeepSeek-V4-Flash 的 task success 从 `75.58%` 提高到 `80.23%`。其余 prompt 变体作为消融。整体结果如下：

| 方法 | 方法类型 | Task success | 主要现象 |
| --- | --- | ---: | --- |
| `baseline` | 原始 EAI prompt | 75.58% | 基线 |
| `plan_then_ground` | 轻量 prompt 方法 | **80.23%** | 当前最强，明显降低 missing step |
| `goal_conditioned_scaffold` | 目标倒推 prompt | 79.65% | relation goal 提升，但不超过主方法 |
| `state_checklist_plan` | checklist prompt | 79.07% | execution success 高，但 additional step 增加 |
| `bidirectional_causal_planning` | 双向因果 prompt | 74.43% | relation goal 很高，但 missing step 变差 |

本周最重要的判断不是 "某个新方法已经超过了 `plan_then_ground`"，而是：

> 单纯增加 prompt 复杂度不可靠；要真正提升 task success，需要把 RAG 和 KG 做成**持久化、可查询、可被 agent 反复调用**的外部知识基础设施，而不是塞在 prompt 里的一段文本。

---

## 2. 目前最稳的方法：`plan_then_ground`

`plan_then_ground` 不是 EAI 原论文中的方法，而是本项目在 EAI action sequencing 任务上设计的轻量 prompt 变体。它的思想可以用中文概括为：**种因得果**。也就是说，不让模型一开始就直接写动作，而是先让模型在内部想清楚高层计划，再把这个计划落地成 VirtualHome JSON 动作序列。

代码中的提示思想大致是：

```text
Step 1: Privately think through a short high-level plan.
Step 2: Convert the plan into the compact JSON action sequence.
Output ONLY the JSON sequence; never reveal the plan.
```

效果如下：

| 方法 | Task success | Execution success | Missing step | Additional step |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 75.58% | 82.6% | 12.79% | 4.07% |
| `plan_then_ground` | **80.23%** | **86.6%** | **8.14%** | 5.23% |

它最主要的贡献是降低 `missing_step`。例如任务是 `Read book`，模型不能只输出 `READ book`，而需要先 `WALK book -> GRAB book -> READ book`。`plan_then_ground` 让模型先想清楚这些中间步骤，再生成动作。

---

## 3. 本周尝试一：目标倒推 Prompt

在 `plan_then_ground` 的基础上，我尝试了另一种思路：不是从任务出发正向规划，而是从最终目标倒推必要动作。这个方法叫 `goal_conditioned_scaffold`，中文可以理解为：**由果倒推因**。

```text
最终目标条件 -> 直接实现目标的动作 -> 必要前置步骤 -> JSON 动作序列
```

例：

```text
Goal: book is read / held
Skeleton: WALK book -> GRAB book -> READ book

Goal: cup INSIDE dishwasher
Skeleton: WALK cup -> GRAB cup -> WALK dishwasher -> OPEN dishwasher -> PUTIN cup dishwasher
```

| 方法 | Task success | Relation goal | Action goal | Missing step |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 75.58% | 64.41% | 62.67% | 12.79% |
| `plan_then_ground` | **80.23%** | 67.80% | **73.33%** | **8.14%** |
| `goal_conditioned_scaffold` | 79.65% | 71.19% | 70.67% | 8.72% |

这个方法有效但没有超过 `plan_then_ground`：relation goal 更高，说明目标倒推有助于关系目标落地；缺点是它没有进一步降低 missing step。

---

## 4. 本周尝试二：双向因果规划

之后我尝试把"种因得果"和"由果倒推因"结合起来，设计了 `bidirectional_causal_planning`：

```text
先由果倒推因：从最终目标反推关键动作
再种因得果：按可执行顺序展开动作
```

理论上它应该同时具备两个优点，但实际结果不理想：

| 方法 | Task success | Relation goal | Action goal | Missing step |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 75.58% | 64.41% | 62.67% | 12.79% |
| `bidirectional_causal_planning` | 74.43% | **80.00%** | 61.49% | 17.05% |

复杂 prompt 让模型更关注最终关系目标，所以 relation goal 明显提高；但同时它反而漏掉更多中间执行步骤，导致 missing step 上升、task success 下降。这个负面结果对论文是有价值的：

> Prompt 不是越复杂越好。对 embodied planning 来说，过多推理要求可能会干扰模型生成完整动作序列。

---

## 5. 为什么需要真正的外部知识基础设施

前面的 prompt 实验说明，继续只在 prompt 上做文章已经不够了。EAI/VirtualHome 的失败不是单纯"模型没想清楚"，还包括两个更具体的问题：

1. **场景 grounding 问题**：模型需要知道有哪些对象、对象 id 是什么、对象状态和关系是什么。VirtualHome 单个场景平均 287 个节点 / 5690 条边，全量塞进 prompt 不现实，必须按需检索。
2. **动作前置条件问题**：模型需要知道每个动作之前必须满足什么条件，比如 `READ` 前要 `GRAB`，`PUTIN` 前要 `OPEN` 容器。规则数量虽小，但需要和**历史失败案例**关联起来才能给出有针对性的反馈。

要解决这两个问题，光靠在 prompt 里拼一段文字是不够的。需要的是：

- **真正的向量检索 RAG**：embedding 模型 + 向量数据库 + 持久化索引，可以按语义检索任务相关对象，而不是关键词匹配。
- **真正的图数据库 KG**：把场景图、动作规则和失败案例同时建模到图数据库中，支持 Cypher k-hop 查询，能反查"过去同类失败案例"。

> Prompt 负责规划流程，向量库负责场景信息检索，图数据库负责动作约束和失败模式查询。

这才是论文题目里 **Knowledge-Grounded Recovery** 真正应该承载的方法。

---

## 6. 下一阶段方案：持久化 RAG + KG 知识库（已设计、待执行）

本周完成了完整的工程方案设计，**所有代码已提交到仓库**（`analysis/kb/` 包），但尚未实际跑通 bootstrap（需要 Docker + 安装依赖）。具体方案如下。

### 6.1 选型

| 组件 | 选型 | 理由 |
| --- | --- | --- |
| 向量数据库 | **Chroma**（本地 sqlite + parquet 持久化） | 零运维、单机够用、便于论文复现 |
| Embedding 模型 | **BAAI/bge-small-en-v1.5**（384 维，本地推理） | 不依赖外部 API、效果优于 MiniLM |
| 图数据库 | **Neo4j 5**（Docker 启动） | 成熟的 Cypher 查询、图可视化、社区方案 |
| Drop-in 切换 | 环境变量 `KB_BACKEND=persistent` | 离线评审或没有 Docker 时自动回退到原内存实现，保证论文实验可复现 |

### 6.2 RAG 层：双 collection 向量库

- `scene_objects`：每个 (file_id, node_id) 一条文档；document = `"{class_name}; properties: ...; states: ..."`；元数据含 `file_id, scene_id, node_id, class_name, category, properties, states`。共 518 个场景 × 平均 287 节点 ≈ **15 万向量**。
- `failure_cases`：每条历史失败案例一条文档（来自 `output/diagnostics/`）；document = `"task=... failure_type=... :: <raw_text>"`；元数据含 `failure_type, model, file_id`。

检索接口：
```python
PersistentSceneGraphRetriever.retrieve(
    identifier="11_1",
    task_prompt="Read the book on the bedside table",
    k_neighbours=1,
    max_objects=20,
) -> str  # 返回 [Scene Subgraph] 文本块（保持与原版字节级一致）
```

### 6.3 KG 层：三层 Neo4j 图谱

```cypher
// 第一层：场景图实例（来自 EAI 数据集 518 个 JSON）
(:Scene {file_id})-[:CONTAINS]->(:Object {file_id, node_id, class_name, properties, states})
(:Object)-[:RELATION {type}]->(:Object)

// 第二层：动作规则 schema（来自 22 条 ActionRule）
(:Action {name, arity})-[:REQUIRES_PROP {slot}]->(:Property {name})
(:Action)-[:REQUIRES_STATE]->(:Precondition {kind})
(:Action)-[:PRODUCES]->(:Effect {kind})

// 第三层：失败案例链路（来自 output/diagnostics/）
(:FailureCase {uid, file_id, model, failure_type, task})-[:OCCURRED_IN]->(:Scene)
(:FailureCase)-[:VIOLATES]->(:Action)
```

关键查询能力（这正是当前内存版做不到的）：

- "在 file 11_1 这个场景下，从 book(123) 出发 1 跳能到达哪些对象？" → Cypher 一行
- "过去所有违反 GRAB 前置条件的失败案例里，哪个 failure_type 占比最高？" → 反查 `(:FailureCase)-[:VIOLATES]->(:Action {name:'GRAB'})`
- "动作 PUTIN 需要满足哪些前置状态？" → schema 子图直接读

### 6.4 Knowledge-Grounded Planning Agent v2（下一步）

有了真正的持久化 KG / RAG 之后，agent 流程会变成：

```text
Task instruction
    ↓
Chroma 语义检索 -> seed objects（按任务语义找最相关对象）
    ↓
Neo4j Cypher k-hop 扩展 -> 紧凑场景子图
    ↓
LLM plan-then-ground 生成 draft
    ↓
Neo4j 规则子图加载 -> 检查 draft 的前置条件
    ↓
若违反规则：Cypher 反查同类失败案例 -> few-shot 注入修复 prompt
    ↓
保守 local repair -> 输出 VirtualHome JSON
    ↓
EAI evaluator 评测
```

相比之前那个内存版 agent，最大区别有三点：

1. **可问历史**：当 verifier 发现违反 `MISSING_WALK` 时，可以查询过去同类失败案例，把它们作为 few-shot 注入到修复 prompt，而不是让 LLM 凭空想。
2. **可问场景**：不再是关键词匹配 + k-hop 暴力扩展，而是 BGE 语义检索找最相关 seed，再用 Cypher 精准扩展，可以处理"模型不知道场景里到底叫 'book' 还是 'novel'"这种语义模糊问题。
3. **可问规则**：规则源在图数据库里，未来可以加新动作类型（如 BEHAVIOR-1K 的更多动作），不用改硬编码。

### 6.5 迭代式 Harness：让 bad case 自动回流到知识库

光有"知识库 + agent"还不够，还需要一套**评测—回流—再评测**的闭环 harness，让每一轮跑出来的失败案例（bad case）自动入库，下一轮再被同类任务检索到。这是把"诊断—改进"从一次性脚本变成持续学习循环的关键。

```text
┌──────────────────────────────────────────────────────────────┐
│   Iteration N                                                │
│                                                              │
│   1. Agent v2 生成 -> EAI evaluator 评测                     │
│   2. summary.json 解析 -> 划分 pass / fail                   │
│   3. 对每个 fail：                                            │
│      - 用 PreconditionKG.verify() 提取 violation code        │
│      - 写入 FailureCase 节点：                                │
│          (:FailureCase {iteration:N, file_id, model,         │
│                         failure_type, task, raw, draft})     │
│        -[:OCCURRED_IN]->(:Scene)                             │
│        -[:VIOLATES]->(:Action)                               │
│      - 同时 upsert 到 Chroma failure_cases collection         │
│   4. 报告：每轮的 task success / 新增 failure / 修复 failure  │
│                                                              │
└────────────┬─────────────────────────────────────────────────┘
             │ 失败案例已入库
             ▼
┌──────────────────────────────────────────────────────────────┐
│   Iteration N+1                                              │
│                                                              │
│   - 同一个 task 再遇到时，agent 检索历史失败：               │
│     "this task previously failed with MISSING_WALK on book   │
│      in scene 11_1; the gold sequence inserts WALK before    │
│      GRAB."                                                  │
│   - 把检索到的过去失败 + 修复方式作为 few-shot 注入 prompt   │
│   - 再生成 -> 再评测 -> 再回流                                │
└──────────────────────────────────────────────────────────────┘
```

**Harness 职责**（即将在 `analysis/kb/harness.py` 中实现）：

| 功能 | 接口 | 数据流 |
| --- | --- | --- |
| 跑一轮评测 | `harness.run_iteration(model, variant, task_ids) -> RunReport` | 调 `generate_outputs.py` + EAI evaluator，得到 per-task summary |
| 划分 bad case | `harness.collect_bad_cases(run_report) -> List[BadCase]` | 解析 `summary.json` 中 `task_success=False` 的样本 |
| 回流到 KG/RAG | `harness.ingest_bad_cases(bad_cases, iteration_id)` | 写入 Neo4j `FailureCase` + Chroma `failure_cases` |
| 跨轮对比 | `harness.diff_iterations(run_a, run_b) -> Diff` | 哪些 fail 被修复了、哪些新失败了、整体 success delta |
| 收敛判断 | `harness.has_converged(history) -> bool` | 连续 K 轮 success 增量 < ε，停止迭代 |

**关键设计点**：

1. **以 `iteration_id` 作为 FailureCase 节点的额外维度**，可以分析"这个 task 在第几轮才被修复"，做出收敛曲线。
2. **保留每一版的 draft action sequence** 在 `FailureCase.raw` 里，未来做 case study 时不需要重跑。
3. **Bad case 回流是单向的**：原始数据集（gold sequence）不变，只把 LLM 生成的失败注入 KG。这避免污染 evaluator 的 ground truth。
4. **Harness 必须能在 dry-run provider 上跑通**：保证论文复现不依赖具体 LLM API。
5. **接口与现有 `scripts/run_action_sequencing_eval.sh` 对齐**：迭代版只是把它包成 Python 函数 + 在循环外加 ingest 步骤，不另起一套评测体系。

### 6.6 论文中的实验设计

围绕 harness，可以做出三组核心实验：

| 实验 | 描述 | 预期结论 |
| --- | --- | --- |
| **E1: Static KB** | 一次性建库（只用 `output/diagnostics/` 里已有的失败案例），不迭代 | 验证持久化 KG/RAG 本身的增量 |
| **E2: Iterative KB** | 跑 N 轮，每轮把 bad case 回流，下一轮用 | 验证迭代是否带来累积提升 |
| **E3: Convergence study** | 记录每轮 task success / 修复数 / 新失败数，画收敛曲线 | 找到收益拐点（K 轮后边际 < ε） |

E2/E3 是这套 harness 真正的论文贡献：把传统的"一次评测"扩展成"持续诊断—修复—再评测"的循环，并且整个循环都是可复现、可审计的（每个 FailureCase 都有 iteration_id 和 raw draft）。

---

## 7. 实施进度与风险

### 已完成（已提交至 GitHub）
- `analysis/kb/config.py` — 路径/模型/Neo4j 连接（环境变量可覆写）
- `analysis/kb/build_vector_store.py` — 一次性构建 Chroma（场景 + 失败案例）
- `analysis/kb/build_graph_db.py` — 一次性写入 Neo4j（Scene / Object / RELATION / Action 规则子图 / FailureCase 链路）
- `analysis/kb/persistent_retriever.py` — `PersistentSceneGraphRetriever`（drop-in 替换原 retriever）
- `analysis/kb/persistent_kg.py` — `PersistentPreconditionKG`（drop-in 替换原 verifier）
- `analysis/kb/schema.cypher` — Neo4j 唯一性约束 + 索引
- `analysis/kb/harness.py` — 迭代 harness 框架骨架（接口 + 数据结构 + 单轮 stub）
- `scripts/start_neo4j.sh` — Docker 启动 Neo4j 5.20
- `scripts/build_knowledge_base.sh` — 一键 bootstrap
- `requirements-kb.txt` — chromadb / sentence-transformers / neo4j

### 待执行
1. 安装 Docker + `pip install -r requirements-kb.txt`
2. 跑 `bash scripts/build_knowledge_base.sh` —— 完成 Chroma 索引和 Neo4j 灌库
3. 把 `harness.run_iteration()` 内部的 stub 替换成真实的 generate + evaluate 调用
4. 实现 v2 agent 的"失败案例 few-shot 注入"修复策略
5. 跑 E1 / E2 / E3 三组实验，得到 Iterative KB 的收敛曲线
6. 重跑 EAI evaluator，对比 `plan_then_ground` 与 v2 agent + iterative harness

### 风险与回退
- Neo4j 需要 Docker：离线评审环境可能不可用 → 已实现自动回退到原内存实现，论文复现不会被卡住。
- BGE 模型首次下载约 130MB → 已配置缓存到 `data/kb/models/`。
- 工程量较大：bootstrap + 实验跑完估计需要本周后半到下周。

---

## 8. 论文方法线索的重新组织

我建议论文中这样组织方法线索：

### 第一阶段：Prompt-only（已完成，主结果）
- **`plan_then_ground`** 作为最强的 prompt-only 方法（task success 75.58 → 80.23）。
- **`state_checklist_plan` / `goal_conditioned_scaffold` / `bidirectional_causal_planning`** 作为消融，说明 prompt 复杂度有上限。
- 关键发现：prompt 越复杂不一定越好，过多推理要求会干扰动作完整性。

### 第二阶段：Knowledge-Grounded Planning Framework with Iterative Harness（下一步）
- **持久化 RAG**（Chroma + BGE）：解决场景 grounding。
- **持久化 KG**（Neo4j 三层：场景/规则/失败链路）：解决动作约束 + 历史失败查询。
- **Planning Agent v2**：把检索、生成、验证、案例反查、保守修复串成完整 pipeline。
- **Iterative Harness**：每轮把 bad case 自动回流到 KG/RAG，下一轮用，画收敛曲线。
- 实验目标：在保留 `plan_then_ground` 收益的基础上，进一步降低 `missing_step` 与 `relation_grounding` 失败，并展示迭代带来的累积提升。

---

## 9. 可以对老师这样说

这周我先继续测试了几种 prompt 方法。结果发现，`plan_then_ground` 仍然是最稳定的，能把成功率从 `75.58%` 提到 `80.23%`。我也尝试了目标倒推和双向因果规划，发现 prompt 越复杂不一定越好。

更重要的是，我对之前的 RAG / KG 模块做了一次诚实的复盘：那两个模块本质上只是 prompt 前缀拼接和内存里的规则查询，并不算真正的 RAG 或 KG。所以本周后半部分我把方向调整成：设计一个真正基于向量数据库（Chroma + BGE）和图数据库（Neo4j）的持久化知识库。代码已经写完并提交到 GitHub，包括建库脚本、drop-in 替换接口、和环境变量切换机制。下一步是实际跑通 bootstrap、把失败案例反查接进 agent，并重新跑 EAI 评测。

---

## 10. 下一步计划

1. 安装 Docker，跑通 `scripts/build_knowledge_base.sh`，得到一个真实可查询的 Chroma 向量库 + Neo4j 图谱。
2. 把 `analysis/kb/harness.py` 里的 stub 替换成真实评测调用，跑通**单轮闭环**（generate → evaluate → 划分 bad case → 回流 KG/RAG）。
3. 实现 v2 Planning Agent：在 verifier 报错时，从 Neo4j / Chroma 反查同类失败案例，作为 few-shot 注入修复 prompt。
4. 跑 E1（静态 KB）/ E2（迭代 KB）/ E3（收敛曲线）三组对比实验，记录每轮 task success / 修复数 / 新失败数。
5. 论文中把方法主线写成两阶段：第一阶段 prompt-only 的实证发现，第二阶段 knowledge-grounded framework + iterative harness 的设计与初步结果。
6. 在 `paper/` 中追加方法学小节，正式描述持久化 RAG / KG 架构与迭代 harness 的查询接口和数据流。
