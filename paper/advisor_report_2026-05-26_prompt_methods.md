# 本周汇报：从 Prompt 方法到 Knowledge-Grounded Planning Agent

**日期**：2026-05-26  
**汇报人**：吴子涵  
**论文题目**：*Bridging the Goal-to-Action Gap: A Diagnostic Study and Knowledge-Grounded Recovery of LLM Failures in Embodied Planning*  
**实验环境**：EAI / VirtualHome action sequencing  

---

## 1. 本周主要结论

本周的重点是继续寻找一个真正能提升 EAI / VirtualHome 动作规划成功率的方法。最开始我主要尝试 prompt 方法，后来发现只在 prompt 上继续加复杂结构，提升空间比较有限。因此，本周后半部分把方向调整为：**从 prompt-only 方法，转向结合 RAG、KG 和固定流程 agent 的 knowledge-grounded planning framework**。

目前最稳定的结果仍然是 `plan_then_ground`，它把 DeepSeek-V4-Flash 的 task success 从 `75.58%` 提高到 `80.23%`。但新的 RAG/KG/agent 实验也提供了一个重要结论：外部知识确实能改善 object grounding 和 relation grounding，只是当前版本还没有解决最核心的 `missing_step` 问题。

整体结果如下：

| 方法 | 方法类型 | Task success | 主要现象 |
| --- | --- | ---: | --- |
| `baseline` | 原始 EAI prompt | 75.58% | 基线结果 |
| `plan_then_ground` | 轻量 prompt 方法 | **80.23%** | 当前最强，明显降低 missing step |
| `goal_conditioned_scaffold` | 目标倒推 prompt | 79.65% | relation goal 提升，但不超过主方法 |
| `state_checklist_plan` | checklist prompt | 79.07% | execution success 高，但 additional step 增加 |
| `bidirectional_causal_planning` | 双向因果 prompt | 74.43% | relation goal 很高，但 missing step 变差 |
| `kg_rag_plan_then_ground` | RAG + KG + plan prompt | 76.39% | grounding 改善，但动作完整性不足 |
| `kg_planning_agent` | 固定流程 knowledge-grounded agent | 76.72% | 比 RAG+KG prompt 略好，但仍不如 `plan_then_ground` |

因此，本周最重要的结论不是“某个复杂方法已经超过了 `plan_then_ground`”，而是：

> 单纯增加 prompt 复杂度并不可靠；RAG 和 KG 能改善 grounding，但要真正提升 task success，需要把它们做成更强的、可控的 planning agent，而不是只作为 prompt 前缀。

---

## 2. 目前最稳的方法：`plan_then_ground`

`plan_then_ground` 不是 EAI 原论文中的方法，而是本项目在 EAI action sequencing 任务上设计的轻量 prompt 变体。它的思想可以用中文概括为：**种因得果**。

也就是说，不让模型一开始就直接写动作，而是先让模型在内部想清楚高层计划，再把这个计划落地成 VirtualHome JSON 动作序列。

代码中的提示思想大致是：

```text
Step 1: Privately think through a short high-level plan.
Step 2: Convert the plan into the compact JSON action sequence.
Output ONLY the JSON sequence; never reveal the plan.
```

这个方法的效果目前最稳定：

| 方法 | Task success | Execution success | Missing step | Additional step |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 75.58% | 82.6% | 12.79% | 4.07% |
| `plan_then_ground` | **80.23%** | **86.6%** | **8.14%** | 5.23% |

它最主要的贡献是降低 `missing_step`。例如任务是 `Read book`，模型不能只输出 `READ book`，而需要先 `WALK book -> GRAB book -> READ book`。`plan_then_ground` 的作用就是让模型先想清楚这些中间步骤，再生成动作。

---

## 3. 本周尝试一：目标倒推 Prompt

在 `plan_then_ground` 的基础上，我尝试了另一种思路：不是从任务出发正向规划，而是从最终目标倒推必要动作。这个方法叫 `goal_conditioned_scaffold`，中文可以理解为：**由果倒推因**。

它的逻辑是：

```text
最终目标条件 -> 直接实现目标的动作 -> 必要前置步骤 -> JSON 动作序列
```

例如：

```text
Goal: book is read / held
Skeleton: WALK book -> GRAB book -> READ book

Goal: cup INSIDE dishwasher
Skeleton: WALK cup -> GRAB cup -> WALK dishwasher -> OPEN dishwasher -> PUTIN cup dishwasher
```

实验结果：

| 方法 | Task success | Relation goal | Action goal | Missing step |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 75.58% | 64.41% | 62.67% | 12.79% |
| `plan_then_ground` | **80.23%** | 67.80% | **73.33%** | **8.14%** |
| `goal_conditioned_scaffold` | 79.65% | 71.19% | 70.67% | 8.72% |

这个方法有效，但没有超过 `plan_then_ground`。它的优点是 relation goal 更高，说明目标倒推有助于关系目标落地；缺点是它没有进一步降低 missing step。

---

## 4. 本周尝试二：双向因果规划

之后我尝试把“种因得果”和“由果倒推因”结合起来，设计了 `bidirectional_causal_planning`。它的想法是：

```text
先由果倒推因：从最终目标反推关键动作
再种因得果：按可执行顺序展开动作
```

理论上，这个方法应该同时具备两个优点：

- 目标倒推可以避免漏掉最终目标；
- 正向规划可以保证动作顺序自然可执行。

但实际结果不理想：

| 方法 | Task success | Relation goal | Action goal | Missing step |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 75.58% | 64.41% | 62.67% | 12.79% |
| `bidirectional_causal_planning` | 74.43% | **80.00%** | 61.49% | 17.05% |

这个实验说明，复杂 prompt 会让模型更关注最终关系目标，所以 relation goal 明显提高；但同时它反而漏掉更多中间执行步骤，导致 missing step 上升，task success 下降。

这个结果对论文是有价值的，因为它说明：

> Prompt 不是越复杂越好。对 embodied planning 来说，过多推理要求可能会干扰模型生成完整动作序列。

---

## 5. 为什么转向 RAG 和 KG

前面的 prompt 实验说明，继续只在 prompt 上做文章已经不够了。EAI/VirtualHome 的失败不是单纯“模型没想清楚”，还包括两个更具体的问题：

1. **场景 grounding 问题**  
   模型需要知道有哪些对象、对象 id 是什么、对象状态和关系是什么。

2. **动作前置条件问题**  
   模型需要知道每个动作之前必须满足什么条件，比如 `READ` 前要 `GRAB`，`PUTIN` 前要 `OPEN` 容器。

因此，本周后半部分我把方法方向调整为：

> Prompt 负责规划流程，RAG 负责提供场景信息，KG 负责提供动作前置条件。

这比单纯 prompt 更符合论文题目里的 **Knowledge-Grounded Recovery**。

---

## 6. 本周尝试三：`kg_rag_plan_then_ground`

这个方法是一个 single-pass 的 knowledge-grounded 版本。它不是只改 prompt，而是在生成前加入两个外部知识源：

- **Scene Graph RAG**：检索任务相关对象、id、状态和关系；
- **Precondition KG**：提供任务相关动作的前置条件规则；
- **Plan-then-ground**：让模型在这些知识约束下生成动作。

流程如下：

```text
Task instruction
        ↓
Scene Graph RAG
        ↓
Precondition KG guidance
        ↓
Plan-then-ground generation
        ↓
VirtualHome JSON action sequence
```

实验结果：

| 方法 | Task success | Execution success | Relation goal | Hallucination | Missing step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 75.58% | 82.6% | 64.41% | 2.33% | 12.79% |
| `kg_rag_plan_then_ground` | 76.39% | 83.0% | **76.67%** | **1.97%** | 12.79% |

这个方法比 baseline 有小幅提升，并且 relation goal 和 hallucination 都改善了。这说明 RAG/KG 的确帮助模型更好地找到对象和关系。但是，它没有解决 missing step，导致总成功率没有超过 `plan_then_ground`。

---

## 7. 本周尝试四：Knowledge-Grounded Planning Agent

接着我进一步做了一个固定流程的 `kg_planning_agent`。这个 agent 不是开放式、多轮自由 agent，而是一个可复现、能被 EAI evaluator 直接评测的固定流程 planning framework。

它的流程是：

```text
RAG + KG 生成 draft
        ↓
加载当前任务的 scene graph
        ↓
用 Precondition KG 检查动作前置条件
        ↓
只做保守 local repair
        ↓
输出最终 VirtualHome action sequence
        ↓
EAI evaluator 评测
```

这里的关键是：不让 LLM 自由重写整个计划，因为之前 `pc_kg_self_check` 的结果说明，LLM 自我纠错容易 over-correct。因此这个 agent 只做高置信局部修复，例如插入 `WALK`、`GRAB`、`OPEN` 这类前置动作，不删除、不重排、不大幅改写原动作序列。

实际执行中，agent 修改了 `25 / 342` 条样本，总共插入了 `49` 个动作。

实验结果：

| 方法 | Task success | Execution success | Relation goal | Missing step | Additional step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `kg_rag_plan_then_ground` | 76.39% | 83.0% | 76.67% | 12.79% | 2.62% |
| `kg_planning_agent` | **76.72%** | **83.3%** | **77.22%** | **12.46%** | 2.62% |

这个 agent 比 single-pass RAG/KG prompt 略有提升，说明 KG verifier + local repair 是有效的。但提升很小，原因是当前 repair policy 太保守：很多样本因为 parse error、unknown action、arity mismatch 等问题被跳过，只有 25 条被真正修复。

---

## 8. 本周实验给出的判断

本周实验可以总结成三点：

### 1. `plan_then_ground` 仍然是当前最强主方法

它简单、稳定，并且真正降低了最关键的 `missing_step`。所以论文中仍然应该把它作为当前最强的 prompt-based improvement。

### 2. RAG 和 KG 的作用是明确的，但不能只作为 prompt 前缀

`kg_rag_plan_then_ground` 明显提高 relation goal，并降低 hallucination，说明外部知识是有用的。但它没有解决动作步骤缺失，所以仅仅把 RAG/KG 塞进 prompt 不够。

### 3. Agent 方向更符合后续方法发展

`kg_planning_agent` 虽然还没有超过 `plan_then_ground`，但它证明了一个更合理的方向：

> 用 RAG 做场景 grounding，用 KG 做动作约束，用固定流程 agent 做检查和局部修复。

这比单纯 prompt 更像一个完整的 knowledge-grounded planning framework。

---

## 9. 建议论文中如何定位

我建议论文中这样组织方法线索：

### 主结果：`plan_then_ground`

作为当前最稳定、最有效的 prompt-based 方法，强调它能显著降低 missing step。

### 补充消融：复杂 prompt 方法

包括 `state_checklist_plan`、`goal_conditioned_scaffold` 和 `bidirectional_causal_planning`。这些实验说明：

- 目标倒推可以改善 relation grounding；
- checklist 和双向推理不一定提升总成功率；
- prompt 复杂度过高可能伤害 action sequencing。

### 新方向：Knowledge-Grounded Planning Agent

把 `kg_rag_plan_then_ground` 和 `kg_planning_agent` 作为从 prompt-only 转向 knowledge-grounded framework 的探索。当前版本还没有超过主方法，但已经证明：

- RAG/KG 能改善 grounding；
- KG local repair 能小幅降低 missing step；
- 后续需要更强的 structured parser 和 constrained repair agent。

---

## 10. 可以对老师这样说

这周我先继续测试了几种 prompt 方法。结果发现，`plan_then_ground` 仍然是最稳定的，能把成功率从 `75.58%` 提到 `80.23%`。我也尝试了目标倒推和双向因果规划，虽然它们能提升 relation goal，但没有进一步提升整体成功率，说明不是 prompt 越复杂越好。

所以我把方向从 prompt-only 调整到了 knowledge-grounded planning。具体来说，我加入了 Scene Graph RAG 和 Precondition KG，让模型在生成动作前看到相关场景对象和动作前置条件。这个方法改善了 relation goal 和 hallucination，但没有明显降低 missing step。

进一步地，我实现了一个固定流程的 knowledge-grounded planning agent：先生成 draft，再用 KG 检查前置条件，并做保守的局部修复。这个 agent 比单纯 RAG/KG prompt 略有提升，但还没有超过 `plan_then_ground`。目前结论是：RAG 和 KG 是有用的，但下一步需要做更强的 constrained repair agent，而不是只把知识塞进 prompt。

---

## 11. 下一步计划

1. 保留 `plan_then_ground` 作为当前主实验结果。
2. 把复杂 prompt 方法作为消融，说明 prompt-only 的上限和风险。
3. 继续发展 `knowledge-grounded planning agent`，重点解决 parse error、unknown action 和 arity mismatch 不能修的问题。
4. 设计更强的 structured parser，把模型输出先转成可操作的 action graph，再做 KG 约束修复。
5. 在论文中把方法主线从“prompt 提升”扩展为“knowledge-grounded planning framework”。
