# Multi-model Experiment Materials

## Existing Model Inventory

- Total deduplicated result rows: 52
- Result selection prefers `output_norm_all` over `output_single_norm` over `output` when the same model/eval_type appears in multiple roots.
- `gold_oracle` is retained as an upper-bound / pipeline sanity-check reference, not as a model prediction.

### Source Coverage

| Eval type | Source root | Models |
| --- | --- | ---: |
| action_sequencing | `output` | 8 |
| action_sequencing | `output/improvement_run` | 11 |
| action_sequencing | `output_norm_all` | 11 |
| goal_interpretation | `output` | 17 |
| goal_interpretation | `output/improvement_run` | 4 |
| goal_interpretation | `output_single_norm` | 1 |

### Top Action Sequencing Results

| Rank | Model | Family | Source | Task success | Execution success | Missing step | Hallucination |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `glm-5-turbo` | Other | `output/improvement_run` | 88.37 | 93.00 | 2.33 | 2.33 |
| 2 | `glm-5-turbo_sg_rag` | Other | `output/improvement_run` | 87.21 | 89.50 | 5.81 | 2.33 |
| 3 | `minimax-m2-stable` | Other | `output/improvement_run` | 86.05 | 90.70 | 3.49 | 2.33 |
| 4 | `deepseek-v4-flash_plan_then_ground` | Other | `output/improvement_run` | 80.23 | 86.60 | 8.14 | 2.33 |
| 5 | `deepseek-v4-flash_few_shot` | Other | `output/improvement_run` | 79.65 | 84.90 | 10.47 | 2.33 |
| 6 | `deepseek-v4-flash_format_constraints` | Other | `output/improvement_run` | 79.07 | 86.00 | 7.56 | 4.07 |
| 7 | `deepseek-v4-flash_sg_rag` | Other | `output/improvement_run` | 79.07 | 87.20 | 8.72 | 2.33 |
| 8 | `minimax-m2-stable_sg_rag` | Other | `output/improvement_run` | 79.07 | 86.00 | 6.98 | 2.33 |
| 9 | `mistral-large-2402` | Mistral | `output_norm_all` | 76.39 | 83.60 | 12.79 | 2.62 |
| 10 | `deepseek-v4-flash` | Other | `output/improvement_run` | 75.58 | 82.60 | 12.79 | 2.33 |

### Top Goal Interpretation Results

| Rank | Model | Family | All F1 | Node F1 | Edge F1 | Action F1 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `o1-preview-2024-09-12` | OpenAI | 42.75 | 38.46 | 52.25 | 39.46 |
| 2 | `deepseek-v4-flash_schema_constrained` | Other | 40.60 | 50.17 | 35.85 | 30.08 |
| 3 | `deepseek-v4-flash_decompose_then_merge` | Other | 39.00 | 49.05 | 35.23 | 27.55 |
| 4 | `deepseek-v4-flash_few_shot` | Other | 38.97 | 51.05 | 35.56 | 25.70 |
| 5 | `deepseek-v4-flash` | Other | 38.97 | 49.74 | 33.06 | 27.47 |
| 6 | `cohere-command-r` | Cohere | 36.69 | 58.90 | 26.32 | 6.54 |
| 7 | `gpt-4o-2024-05-13` | OpenAI | 36.53 | 39.08 | 36.85 | 33.10 |
| 8 | `gemini-1.5-pro-preview-0409` | Google | 36.22 | 47.32 | 12.35 | 37.19 |
| 9 | `gpt-4-turbo-2024-04-09` | OpenAI | 33.24 | 38.38 | 28.49 | 30.89 |
| 10 | `claude-3-opus-20240229` | Anthropic | 31.47 | 38.50 | 30.68 | 25.13 |

## Related Work Notes

| Topic | Citation | How to use in the write-up |
| --- | --- | --- |
| Embodied Agent Interface | [Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making, NeurIPS Datasets and Benchmarks 2024](https://arxiv.org/abs/2410.07166) | 作为本实验的核心基准来源，说明 EAI 如何统一 goal interpretation、subgoal decomposition、action sequencing、transition modeling，并提供细粒度错误指标。 |
| VirtualHome | [VirtualHome: Simulating Household Activities via Programs](http://virtual-home.org/) | 作为家庭活动动作序列环境背景，用来解释对象、状态、关系、动作可执行性和长程 household task 的难点。 |
| Long-horizon planning | [Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks](https://arxiv.org/html/2503.09572) | 支撑“先规划再执行”和 dynamic replanning 的实验动机，强调复杂任务不能只依赖简单 prompt engineering。 |
| Embodied multimodal benchmarks | [EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents](https://arxiv.org/abs/2502.09560) | 补充具身智能评测背景，尤其是长程规划、低层执行和环境反馈对模型成功率的影响。 |
| Self-refinement for planning | [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | 为生成后自检、修复和 self-consistency 提供方法依据；重点关注动作前置条件、状态更新和错误路径纠正。 |
| Verbal feedback for agents | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | 用于说明失败轨迹可转化为语言反馈，支持下一轮 action sequence 生成或 prompt 修复。 |
| Reasoning-action prompting | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | 可作为 plan-then-ground 或 thought/action 分离实验的背景，但最终输出仍需压回 EAI 的合法动作格式。 |

## New Model Evaluation Matrix

| Family | Candidate model | Priority | Sample scope | Generation parameters | Output location | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | gpt-4.1 or latest available GPT-4-class model | high | full VirtualHome action_sequencing set after 10-sample smoke test | temperature=0, max_tokens=2048, same EAI prompt | `output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json` | 与已有 gpt-4o / gpt-4-turbo 形成同家族纵向对比。 |
| OpenAI | gpt-4o-mini or latest small model | medium | full set; use for ablations if budget is constrained | temperature=0, max_tokens=2048, same EAI prompt | `output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json` | 适合先跑全量或多轮消融，成本低。 |
| Anthropic | Claude 3.5/3.7 Sonnet or latest Sonnet | high | same identifiers as OpenAI run; smoke test first 10 rows | temperature=0, max_tokens=2048, same system format contract | `output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json` | 与已有 Claude 3/3.5 结果比较格式稳定性和 planning order。 |
| Google | Gemini 1.5/2.x Pro or Flash | high | same identifiers as OpenAI run; smoke test first 10 rows | temperature=0, max_output_tokens=2048, same EAI prompt | `output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json` | 重点观察 relation grounding 与 missing_step。 |
| Open-weight | Llama 3.1/3.2 70B or Qwen2.5 72B Instruct | medium | same identifiers; optionally start with representative 100-row subset | temperature=0, max_tokens=2048, OpenAI-compatible chat format | `output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json` | 用于说明开源模型与闭源模型的差距和可复现实验价值。 |
| Mistral | Mistral Large latest | medium | same identifiers as OpenAI run; smoke test first 10 rows | temperature=0, max_tokens=2048, same EAI prompt | `output/api_helm/helm_output/virtualhome/action_sequencing/<model>_outputs.json` | EAI 官方结果中 Mistral Large 在 VirtualHome action sequencing 表现较强。 |

每个新增模型建议使用同一条评测命令链，先规范化，再进入 EAI 评测：

```bash
python analysis/normalize_action_outputs.py \
  --input-dir output/<new_response_root>/helm_output/virtualhome/action_sequencing \
  --output-dir output/<new_response_root>_norm/helm_output/virtualhome/action_sequencing
LLM_RESPONSE_PATH="$PWD/output/<new_response_root>_norm/helm_output" NUM_WORKERS=1 \
  ./scripts/run_action_sequencing_eval.sh virtualhome eai-eval output
```

## Success-rate Improvement Ablation

| Stage | Method | Hypothesis | Primary measures |
| --- | --- | --- | --- |
| `baseline` | 现有 EAI prompt + temperature=0 | 建立与已有结果可比的直接生成基线。 | task_success_rate, execution_success_rate, parsing, missing_step, wrong_order |
| `format_constraints` | 强化 JSON 拼接格式、动作大写、name/id 成对、禁止 Markdown | 主要降低 parsing 和 predicate_argument_number 错误。 | grammar_error.parsing, grammar_error.predicate_argument_number |
| `few_shot_valid_actions` | 加入 1-3 个合法 VirtualHome action sequence 示例 | 改善动作选择和参数数量，减少 hallucination。 | hallucination, task_success_rate |
| `self_check_rewrite` | 生成后执行一次自检，检查动作合法性、对象 id、前置条件和顺序，再输出修正版 | 降低 wrong_order、missing_step 和 additional_step。 | runtime_error.wrong_order, runtime_error.missing_step, runtime_error.additional_step |
| `plan_then_ground` | 先生成 high-level plan，再映射到 VirtualHome name/id action sequence | 改善长程动作顺序和 relation grounding。 | relation_goal, action_goal, execution_success_rate |
| `failure_driven_prompt` | 根据 diagnostics 中 relation_grounding、planning_order、format_or_parsing 的高频失败定向改 prompt | 针对当前错误分布提升总体 task_success_rate。 | failure-type distribution before/after |

## Presentation Figure Plan

| Figure | Main claim | Data source |
| --- | --- | --- |
| `fig_action_task_success.svg` | 动作序列任务中不同模型的可执行成功率差异明显。 | action_sequencing goal_evaluation.task_success_rate |
| `fig_goal_interpretation_f1.svg` | 目标理解能力与动作执行能力不是同一个指标，应分开讨论。 | goal_interpretation all_f1 |
| `fig_action_execution_success.svg` | 执行成功率能揭示格式合法但任务目标仍未完成的情况。 | action_sequencing trajectory_evaluation.execution_success_rate |
| `fig_failure_profile.svg` | 失败主要来自 missing_step、additional_step、hallucination 等细粒度错误，而不只是最终失败。 | action_sequencing grammar_error and runtime_error metrics |
| `fig_family_average.svg` | 模型家族层面的平均表现可辅助解释闭源、开源和推理模型差异。 | family average of rank metrics |

## Suggested Slide Narrative

1. 先说明为什么使用 EAI/VirtualHome：它不仅看最终成功率，还能拆出目标理解、动作序列和细粒度错误类型。
2. 展示已有多模型覆盖：当前结果已经包含 OpenAI、Anthropic、Google、Meta、Mistral、Cohere 等模型家族。
3. 分开讨论 goal interpretation 和 action sequencing：模型可能理解目标，但仍会在对象接地、动作顺序或执行约束上失败。
4. 先展示失败分布，再引出改进方法：missing step、additional step、hallucinated action 和 relation grounding 是后续消融实验的动机。
5. 最后给出下一轮实验矩阵：新增最新模型家族，并控制 prompt、self-check、planning decomposition 等变量。

## Generated Artifacts

- `output/diagnostics/multimodel_experiment_materials.md`
- `output/diagnostics/multimodel_existing_inventory.csv`
- `output/diagnostics/multimodel_family_averages.csv`
- `output/diagnostics/multimodel_failure_profile.csv`
- `output/diagnostics/multimodel_new_model_matrix.csv`
- `output/diagnostics/multimodel_success_improvement_plan.csv`
- `output/diagnostics/multimodel_prompt_templates.md`
- `output/diagnostics/multimodel_ablation_summary.csv`
- `output/diagnostics/multimodel_ablation_summary.md`
- `output/diagnostics/multimodel_goal_vs_action.csv`
- `output/diagnostics/progress_report.md`
- `output/diagnostics/figures/*.svg`
