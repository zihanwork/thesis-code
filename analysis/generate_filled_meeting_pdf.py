#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_top_goal_rows(rows: List[Dict], n: int = 5) -> List[Dict]:
    goal_rows = [r for r in rows if r.get("eval_type") == "goal_interpretation"]
    goal_rows.sort(key=lambda x: float(x.get("all_f1", 0.0) or 0.0), reverse=True)
    return goal_rows[:n]


def pick_action_rows(rows: List[Dict], n: int = 3) -> List[Dict]:
    action_rows = [r for r in rows if r.get("eval_type") == "action_sequencing"]
    action_rows.sort(
        key=lambda x: float(x.get("goal_evaluation.task_success_rate", 0.0) or 0.0),
        reverse=True,
    )
    return action_rows[:n]


def create_markdown(
    summary_rows: List[Dict], top_cases: List[Dict], failure_counts: Dict[str, int]
) -> str:
    top_goal = pick_top_goal_rows(summary_rows, n=5)
    top_action = pick_action_rows(summary_rows, n=3)
    total_models = len([r for r in summary_rows if r.get("eval_type") == "action_sequencing"])
    zero_task_success = sum(
        1
        for r in summary_rows
        if r.get("eval_type") == "action_sequencing"
        and float(r.get("goal_evaluation.task_success_rate", 0.0) or 0.0) == 0.0
    )

    lines: List[str] = []
    lines.append("# From Goal Understanding to Action Failure")
    lines.append("## Meeting Update Report (Filled Version)")
    lines.append("")
    lines.append("### 1. Basic Information")
    lines.append("- Tentative Thesis Title: From Goal Understanding to Action Failure: A Diagnostic Study of LLMs for Embodied Decision Making")
    lines.append("- Meeting Type: Hybrid meeting")
    lines.append("- Date: 4.20")
    lines.append("- Presenter: wu zihan")
    lines.append("- Current Stage: Baseline reproduction completed, downstream diagnosis started")
    lines.append("")
    lines.append("### 2. This Week's Core Update")
    lines.append("#### 2.1 One-sentence update")
    lines.append("I have reproduced the EAI goal interpretation baseline, completed one downstream action sequencing run, and started diagnosing why goal-level understanding does not transfer to executable actions.")
    lines.append("")
    lines.append("#### 2.2 What I focused on")
    lines.append("- Read EAI framework and re-check module interfaces")
    lines.append("- Run `goal_interpretation` and organize multi-model baseline outputs")
    lines.append("- Run `action_sequencing` and collect downstream logs")
    lines.append("- Build a tiny diagnostic pipeline linking goal-level and action-level outcomes")
    lines.append("")
    lines.append("#### 2.3 What has been completed")
    lines.append("- Environment setup and CLI reproduction completed (`conda + Python 3.8 + eai-eval`)")
    lines.append("- Goal interpretation evaluation completed and ranked")
    lines.append("- Action sequencing evaluation completed for VirtualHome with official HELM responses")
    lines.append("- Diagnostic scripts completed: run / summarize / link-and-diagnose")
    lines.append("")
    lines.append("#### 2.4 What is still in progress")
    lines.append("- Distinguish formatting/parsing failures from true planning-reasoning failures")
    lines.append("- Add a cleaner per-model case alignment to avoid unknown-model cases")
    lines.append("- Expand diagnosis beyond one downstream module")
    lines.append("")
    lines.append("### 3. Research Direction")
    lines.append("#### 3.1 Main research question")
    lines.append("When LLMs appear to understand the goal, why do they still fail to generate executable or successful action plans?")
    lines.append("")
    lines.append("#### 3.2 Motivation")
    lines.append("- Overall success rate alone is insufficient for failure diagnosis")
    lines.append("- Embodied pipelines contain multiple failure points after goal understanding")
    lines.append("- Need module-level and case-level diagnosis")
    lines.append("")
    lines.append("#### 3.3 Current scope")
    lines.append("- Benchmark/framework: EAI")
    lines.append("- Simulator: VirtualHome")
    lines.append("- Main module: Goal Interpretation")
    lines.append("- Downstream module: Action Sequencing")
    lines.append("- Diagnostic focus: relation grounding, implicit preconditions, planning/executability")
    lines.append("")
    lines.append("#### 3.4 Expected contribution")
    lines.append("Provide a diagnostic bridge from goal understanding to downstream action failure, rather than only reporting benchmark-level success rates.")
    lines.append("")
    lines.append("### 4. Benchmark / Codebase Progress")
    lines.append("#### 4.1 Environment and code setup")
    lines.append("- Repository used: https://github.com/embodied-agent-interface/embodied-agent-interface?tab=readme-ov-file")
    lines.append("- Main scripts: `scripts/run_action_sequencing_eval.sh`, `analysis/summarize_downstream.py`, `analysis/link_goal_to_action_failures.py`")
    lines.append("- Output structure: `output/virtualhome/evaluate_results/{goal_interpretation,action_sequencing}/<model>/summary.json`")
    lines.append("- Diagnostics output: `output/diagnostics/`")
    lines.append("")
    lines.append("#### 4.2 Modules located")
    lines.append("- Goal Interpretation: completed")
    lines.append("- Action Sequencing: completed")
    lines.append("- Subgoal Decomposition: located, not yet diagnosed")
    lines.append("- Transition Modeling: located, not yet diagnosed")
    lines.append("")
    lines.append("#### 4.3 Current run status")
    lines.append("- Goal Interpretation / VirtualHome / 18 models / completed")
    lines.append("- Action Sequencing / VirtualHome / 18 models / completed (with severe formatting/parsing issues)")
    lines.append("")
    lines.append("### 5. Baseline Results")
    lines.append("#### 5.1 Goal Interpretation baseline summary (Top-5 by all_f1)")
    for idx, row in enumerate(top_goal, start=1):
        lines.append(
            f"- Rank {idx}: {row.get('model')} | all_f1={float(row.get('all_f1', 0.0)):.4f}, "
            f"all_precision={float(row.get('all_precision', 0.0)):.4f}, all_recall={float(row.get('all_recall', 0.0)):.4f}"
        )
    lines.append("")
    lines.append("#### 5.2 Key observations from goal interpretation")
    lines.append("- Many models show recall > precision, implying over-generation")
    lines.append("- Node/edge/action metrics are imbalanced across models")
    lines.append("- A few models have abnormal zero outputs and need input-format checks")
    lines.append("")
    lines.append("#### 5.3 Downstream module baseline summary (Action Sequencing)")
    lines.append(
        f"- Models evaluated: {total_models}; models with task_success_rate = 0: {zero_task_success}"
    )
    for row in top_action:
        lines.append(
            f"- {row.get('model')}: task_success_rate={float(row.get('goal_evaluation.task_success_rate', 0.0)):.1f}, "
            f"execution_success_rate={float(row.get('trajectory_evaluation.execution_success_rate', 0.0)):.1f}, "
            f"grammar.parsing={float(row.get('trajectory_evaluation.grammar_error.parsing', 0.0)):.1f}"
        )
    lines.append("")
    lines.append("#### 5.4 Key observations from downstream module")
    lines.append("- Current downstream failures are dominated by format/parsing issues")
    lines.append("- Many logs indicate: `Action WALK does not follow name_id format`")
    lines.append("- This means immediate failures may happen before deep planning logic is evaluated")
    lines.append("")
    lines.append("### 6. Very Small Diagnostic Slicing")
    lines.append("#### 6.1 Diagnostic question")
    lines.append("Which cases look correct at the goal level, but still fail at the action level?")
    lines.append("")
    lines.append("#### 6.2 Case summary (Top-3)")
    for idx, case in enumerate(top_cases[:3], start=1):
        lines.append(f"- Case {idx}:")
        lines.append(f"  - Task: {case.get('task')}")
        lines.append(f"  - file_id: {case.get('file_id')}")
        lines.append(f"  - Goal indicator (goal_f1): {case.get('goal_f1')}")
        lines.append("  - Goal interpretation judgment: relatively acceptable at case-level metric")
        lines.append("  - Downstream action outcome: no executable prediction")
        lines.append(f"  - Failure type: {case.get('failure_type')}")
        evidence = str(case.get("raw_failure_text", ""))[:180]
        lines.append(f"  - Diagnostic note: {evidence}")
    lines.append("")
    lines.append("#### 6.3 Emerging failure patterns")
    for k, v in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### 7. Current Hypothesis")
    lines.append("#### 7.1 Working hypothesis")
    lines.append("LLM failures in embodied decision making may come not only from goal misunderstanding but also from relation grounding, implicit preconditions, and executable action formatting/planning.")
    lines.append("")
    lines.append("#### 7.2 Evidence I currently have")
    lines.append("- EAI emphasizes module-level diagnosis rather than overall success only")
    lines.append("- Current runs show a clear gap between goal-level scores and action-level executability")
    lines.append("- Parsing-format failures are frequent in action_sequencing logs")
    lines.append("")
    lines.append("#### 7.3 Evidence I still need")
    lines.append("- Controlled experiments separating format errors from planning errors")
    lines.append("- More cases showing high goal quality but downstream runtime failure categories")
    lines.append("- Extension to another module (subgoal decomposition or transition modeling)")
    lines.append("")
    lines.append("### 8. Questions to Discuss in the Meeting")
    lines.append("1. Should the thesis scope stay focused on `goal_interpretation + action_sequencing` for depth first?")
    lines.append("2. For current all-zero downstream results, should we first normalize output action format before deeper diagnosis?")
    lines.append("3. Is VirtualHome enough for phase-1, with BEHAVIOR moved to phase-2 validation?")
    lines.append("4. Is the best framing a diagnostic thesis rather than full benchmark reproduction?")
    lines.append("")
    return "\n".join(lines) + "\n"


def create_markdown_zh(
    summary_rows: List[Dict], top_cases: List[Dict], failure_counts: Dict[str, int]
) -> str:
    top_goal = pick_top_goal_rows(summary_rows, n=5)
    top_action = pick_action_rows(summary_rows, n=3)
    total_models = len([r for r in summary_rows if r.get("eval_type") == "action_sequencing"])
    zero_task_success = sum(
        1
        for r in summary_rows
        if r.get("eval_type") == "action_sequencing"
        and float(r.get("goal_evaluation.task_success_rate", 0.0) or 0.0) == 0.0
    )

    lines: List[str] = []
    lines.append("# 从目标理解到动作失败")
    lines.append("## 组会更新报告（补全版）")
    lines.append("")
    lines.append("### 1. 基本信息")
    lines.append("- 暂定论文题目：从目标理解到动作失败：面向具身决策的大语言模型诊断研究")
    lines.append("- 会议形式：线上线下混合")
    lines.append("- 日期：4.20")
    lines.append("- 汇报人：wu zihan")
    lines.append("- 当前阶段：已完成基线复现，正在开展下游诊断")
    lines.append("")
    lines.append("### 2. 本周核心更新")
    lines.append("#### 2.1 一句话更新")
    lines.append("本周已完成 EAI 中 goal_interpretation 基线复现，并完成 action_sequencing 下游运行，开始诊断“目标看似正确但动作不可执行”的原因。")
    lines.append("")
    lines.append("#### 2.2 本周重点工作")
    lines.append("- 回顾 EAI 框架并复核模块接口")
    lines.append("- 跑通 `goal_interpretation` 并整理多模型结果")
    lines.append("- 跑通 `action_sequencing` 并收集日志")
    lines.append("- 构建 goal 到 action 的小规模诊断链路")
    lines.append("")
    lines.append("#### 2.3 已完成内容")
    lines.append("- 环境与 CLI 复现完成（`conda + Python 3.8 + eai-eval`）")
    lines.append("- Goal Interpretation 评测完成并完成排序")
    lines.append("- Action Sequencing（VirtualHome）评测完成")
    lines.append("- 诊断脚本已完成：运行、汇总、跨模块对齐")
    lines.append("")
    lines.append("#### 2.4 正在进行内容")
    lines.append("- 区分“格式/解析失败”与“真实规划推理失败”")
    lines.append("- 优化 case 与模型的一一映射，减少 unknown-model 记录")
    lines.append("- 扩展到更多下游模块进行对比")
    lines.append("")
    lines.append("### 3. 研究方向")
    lines.append("#### 3.1 研究问题")
    lines.append("当 LLM 在目标层面看起来理解正确时，为什么仍会在动作层面产生不可执行或失败的计划？")
    lines.append("")
    lines.append("#### 3.2 研究动机")
    lines.append("- 仅看总体成功率无法定位错误发生在哪一层")
    lines.append("- 具身决策是流水线问题，目标正确不代表执行成功")
    lines.append("- 需要模块级 + 案例级诊断")
    lines.append("")
    lines.append("#### 3.3 当前范围")
    lines.append("- 基准框架：EAI")
    lines.append("- 仿真器：VirtualHome")
    lines.append("- 主模块：Goal Interpretation")
    lines.append("- 下游模块：Action Sequencing")
    lines.append("- 诊断重点：关系落地、隐式前提、可执行性与规划")
    lines.append("")
    lines.append("#### 3.4 预期贡献")
    lines.append("在基准复现之外，提供从“目标理解”到“动作失败”的可诊断分析链路。")
    lines.append("")
    lines.append("### 4. Benchmark / Codebase 进展")
    lines.append("#### 4.1 环境与代码进展")
    lines.append("- 参考仓库：https://github.com/embodied-agent-interface/embodied-agent-interface?tab=readme-ov-file")
    lines.append("- 关键脚本：`scripts/run_action_sequencing_eval.sh`、`analysis/summarize_downstream.py`、`analysis/link_goal_to_action_failures.py`")
    lines.append("- 输出结构：`output/virtualhome/evaluate_results/{goal_interpretation,action_sequencing}/<model>/summary.json`")
    lines.append("- 诊断产物：`output/diagnostics/`")
    lines.append("")
    lines.append("#### 4.2 已定位模块")
    lines.append("- Goal Interpretation：已完成")
    lines.append("- Action Sequencing：已完成")
    lines.append("- Subgoal Decomposition：已定位，待诊断")
    lines.append("- Transition Modeling：已定位，待诊断")
    lines.append("")
    lines.append("#### 4.3 当前运行状态")
    lines.append("- Goal Interpretation / VirtualHome / 18 模型 / 已完成")
    lines.append("- Action Sequencing / VirtualHome / 18 模型 / 已完成（但存在明显格式与解析问题）")
    lines.append("")
    lines.append("### 5. 基线结果")
    lines.append("#### 5.1 Goal Interpretation 基线摘要（按 all_f1 前5）")
    for idx, row in enumerate(top_goal, start=1):
        lines.append(
            f"- 第{idx}名：{row.get('model')} | all_f1={float(row.get('all_f1', 0.0)):.4f}，"
            f"all_precision={float(row.get('all_precision', 0.0)):.4f}，all_recall={float(row.get('all_recall', 0.0)):.4f}"
        )
    lines.append("")
    lines.append("#### 5.2 Goal Interpretation 关键观察")
    lines.append("- 多数模型出现 recall 高于 precision，表现为“多生成”倾向")
    lines.append("- node/edge/action 三类指标不均衡，说明能力分布差异明显")
    lines.append("- 存在异常全 0 结果，需继续排查输入输出配置")
    lines.append("")
    lines.append("#### 5.3 下游模块基线摘要（Action Sequencing）")
    lines.append(f"- 评测模型数：{total_models}；task_success_rate 为 0 的模型数：{zero_task_success}")
    for row in top_action:
        lines.append(
            f"- {row.get('model')}：task_success_rate={float(row.get('goal_evaluation.task_success_rate', 0.0)):.1f}，"
            f"execution_success_rate={float(row.get('trajectory_evaluation.execution_success_rate', 0.0)):.1f}，"
            f"grammar.parsing={float(row.get('trajectory_evaluation.grammar_error.parsing', 0.0)):.1f}"
        )
    lines.append("")
    lines.append("#### 5.4 下游模块关键观察")
    lines.append("- 当前失败主要集中在格式与解析层")
    lines.append("- 日志高频出现：`Action WALK does not follow name_id format`")
    lines.append("- 说明很多样本在规划深层评估前已因格式问题失败")
    lines.append("")
    lines.append("### 6. 小规模诊断切片")
    lines.append("#### 6.1 诊断问题")
    lines.append("哪些样本在 goal 层面看起来不错，但在 action 层面仍失败？")
    lines.append("")
    lines.append("#### 6.2 案例摘要（Top-3）")
    for idx, case in enumerate(top_cases[:3], start=1):
        lines.append(f"- Case {idx}:")
        lines.append(f"  - Task: {case.get('task')}")
        lines.append(f"  - file_id: {case.get('file_id')}")
        lines.append(f"  - Goal 指标（goal_f1）: {case.get('goal_f1')}")
        lines.append("  - Goal 判断：在 case-level 指标上相对可接受")
        lines.append("  - Action 结果：无可执行预测")
        lines.append(f"  - 失败类型：{case.get('failure_type')}")
        evidence = str(case.get("raw_failure_text", ""))[:180]
        lines.append(f"  - 诊断备注：{evidence}")
    lines.append("")
    lines.append("#### 6.3 初步失败模式")
    for k, v in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### 7. 当前假设")
    lines.append("#### 7.1 工作假设")
    lines.append("具身决策失败不仅来自目标理解错误，也来自关系落地、隐式前提以及动作格式/可执行性问题。")
    lines.append("")
    lines.append("#### 7.2 目前已有证据")
    lines.append("- EAI 框架强调模块化诊断而非只看总成功率")
    lines.append("- 当前结果显示 goal 分数与 action 可执行性存在明显断层")
    lines.append("- action 日志显示大量 parsing/format 失败")
    lines.append("")
    lines.append("#### 7.3 仍需补充证据")
    lines.append("- 设计实验区分“格式失败”与“规划推理失败”")
    lines.append("- 增加高 goal 质量但 runtime 失败的案例")
    lines.append("- 扩展到 subgoal decomposition 或 transition modeling 进行对照")
    lines.append("")
    lines.append("### 8. 组会讨论问题")
    lines.append("1. 论文阶段一是否先聚焦 `goal_interpretation + action_sequencing`？")
    lines.append("2. 当前下游全 0 结果下，是否应先做动作格式标准化，再进入深层诊断？")
    lines.append("3. 第一阶段是否只用 VirtualHome，后续再纳入 BEHAVIOR？")
    lines.append("4. 论文定位是否以“诊断分析”优先，而非“全量基准复现”？")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_pdf(text: str, output_pdf: Path) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName="STSong-Light",
        fontSize=16,
        leading=20,
    )
    body_style = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=16,
    )
    h_style = ParagraphStyle(
        "HeadingCN",
        parent=styles["Heading2"],
        fontName="STSong-Light",
        fontSize=12,
        leading=18,
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    elements = []

    lines = text.splitlines()
    for line in lines:
        clean = line.strip()
        if not clean:
            elements.append(Spacer(1, 5))
            continue

        if clean.startswith("# "):
            elements.append(Paragraph(clean[2:], title_style))
            elements.append(Spacer(1, 6))
        elif clean.startswith("### "):
            elements.append(Paragraph(clean[4:], h_style))
            elements.append(Spacer(1, 3))
        elif clean.startswith("#### "):
            elements.append(Paragraph(clean[5:], body_style))
            elements.append(Spacer(1, 2))
        elif clean.startswith("- "):
            elements.append(Paragraph(f"• {clean[2:]}", body_style))
        else:
            elements.append(Paragraph(clean, body_style))

    # Add one compact table required by report acceptance.
    table_data = [
        ["Module", "Dataset", "Models", "Status"],
        ["Goal Interpretation", "VirtualHome", "18", "Completed"],
        ["Action Sequencing", "VirtualHome", "18", "Completed"],
    ]
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Run Status Snapshot", h_style))
    t = Table(table_data, colWidths=[48 * mm, 34 * mm, 20 * mm, 34 * mm])
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (2, 1), (2, -1), "CENTER"),
            ]
        )
    )
    elements.append(t)
    doc.build(elements)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filled meeting report PDF.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--top-cases-json", required=True)
    parser.add_argument("--failure-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--language", choices=["en", "zh"], default="zh")
    args = parser.parse_args()

    summary_rows = load_json(Path(args.summary_json))
    top_cases = load_json(Path(args.top_cases_json))
    failure_counts = load_json(Path(args.failure_json))

    if args.language == "zh":
        md_text = create_markdown_zh(summary_rows, top_cases, failure_counts)
    else:
        md_text = create_markdown(summary_rows, top_cases, failure_counts)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")

    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(md_text, out_pdf)

    print(f"[DONE] markdown: {out_md}")
    print(f"[DONE] pdf: {out_pdf}")


if __name__ == "__main__":
    main()
