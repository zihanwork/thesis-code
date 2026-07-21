# Benchmark and literature positioning

Last verified: 2026-07-16.

## What the official EAI benchmark measures

Embodied Agent Interface (EAI) standardizes four modules—goal interpretation,
subgoal decomposition, action sequencing, and transition modeling—over
VirtualHome and BEHAVIOR. It uses LTL for state and temporally extended goals and
reports module-specific symbolic, trajectory, goal-satisfaction, and planner
metrics rather than a single custom final-state score. The CVPR 2026 overview
lists 338 VirtualHome tasks and 100 BEHAVIOR tasks. Sources: [EAI paper],
[official benchmark], [CVPR 2026 overview].

The official challenge requires hidden-test final evaluation, prohibits manual
tuning, hard-coding, and test-set engineering, and requires reproducible,
automated, deterministic submissions. Source: [CVPR 2026 rules]. The NeurIPS
starter-kit submission contains eight output files: four modules for each of the
two environments. Source: [NeurIPS 2025 participation guide].

Therefore this project's current custom score and an official EAI score are
different estimands:

| Result | Data/protocol | Metric | Directly comparable? |
|---|---|---|---|
| Current `197/202` historical result | Custom 202-task executable development subset | Local PDDL final-state task success after symbolic replanning | No |
| Frozen local held-out result | 119 task-ID-unseen but task-family-seen executable VirtualHome instances | Local evaluator task success | No; useful as a seen-family instance-transfer test |
| Official EAI challenge result | Official hidden set, eight required outputs, official evaluator | Four-module, two-environment official metrics and aggregate | Yes, but only after official submission/evaluation |

The thesis must label the first two as **custom subset** results and must never
rank `97.5%` against an official leaderboard average.

## Public challenge evidence

The official NeurIPS 2025 site names AxisTilted2 first, SingaX second, CtrlAct
third, and nju-lamda12/Re² Agent as the most innovative approach. Source:
[NeurIPS 2025 winners].

| Work | Reported evidence | Main method | Relationship to this project |
|---|---|---|---|
| AxisTilted2 | First place. Official report gives BEHAVIOR module scores 99.6 GI, 97.0 SD, 98.0 AS, 99.5 TM; VirtualHome 65.4 GI, 78.7 SD, 82.6 AS, 99.9 TM. | Evaluator-guided data construction, task-specialized Qwen3 fine-tuning, plus retrieval, voting, and a learned evaluator. | Covers RAG and evaluator feedback, but focuses primarily on distillation/fine-tuning and module-specialized systems. |
| SingaX | Second place; A*STAR reports average score 84.32 across 48 teams. | Training-free instruction induction from verifier logs, task-specific prompts, and validation. | Closest overlap with error-log-driven prompt/memory improvement; the thesis must distinguish online task repair from offline prompt induction. |
| CtrlAct | Third place. | Declarative physical constraints in prompts, SFT, and experiments with RL/activation steering/Best-of-N. | Closest overlap with safety/constraint prompting; its SFT comparison makes fine-tuning a relevant but optional future control. |
| Re² Agent | Most innovative; report states 81.36 overall, 86.35 BEHAVIOR, 76.36 VirtualHome. | Execute, reflect on evaluator/environment feedback, extract task rules, and re-execute. | Direct prior art for reflection and re-execution. Plain `H2-LLM` cannot itself be claimed as novel. |

Sources: [AxisTilted2 report], [SingaX report], [SingaX result], [CtrlAct report],
[Re² Agent report].

AxisTilted2 also reports that its larger-model experiments used LoRA on two H100
GPUs, approximately 257 H100 wall-clock hours in total, and about USD 2,000 in
compute/API spend. This supports deferring fine-tuning under the present resource
budget; it does not support claiming that fine-tuning is unimportant.

## Defensible thesis position

The current project should be positioned as a controlled study of **where
planning-time augmentation and execution-time recovery help**, not as the first
system to use RAG, reflection, memory, constraints, or symbolic planning.

The defensible contribution package is:

1. A controlled two-axis experiment that separates prompt/RAG planning from
   local, feedback-based, memory-based, and symbolic recovery.
2. A mechanism-overlap audit: symbolic PDDL planning and PDDL fallback are
   reported separately instead of being counted as two independent gains.
3. Task-paired statistical analysis, cost/search accounting, frozen data
   boundaries, and failure-type analysis across model strengths and environments.
4. A direct comparison of plain reflection, error-specific repair, frozen-memory
   repair, and leave-one-component-out combinations under the same planner and
   task set.

The strongest research question is not “Can a harness reach 97.5%?” It is:

> Under matched tasks and models, how much does retrieval improve initial plans,
> how many residual RAG failures does feedback recovery repair, and at what
> token, latency, and symbolic-search cost?

The frozen final design does not contain the P0/Reflection cell required to
estimate a planning-by-recovery interaction. It must not use "complementary" as
a factorial or causal interaction claim.

## Literature structure for the thesis

Use five related-work subsections:

1. **Embodied decision-making benchmarks:** EAI, BEHAVIOR, VirtualHome, and the
   distinction between state and temporally extended goals.
2. **LLMs as embodied planners:** prompt-based action sequencing and structured
   symbolic interfaces.
3. **Retrieval and planning augmentation:** example retrieval, structured
   state/goal/action-schema retrieval, and hybrid LLM-symbolic planning.
4. **Feedback, reflection, and memory:** EAI replanning, Re², verifier-log prompt
   induction, and failure-memory repair.
5. **Constrained and trained systems:** constraint prompting, evaluator-guided
   distillation, SFT/LoRA, voting, and learned evaluators.

The original EAI paper explicitly includes replanning/feedback analysis and
prompt-strategy comparisons. Cite that fact when describing `H2-LLM`; the
project's value must come from controlled isolation and analysis rather than
from presenting feedback replanning as new.

## Immediate experimental consequences

- Keep fine-tuning outside the critical path, but discuss AxisTilted2 and CtrlAct
  as evidence that training can be a strong comparison when resources permit.
- Treat prompt engineering as a required baseline, not a cosmetic addition.
- Compare BEHAVIOR and VirtualHome separately; both the EAI paper and public
  reports show materially different failure patterns and distributions.
- Do not promote development scores to final claims.
- Generate official eight-file outputs and run the official evaluator before
  making leaderboard comparisons.
- Cite unsuccessful methods and resource costs, as requested by the official
  technical-report guidance.

[EAI paper]: https://arxiv.org/abs/2410.07166
[official benchmark]: https://embodied-agent-interface.github.io/
[CVPR 2026 overview]: https://eai-challenge-cvpr2026.github.io/
[CVPR 2026 rules]: https://eai-challenge-cvpr2026.github.io/rules
[NeurIPS 2025 participation guide]: https://neurips25-eai.github.io/participate
[NeurIPS 2025 winners]: https://neurips25-eai.github.io/
[AxisTilted2 report]: https://openreview.net/attachment?id=gABfrJI5ni&name=pdf
[SingaX report]: https://openreview.net/forum?id=WbwGPPxk88
[SingaX result]: https://www.a-star.edu.sg/cfar/news/features/news/features/singax-team-wins-second-place-in-eai-challenge
[CtrlAct report]: https://openreview.net/forum?id=0dt9Ho6dXA
[Re² Agent report]: https://openreview.net/pdf/7816fcb9b33d7b6f1d534d35c90209e7b68dd1da.pdf
