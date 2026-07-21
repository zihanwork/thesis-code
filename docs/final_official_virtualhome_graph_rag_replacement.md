# Official VirtualHome Action Sequencing Results

## Scope

The single outcome authority is the pinned official evaluator. The fixed cohort contains 84 tasks across 8 task families. Local execution checks are used only inside the verifier/recovery mechanism.

## Official results

| Cell | Task success | Total goal | Execution success | Parsing error |
|---|---:|---:|---:|---:|
| `deepseek_v4_flash__P1_rag__H0_open_loop` | 47.619% | 67.797% | 89.300% | 0.000% |
| `deepseek_v4_flash__P1_rag__H2_llm_reflection` | 50.000% | 72.881% | 100.000% | 0.000% |
| `deepseek_v4_flash__P1_rag__H2_memory` | 50.000% | 71.751% | 100.000% | 0.000% |
| `deepseek_v4_flash__P1_rag__H2_pddl_recovery` | 50.000% | 71.186% | 100.000% | 0.000% |
| `deepseek_v4_flash__P2_graph_rag__H0_open_loop` | 52.381% | 71.186% | 95.200% | 3.571% |
| `deepseek_v4_flash__P2_graph_rag__H2_llm_reflection` | 52.381% | 71.186% | 96.400% | 0.000% |
| `deepseek_v4_flash__P2_graph_rag__H2_memory` | 52.381% | 72.316% | 98.800% | 0.000% |
| `deepseek_v4_flash__P2_graph_rag__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `glm_5_turbo__P1_rag__H0_open_loop` | 52.381% | 71.186% | 92.900% | 0.000% |
| `glm_5_turbo__P1_rag__H2_llm_reflection` | 52.381% | 74.011% | 98.800% | 0.000% |
| `glm_5_turbo__P1_rag__H2_memory` | 52.381% | 74.011% | 98.800% | 0.000% |
| `glm_5_turbo__P1_rag__H2_pddl_recovery` | 52.381% | 74.011% | 98.800% | 0.000% |
| `glm_5_turbo__P2_graph_rag__H0_open_loop` | 50.000% | 67.797% | 91.700% | 8.333% |
| `glm_5_turbo__P2_graph_rag__H2_llm_reflection` | 51.191% | 70.056% | 92.900% | 4.762% |
| `glm_5_turbo__P2_graph_rag__H2_memory` | 53.571% | 72.881% | 98.800% | 0.000% |
| `glm_5_turbo__P2_graph_rag__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `gpt_5_5__P1_rag__H0_open_loop` | 48.809% | 71.186% | 96.400% | 1.190% |
| `gpt_5_5__P1_rag__H2_llm_reflection` | 50.000% | 72.316% | 98.800% | 1.190% |
| `gpt_5_5__P1_rag__H2_memory` | 50.000% | 72.316% | 98.800% | 1.190% |
| `gpt_5_5__P1_rag__H2_pddl_recovery` | 50.000% | 72.316% | 98.800% | 1.190% |
| `gpt_5_5__P2_graph_rag__H0_open_loop` | 54.762% | 76.836% | 98.800% | 1.190% |
| `gpt_5_5__P2_graph_rag__H2_llm_reflection` | 54.762% | 76.836% | 98.800% | 1.190% |
| `gpt_5_5__P2_graph_rag__H2_memory` | 54.762% | 76.836% | 98.800% | 1.190% |
| `gpt_5_5__P2_graph_rag__H2_pddl_recovery` | 54.762% | 76.836% | 98.800% | 1.190% |

## Family-clustered paired comparisons

| Contrast | Uplift | Family-clustered 95% CI | McNemar p |
|---|---:|---:|---:|
| `deepseek_v4_flash__H0_open_loop__P1_rag_to_P2_graph_rag` | 4.76 pp | [0.00, 9.30] pp | 0.1250 |
| `deepseek_v4_flash__H2_llm_reflection__P1_rag_to_P2_graph_rag` | 2.38 pp | [0.00, 5.88] pp | 0.5000 |
| `deepseek_v4_flash__H2_memory__P1_rag_to_P2_graph_rag` | 2.38 pp | [0.00, 5.88] pp | 0.5000 |
| `deepseek_v4_flash__H2_pddl_recovery__P1_rag_to_P2_graph_rag` | 2.38 pp | [0.00, 5.88] pp | 0.5000 |
| `deepseek_v4_flash__P1_rag__H0_to_H2_llm_reflection` | 2.38 pp | [0.00, 7.60] pp | 0.5000 |
| `deepseek_v4_flash__P1_rag__H0_to_H2_memory` | 2.38 pp | [0.00, 7.60] pp | 0.5000 |
| `deepseek_v4_flash__P1_rag__H0_to_H2_pddl_recovery` | 2.38 pp | [0.00, 7.60] pp | 0.5000 |
| `deepseek_v4_flash__P2_graph_rag__H0_to_H2_llm_reflection` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__P2_graph_rag__H0_to_H2_memory` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__P2_graph_rag__H0_to_H2_pddl_recovery` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `gpt_5_5__H0_open_loop__P1_rag_to_P2_graph_rag` | 5.95 pp | [1.02, 14.49] pp | 0.0625 |
| `gpt_5_5__H2_llm_reflection__P1_rag_to_P2_graph_rag` | 4.76 pp | [0.00, 13.16] pp | 0.1250 |
| `gpt_5_5__H2_memory__P1_rag_to_P2_graph_rag` | 4.76 pp | [0.00, 13.16] pp | 0.1250 |
| `gpt_5_5__H2_pddl_recovery__P1_rag_to_P2_graph_rag` | 4.76 pp | [0.00, 13.16] pp | 0.1250 |
| `gpt_5_5__P1_rag__H0_to_H2_llm_reflection` | 1.19 pp | [0.00, 4.35] pp | 1.0000 |
| `gpt_5_5__P1_rag__H0_to_H2_memory` | 1.19 pp | [0.00, 4.35] pp | 1.0000 |
| `gpt_5_5__P1_rag__H0_to_H2_pddl_recovery` | 1.19 pp | [0.00, 4.35] pp | 1.0000 |
| `gpt_5_5__P2_graph_rag__H0_to_H2_llm_reflection` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `gpt_5_5__P2_graph_rag__H0_to_H2_memory` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `gpt_5_5__P2_graph_rag__H0_to_H2_pddl_recovery` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__H0_open_loop__P1_rag_to_P2_graph_rag` | -2.38 pp | [-4.65, 0.00] pp | 0.5000 |
| `glm_5_turbo__H2_llm_reflection__P1_rag_to_P2_graph_rag` | -1.19 pp | [-4.39, 3.03] pp | 1.0000 |
| `glm_5_turbo__H2_memory__P1_rag_to_P2_graph_rag` | 1.19 pp | [0.00, 3.90] pp | 1.0000 |
| `glm_5_turbo__H2_pddl_recovery__P1_rag_to_P2_graph_rag` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__P1_rag__H0_to_H2_llm_reflection` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__P1_rag__H0_to_H2_memory` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__P1_rag__H0_to_H2_pddl_recovery` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__P2_graph_rag__H0_to_H2_llm_reflection` | 1.19 pp | [0.00, 4.23] pp | 1.0000 |
| `glm_5_turbo__P2_graph_rag__H0_to_H2_memory` | 3.57 pp | [1.19, 6.02] pp | 0.2500 |
| `glm_5_turbo__P2_graph_rag__H0_to_H2_pddl_recovery` | 2.38 pp | [0.00, 4.65] pp | 0.5000 |

## Claim boundaries

- All reported outcome scores come from the pinned official evaluator.
- The local verifier is an intervention component, not an outcome authority.
- The 84-task cohort is not the complete official hidden challenge set.
- This report is a post-hoc replacement replication on the observed cohort; it supports same-cohort P1-to-P2 comparisons, not untouched confirmatory claims.
- The reported grid contains 2 planners and 4 harnesses.
