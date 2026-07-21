# Official VirtualHome Action Sequencing Results

## Scope

The single outcome authority is the pinned official evaluator. The fixed cohort contains 84 tasks across 8 task families. Local execution checks are used only inside the verifier/recovery mechanism.

## Official results

| Cell | Task success | Total goal | Execution success | Parsing error |
|---|---:|---:|---:|---:|
| `deepseek_v4_flash__B0_minimal_prompt__H0_open_loop` | 0.000% | 0.000% | 0.000% | 100.000% |
| `deepseek_v4_flash__B0_minimal_prompt__H2_llm_reflection` | 39.286% | 47.458% | 48.800% | 22.619% |
| `deepseek_v4_flash__B0_minimal_prompt__H2_memory` | 50.000% | 64.972% | 78.600% | 8.333% |
| `deepseek_v4_flash__B0_minimal_prompt__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `deepseek_v4_flash__P0_engineered_prompt__H0_open_loop` | 13.095% | 31.638% | 27.400% | 8.333% |
| `deepseek_v4_flash__P0_engineered_prompt__H2_llm_reflection` | 44.048% | 69.492% | 91.700% | 1.190% |
| `deepseek_v4_flash__P0_engineered_prompt__H2_memory` | 47.619% | 69.492% | 97.600% | 1.190% |
| `deepseek_v4_flash__P0_engineered_prompt__H2_pddl_recovery` | 51.191% | 71.751% | 100.000% | 0.000% |
| `deepseek_v4_flash__P0_structured_prompt__H0_open_loop` | 8.333% | 30.509% | 25.000% | 5.952% |
| `deepseek_v4_flash__P0_structured_prompt__H2_llm_reflection` | 47.619% | 72.316% | 95.200% | 1.190% |
| `deepseek_v4_flash__P0_structured_prompt__H2_memory` | 48.809% | 71.751% | 96.400% | 2.381% |
| `deepseek_v4_flash__P0_structured_prompt__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `deepseek_v4_flash__P1_rag__H0_open_loop` | 52.381% | 72.316% | 97.600% | 0.000% |
| `deepseek_v4_flash__P1_rag__H2_llm_reflection` | 52.381% | 73.446% | 100.000% | 0.000% |
| `deepseek_v4_flash__P1_rag__H2_memory` | 52.381% | 73.446% | 100.000% | 0.000% |
| `deepseek_v4_flash__P1_rag__H2_pddl_recovery` | 52.381% | 73.446% | 100.000% | 0.000% |
| `deepseek_v4_flash__P2_graph_rag__H0_open_loop` | 46.429% | 68.927% | 91.700% | 1.190% |
| `deepseek_v4_flash__P2_graph_rag__H2_llm_reflection` | 51.191% | 72.881% | 98.800% | 1.190% |
| `deepseek_v4_flash__P2_graph_rag__H2_memory` | 51.191% | 72.881% | 98.800% | 1.190% |
| `deepseek_v4_flash__P2_graph_rag__H2_pddl_recovery` | 52.381% | 73.446% | 100.000% | 0.000% |
| `glm_5_turbo__B0_minimal_prompt__H0_open_loop` | 0.000% | 0.000% | 0.000% | 100.000% |
| `glm_5_turbo__B0_minimal_prompt__H2_llm_reflection` | 67.857% | 80.791% | 76.200% | 3.571% |
| `glm_5_turbo__B0_minimal_prompt__H2_memory` | 70.238% | 84.181% | 95.200% | 0.000% |
| `glm_5_turbo__B0_minimal_prompt__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `glm_5_turbo__P0_engineered_prompt__H0_open_loop` | 11.905% | 31.638% | 26.200% | 13.095% |
| `glm_5_turbo__P0_engineered_prompt__H2_llm_reflection` | 48.809% | 68.927% | 89.300% | 7.143% |
| `glm_5_turbo__P0_engineered_prompt__H2_memory` | 52.381% | 72.881% | 96.400% | 3.571% |
| `glm_5_turbo__P0_engineered_prompt__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `glm_5_turbo__P0_structured_prompt__H0_open_loop` | 17.857% | 33.333% | 29.800% | 10.714% |
| `glm_5_turbo__P0_structured_prompt__H2_llm_reflection` | 53.571% | 72.881% | 91.700% | 3.571% |
| `glm_5_turbo__P0_structured_prompt__H2_memory` | 52.381% | 73.446% | 97.600% | 2.381% |
| `glm_5_turbo__P0_structured_prompt__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `glm_5_turbo__P1_rag__H0_open_loop` | 47.619% | 65.537% | 83.300% | 3.571% |
| `glm_5_turbo__P1_rag__H2_llm_reflection` | 50.000% | 70.621% | 95.200% | 2.381% |
| `glm_5_turbo__P1_rag__H2_memory` | 51.191% | 74.011% | 98.800% | 0.000% |
| `glm_5_turbo__P1_rag__H2_pddl_recovery` | 52.381% | 74.576% | 100.000% | 0.000% |
| `glm_5_turbo__P2_graph_rag__H0_open_loop` | 44.048% | 59.322% | 76.200% | 16.667% |
| `glm_5_turbo__P2_graph_rag__H2_llm_reflection` | 47.619% | 64.972% | 86.900% | 10.714% |
| `glm_5_turbo__P2_graph_rag__H2_memory` | 48.809% | 69.492% | 94.000% | 5.952% |
| `glm_5_turbo__P2_graph_rag__H2_pddl_recovery` | 52.381% | 73.446% | 100.000% | 0.000% |
| `gpt_5_5__B0_minimal_prompt__H0_open_loop` | 0.000% | 0.000% | 0.000% | 100.000% |
| `gpt_5_5__B0_minimal_prompt__H2_llm_reflection` | 76.191% | 85.876% | 91.700% | 1.190% |
| `gpt_5_5__B0_minimal_prompt__H2_memory` | 73.809% | 85.311% | 95.200% | 0.000% |
| `gpt_5_5__B0_minimal_prompt__H2_pddl_recovery` | 52.381% | 72.881% | 100.000% | 0.000% |
| `gpt_5_5__P0_engineered_prompt__H0_open_loop` | 29.762% | 52.542% | 48.800% | 1.190% |
| `gpt_5_5__P0_engineered_prompt__H2_llm_reflection` | 59.524% | 78.531% | 97.600% | 0.000% |
| `gpt_5_5__P0_engineered_prompt__H2_memory` | 55.952% | 75.706% | 97.600% | 1.190% |
| `gpt_5_5__P0_engineered_prompt__H2_pddl_recovery` | 60.714% | 80.226% | 100.000% | 0.000% |
| `gpt_5_5__P0_structured_prompt__H0_open_loop` | 41.667% | 61.017% | 63.100% | 1.190% |
| `gpt_5_5__P0_structured_prompt__H2_llm_reflection` | 58.333% | 77.966% | 95.200% | 1.190% |
| `gpt_5_5__P0_structured_prompt__H2_memory` | 55.952% | 76.271% | 96.400% | 1.190% |
| `gpt_5_5__P0_structured_prompt__H2_pddl_recovery` | 55.952% | 75.706% | 100.000% | 0.000% |
| `gpt_5_5__P1_rag__H0_open_loop` | 48.809% | 69.492% | 92.900% | 1.190% |
| `gpt_5_5__P1_rag__H2_llm_reflection` | 50.000% | 70.621% | 95.200% | 1.190% |
| `gpt_5_5__P1_rag__H2_memory` | 50.000% | 71.751% | 97.600% | 1.190% |
| `gpt_5_5__P1_rag__H2_pddl_recovery` | 50.000% | 71.751% | 97.600% | 1.190% |
| `gpt_5_5__P2_graph_rag__H0_open_loop` | 52.381% | 72.316% | 92.900% | 1.190% |
| `gpt_5_5__P2_graph_rag__H2_llm_reflection` | 51.191% | 74.011% | 98.800% | 1.190% |
| `gpt_5_5__P2_graph_rag__H2_memory` | 50.000% | 73.446% | 98.800% | 1.190% |
| `gpt_5_5__P2_graph_rag__H2_pddl_recovery` | 50.000% | 72.881% | 98.800% | 1.190% |

## Family-clustered paired comparisons

| Contrast | Uplift | Family-clustered 95% CI | McNemar p |
|---|---:|---:|---:|
| `deepseek_v4_flash__H0_open_loop__B0_minimal_prompt_to_P0_structured_prompt` | 8.33 pp | [0.00, 17.24] pp | 0.0156 |
| `deepseek_v4_flash__H0_open_loop__P0_structured_prompt_to_P0_engineered_prompt` | 4.76 pp | [-0.96, 11.69] pp | 0.3877 |
| `deepseek_v4_flash__H0_open_loop__P0_engineered_prompt_to_P1_rag` | 39.29 pp | [7.25, 67.53] pp | 0.0000 |
| `deepseek_v4_flash__H0_open_loop__P1_rag_to_P2_graph_rag` | -5.95 pp | [-12.36, 0.00] pp | 0.0625 |
| `deepseek_v4_flash__H2_llm_reflection__B0_minimal_prompt_to_P0_structured_prompt` | 8.33 pp | [-21.95, 34.41] pp | 0.3240 |
| `deepseek_v4_flash__H2_llm_reflection__P0_structured_prompt_to_P0_engineered_prompt` | -3.57 pp | [-6.67, -1.10] pp | 0.2500 |
| `deepseek_v4_flash__H2_llm_reflection__P0_engineered_prompt_to_P1_rag` | 8.33 pp | [1.30, 14.81] pp | 0.0156 |
| `deepseek_v4_flash__H2_llm_reflection__P1_rag_to_P2_graph_rag` | -1.19 pp | [-3.80, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__H2_memory__B0_minimal_prompt_to_P0_structured_prompt` | -1.19 pp | [-18.99, 12.26] pp | 1.0000 |
| `deepseek_v4_flash__H2_memory__P0_structured_prompt_to_P0_engineered_prompt` | -1.19 pp | [-2.94, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__H2_memory__P0_engineered_prompt_to_P1_rag` | 4.76 pp | [0.00, 9.65] pp | 0.1250 |
| `deepseek_v4_flash__H2_memory__P1_rag_to_P2_graph_rag` | -1.19 pp | [-3.80, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__H2_pddl_recovery__B0_minimal_prompt_to_P0_structured_prompt` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__H2_pddl_recovery__P0_structured_prompt_to_P0_engineered_prompt` | -1.19 pp | [-2.94, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__H2_pddl_recovery__P0_engineered_prompt_to_P1_rag` | 1.19 pp | [0.00, 2.94] pp | 1.0000 |
| `deepseek_v4_flash__H2_pddl_recovery__P1_rag_to_P2_graph_rag` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__B0_minimal_prompt__H0_to_H2_llm_reflection` | 39.29 pp | [25.56, 48.81] pp | 0.0000 |
| `deepseek_v4_flash__B0_minimal_prompt__H0_to_H2_memory` | 50.00 pp | [25.67, 70.11] pp | 0.0000 |
| `deepseek_v4_flash__B0_minimal_prompt__H0_to_H2_pddl_recovery` | 52.38 pp | [14.29, 84.62] pp | 0.0000 |
| `deepseek_v4_flash__P0_structured_prompt__H0_to_H2_llm_reflection` | 39.29 pp | [10.14, 64.08] pp | 0.0000 |
| `deepseek_v4_flash__P0_structured_prompt__H0_to_H2_memory` | 40.48 pp | [10.14, 65.88] pp | 0.0000 |
| `deepseek_v4_flash__P0_structured_prompt__H0_to_H2_pddl_recovery` | 44.05 pp | [10.14, 70.73] pp | 0.0000 |
| `deepseek_v4_flash__P0_engineered_prompt__H0_to_H2_llm_reflection` | 30.95 pp | [5.56, 54.67] pp | 0.0000 |
| `deepseek_v4_flash__P0_engineered_prompt__H0_to_H2_memory` | 34.52 pp | [6.94, 59.60] pp | 0.0000 |
| `deepseek_v4_flash__P0_engineered_prompt__H0_to_H2_pddl_recovery` | 38.10 pp | [7.25, 64.82] pp | 0.0000 |
| `deepseek_v4_flash__P1_rag__H0_to_H2_llm_reflection` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__P1_rag__H0_to_H2_memory` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__P1_rag__H0_to_H2_pddl_recovery` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `deepseek_v4_flash__P2_graph_rag__H0_to_H2_llm_reflection` | 4.76 pp | [0.00, 10.39] pp | 0.1250 |
| `deepseek_v4_flash__P2_graph_rag__H0_to_H2_memory` | 4.76 pp | [0.00, 10.39] pp | 0.1250 |
| `deepseek_v4_flash__P2_graph_rag__H0_to_H2_pddl_recovery` | 5.95 pp | [0.00, 12.36] pp | 0.0625 |
| `gpt_5_5__H0_open_loop__B0_minimal_prompt_to_P0_structured_prompt` | 41.67 pp | [24.62, 54.05] pp | 0.0000 |
| `gpt_5_5__H0_open_loop__P0_structured_prompt_to_P0_engineered_prompt` | -11.90 pp | [-26.53, 5.56] pp | 0.1102 |
| `gpt_5_5__H0_open_loop__P0_engineered_prompt_to_P1_rag` | 19.05 pp | [-23.19, 54.63] pp | 0.0195 |
| `gpt_5_5__H0_open_loop__P1_rag_to_P2_graph_rag` | 3.57 pp | [-3.41, 14.29] pp | 0.4531 |
| `gpt_5_5__H2_llm_reflection__B0_minimal_prompt_to_P0_structured_prompt` | -17.86 pp | [-44.59, 3.39] pp | 0.0059 |
| `gpt_5_5__H2_llm_reflection__P0_structured_prompt_to_P0_engineered_prompt` | 1.19 pp | [-3.45, 6.90] pp | 1.0000 |
| `gpt_5_5__H2_llm_reflection__P0_engineered_prompt_to_P1_rag` | -9.52 pp | [-26.09, 1.02] pp | 0.0386 |
| `gpt_5_5__H2_llm_reflection__P1_rag_to_P2_graph_rag` | 1.19 pp | [-3.03, 4.35] pp | 1.0000 |
| `gpt_5_5__H2_memory__B0_minimal_prompt_to_P0_structured_prompt` | -17.86 pp | [-38.75, -1.02] pp | 0.0041 |
| `gpt_5_5__H2_memory__P0_structured_prompt_to_P0_engineered_prompt` | 0.00 pp | [-2.75, 4.17] pp | 1.0000 |
| `gpt_5_5__H2_memory__P0_engineered_prompt_to_P1_rag` | -5.95 pp | [-22.22, 2.17] pp | 0.1250 |
| `gpt_5_5__H2_memory__P1_rag_to_P2_graph_rag` | 0.00 pp | [-4.17, 2.75] pp | 1.0000 |
| `gpt_5_5__H2_pddl_recovery__B0_minimal_prompt_to_P0_structured_prompt` | 3.57 pp | [-2.17, 13.85] pp | 0.3750 |
| `gpt_5_5__H2_pddl_recovery__P0_structured_prompt_to_P0_engineered_prompt` | 4.76 pp | [0.00, 13.16] pp | 0.2188 |
| `gpt_5_5__H2_pddl_recovery__P0_engineered_prompt_to_P1_rag` | -10.71 pp | [-24.32, -2.38] pp | 0.0039 |
| `gpt_5_5__H2_pddl_recovery__P1_rag_to_P2_graph_rag` | 0.00 pp | [-4.17, 2.75] pp | 1.0000 |
| `gpt_5_5__B0_minimal_prompt__H0_to_H2_llm_reflection` | 76.19 pp | [52.86, 92.86] pp | 0.0000 |
| `gpt_5_5__B0_minimal_prompt__H0_to_H2_memory` | 73.81 pp | [53.49, 89.61] pp | 0.0000 |
| `gpt_5_5__B0_minimal_prompt__H0_to_H2_pddl_recovery` | 52.38 pp | [14.29, 84.62] pp | 0.0000 |
| `gpt_5_5__P0_structured_prompt__H0_to_H2_llm_reflection` | 16.67 pp | [-11.39, 40.58] pp | 0.0125 |
| `gpt_5_5__P0_structured_prompt__H0_to_H2_memory` | 14.29 pp | [-13.92, 37.68] pp | 0.0357 |
| `gpt_5_5__P0_structured_prompt__H0_to_H2_pddl_recovery` | 14.29 pp | [-15.58, 38.00] pp | 0.0357 |
| `gpt_5_5__P0_engineered_prompt__H0_to_H2_llm_reflection` | 29.76 pp | [-2.74, 59.18] pp | 0.0000 |
| `gpt_5_5__P0_engineered_prompt__H0_to_H2_memory` | 26.19 pp | [-10.00, 58.25] pp | 0.0003 |
| `gpt_5_5__P0_engineered_prompt__H0_to_H2_pddl_recovery` | 30.95 pp | [-9.46, 65.35] pp | 0.0000 |
| `gpt_5_5__P1_rag__H0_to_H2_llm_reflection` | 1.19 pp | [0.00, 4.35] pp | 1.0000 |
| `gpt_5_5__P1_rag__H0_to_H2_memory` | 1.19 pp | [0.00, 4.35] pp | 1.0000 |
| `gpt_5_5__P1_rag__H0_to_H2_pddl_recovery` | 1.19 pp | [0.00, 4.35] pp | 1.0000 |
| `gpt_5_5__P2_graph_rag__H0_to_H2_llm_reflection` | -1.19 pp | [-10.53, 4.76] pp | 1.0000 |
| `gpt_5_5__P2_graph_rag__H0_to_H2_memory` | -2.38 pp | [-14.29, 4.76] pp | 0.6875 |
| `gpt_5_5__P2_graph_rag__H0_to_H2_pddl_recovery` | -2.38 pp | [-14.29, 4.76] pp | 0.6875 |
| `glm_5_turbo__H0_open_loop__B0_minimal_prompt_to_P0_structured_prompt` | 17.86 pp | [7.35, 29.27] pp | 0.0001 |
| `glm_5_turbo__H0_open_loop__P0_structured_prompt_to_P0_engineered_prompt` | -5.95 pp | [-13.68, 1.52] pp | 0.3323 |
| `glm_5_turbo__H0_open_loop__P0_engineered_prompt_to_P1_rag` | 35.71 pp | [-1.32, 65.85] pp | 0.0000 |
| `glm_5_turbo__H0_open_loop__P1_rag_to_P2_graph_rag` | -3.57 pp | [-11.34, 2.86] pp | 0.5488 |
| `glm_5_turbo__H2_llm_reflection__B0_minimal_prompt_to_P0_structured_prompt` | -14.29 pp | [-46.48, 19.45] pp | 0.0730 |
| `glm_5_turbo__H2_llm_reflection__P0_structured_prompt_to_P0_engineered_prompt` | -4.76 pp | [-8.33, -1.20] pp | 0.4240 |
| `glm_5_turbo__H2_llm_reflection__P0_engineered_prompt_to_P1_rag` | 1.19 pp | [-18.31, 13.68] pp | 1.0000 |
| `glm_5_turbo__H2_llm_reflection__P1_rag_to_P2_graph_rag` | -2.38 pp | [-8.70, 2.63] pp | 0.7266 |
| `glm_5_turbo__H2_memory__B0_minimal_prompt_to_P0_structured_prompt` | -17.86 pp | [-36.11, -2.70] pp | 0.0015 |
| `glm_5_turbo__H2_memory__P0_structured_prompt_to_P0_engineered_prompt` | 0.00 pp | [-2.75, 4.17] pp | 1.0000 |
| `glm_5_turbo__H2_memory__P0_engineered_prompt_to_P1_rag` | -1.19 pp | [-20.59, 10.91] pp | 1.0000 |
| `glm_5_turbo__H2_memory__P1_rag_to_P2_graph_rag` | -2.38 pp | [-7.60, 0.00] pp | 0.6250 |
| `glm_5_turbo__H2_pddl_recovery__B0_minimal_prompt_to_P0_structured_prompt` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__H2_pddl_recovery__P0_structured_prompt_to_P0_engineered_prompt` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__H2_pddl_recovery__P0_engineered_prompt_to_P1_rag` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__H2_pddl_recovery__P1_rag_to_P2_graph_rag` | 0.00 pp | [0.00, 0.00] pp | 1.0000 |
| `glm_5_turbo__B0_minimal_prompt__H0_to_H2_llm_reflection` | 67.86 pp | [42.71, 87.50] pp | 0.0000 |
| `glm_5_turbo__B0_minimal_prompt__H0_to_H2_memory` | 70.24 pp | [46.07, 89.53] pp | 0.0000 |
| `glm_5_turbo__B0_minimal_prompt__H0_to_H2_pddl_recovery` | 52.38 pp | [14.29, 84.62] pp | 0.0000 |
| `glm_5_turbo__P0_structured_prompt__H0_to_H2_llm_reflection` | 35.71 pp | [9.33, 60.00] pp | 0.0000 |
| `glm_5_turbo__P0_structured_prompt__H0_to_H2_memory` | 34.52 pp | [3.37, 61.90] pp | 0.0000 |
| `glm_5_turbo__P0_structured_prompt__H0_to_H2_pddl_recovery` | 34.52 pp | [-2.78, 66.32] pp | 0.0000 |
| `glm_5_turbo__P0_engineered_prompt__H0_to_H2_llm_reflection` | 36.90 pp | [12.09, 58.34] pp | 0.0000 |
| `glm_5_turbo__P0_engineered_prompt__H0_to_H2_memory` | 40.48 pp | [11.49, 66.32] pp | 0.0000 |
| `glm_5_turbo__P0_engineered_prompt__H0_to_H2_pddl_recovery` | 40.48 pp | [1.47, 73.24] pp | 0.0000 |
| `glm_5_turbo__P1_rag__H0_to_H2_llm_reflection` | 2.38 pp | [0.00, 8.70] pp | 0.5000 |
| `glm_5_turbo__P1_rag__H0_to_H2_memory` | 3.57 pp | [0.00, 9.20] pp | 0.2500 |
| `glm_5_turbo__P1_rag__H0_to_H2_pddl_recovery` | 4.76 pp | [0.00, 13.43] pp | 0.1250 |
| `glm_5_turbo__P2_graph_rag__H0_to_H2_llm_reflection` | 3.57 pp | [0.00, 6.58] pp | 0.3750 |
| `glm_5_turbo__P2_graph_rag__H0_to_H2_memory` | 4.76 pp | [-1.45, 10.94] pp | 0.2188 |
| `glm_5_turbo__P2_graph_rag__H0_to_H2_pddl_recovery` | 8.33 pp | [-1.23, 18.29] pp | 0.0391 |

## Claim boundaries

- All reported outcome scores come from the pinned official evaluator.
- The local verifier is an intervention component, not an outcome authority.
- The 84-task cohort is not the complete official hidden challenge set.
- The complete 5 x 4 planner-harness grid supports model-stratified planning, recovery, and interaction analyses.
