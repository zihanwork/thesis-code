# Multi-model Ablation Summary

Each row is a `<model>_<variant>_outputs.json` evaluation. `baseline` rows
are kept only when an explicit baseline directory exists; same-name model
directories are interpreted as the implicit baseline.

## action_sequencing

| Base model | Variant | Task success | Execution success | Missing step | Wrong order | Hallucination |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-flash` | `baseline` | 75.58 | 82.60 | 12.79 | 0.00 | 2.33 |
| `deepseek-v4-flash` | `few_shot` | 79.65 | 84.90 | 10.47 | 0.00 | 2.33 |
| `deepseek-v4-flash` | `format_constraints` | 79.07 | 86.00 | 7.56 | 0.00 | 4.07 |
| `deepseek-v4-flash` | `plan_then_ground` | 80.23 | 86.60 | 8.14 | 0.00 | 2.33 |
| `deepseek-v4-flash` | `sg_rag` | 79.07 | 87.20 | 8.72 | 0.00 | 2.33 |
| `deepseek-v4-flash` | `sg_rag_pc_kg` | 60.47 | 79.70 | 16.86 | 0.00 | 2.33 |
| `deepseek-v4-flash_pc_kg` | `self_check` | 52.33 | 69.80 | 23.84 | 0.00 | 2.33 |
| `glm-5-turbo` | `baseline` | 88.37 | 93.00 | 2.33 | 0.00 | 2.33 |
| `glm-5-turbo` | `sg_rag` | 87.21 | 89.50 | 5.81 | 2.33 | 2.33 |
| `gold_oracle` | `baseline` | 2.62 | 89.20 | 1.64 | 0.33 | 8.85 |
| `minimax-m2-stable` | `baseline` | 86.05 | 90.70 | 3.49 | 1.16 | 2.33 |
| `minimax-m2-stable` | `sg_rag` | 79.07 | 86.00 | 6.98 | 2.33 | 2.33 |

## goal_interpretation

| Base model | Variant | All F1 | Node F1 | Edge F1 | Action F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `deepseek-v4-flash` | `baseline` | 38.97 | 49.74 | 33.06 | 27.47 |
| `deepseek-v4-flash` | `decompose_then_merge` | 39.00 | 49.05 | 35.23 | 27.55 |
| `deepseek-v4-flash` | `few_shot` | 38.97 | 51.05 | 35.56 | 25.70 |
| `deepseek-v4-flash` | `schema_constrained` | 40.60 | 50.17 | 35.85 | 30.08 |

