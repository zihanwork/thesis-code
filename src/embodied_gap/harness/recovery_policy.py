from __future__ import annotations

from enum import StrEnum


class HarnessMode(StrEnum):
    H0_OPEN_LOOP = "H0_open_loop"
    H2_LLM_REFLECTION = "H2_llm_reflection"
    H2_MEMORY = "H2_memory"
    H2_PDDL_RECOVERY = "H2_pddl_recovery"
