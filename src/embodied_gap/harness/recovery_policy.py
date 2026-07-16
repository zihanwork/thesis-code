from __future__ import annotations

from enum import StrEnum


class HarnessMode(StrEnum):
    H0_OPEN_LOOP = "H0_open_loop"
    H1_VERIFIER_GATED = "H1_verifier_gated"
    H2_LOCAL_RECOVERY = "H2_local_recovery"
    H2_LLM_REFLECTION = "H2_llm_reflection"
    H2_ERROR_SPECIFIC = "H2_error_specific"
    H2_MEMORY = "H2_memory"
    H2_COMBINED = "H2_combined"
    H2_COMBINED_NO_LOCAL = "H2_combined_no_local"
    H2_COMBINED_NO_ERROR = "H2_combined_no_error"
    H2_COMBINED_NO_MEMORY = "H2_combined_no_memory"
    H2_PDDL_RECOVERY = "H2_pddl_recovery"
    # Legacy mixed policy retained only to reproduce historical pilot runs.
    H2_FULL_RECOVERY = "H2_full_recovery"
