from __future__ import annotations

from enum import StrEnum


class HarnessMode(StrEnum):
    H0_OPEN_LOOP = "H0_open_loop"
    H1_VERIFIER_GATED = "H1_verifier_gated"
    H2_FULL_RECOVERY = "H2_full_recovery"
