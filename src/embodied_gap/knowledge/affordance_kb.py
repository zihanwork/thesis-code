from __future__ import annotations

AFFORDANCE_PREFIXES = (
    "openable(",
    "pickupable(",
    "surface(",
    "sink(",
    "cleanable(",
    "heatable(",
)


def is_affordance_fact(fact: str) -> bool:
    return fact.startswith(AFFORDANCE_PREFIXES)
