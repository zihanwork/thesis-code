from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeanDifference:
    left: str
    right: str
    difference: float


def paired_mean_difference(
    left_name: str,
    right_name: str,
    left_values: list[float],
    right_values: list[float],
) -> MeanDifference:
    if len(left_values) != len(right_values):
        raise ValueError("Paired values must have the same length.")
    if not left_values:
        return MeanDifference(left_name, right_name, 0.0)
    diffs = [right - left for left, right in zip(left_values, right_values, strict=True)]
    return MeanDifference(left_name, right_name, sum(diffs) / len(diffs))
