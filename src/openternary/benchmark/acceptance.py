"""Preregistered quality acceptance calculations."""

from __future__ import annotations

import math
from typing import Any

REQUIRED_METRICS = ("general_ppl", "japanese_ppl", "instruction_score", "collapse_count")


def _validate_metrics(name: str, metrics: dict[str, float | int]) -> None:
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"{name} metrics missing: {missing}")
    for key in REQUIRED_METRICS:
        value = metrics[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name}.{key} must be finite numeric")
    if float(metrics["general_ppl"]) <= 0 or float(metrics["japanese_ppl"]) <= 0:
        raise ValueError(f"{name} perplexities must be positive")
    if not 0 <= float(metrics["instruction_score"]) <= 100:
        raise ValueError(f"{name}.instruction_score must be in [0, 100]")
    if int(metrics["collapse_count"]) != metrics["collapse_count"] or int(metrics["collapse_count"]) < 0:
        raise ValueError(f"{name}.collapse_count must be a non-negative integer")


def compare_quality_metrics(
    control: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, Any]:
    """Apply the fixed balanced gate to one candidate/control pair."""
    _validate_metrics("control", control)
    _validate_metrics("candidate", candidate)

    general_change = (float(candidate["general_ppl"]) - float(control["general_ppl"])) / float(control["general_ppl"])
    japanese_change = (float(candidate["japanese_ppl"]) - float(control["japanese_ppl"])) / float(
        control["japanese_ppl"]
    )
    instruction_delta = float(candidate["instruction_score"]) - float(control["instruction_score"])
    composite = 0.35 * (-general_change) + 0.35 * (-japanese_change) + 0.30 * (instruction_delta / 100)

    failed_gates: list[str] = []
    if composite <= 0:
        failed_gates.append("composite_not_positive")
    if general_change > 0.02:
        failed_gates.append("general_ppl_regression")
    if japanese_change > 0.02:
        failed_gates.append("japanese_ppl_regression")
    if instruction_delta < -2.0:
        failed_gates.append("instruction_regression")
    if int(candidate["collapse_count"]) != 0:
        failed_gates.append("collapse")

    return {
        "protocol_version": "balanced-quality-gate-v1",
        "weights": {"general_ppl": 0.35, "japanese_ppl": 0.35, "instruction": 0.30},
        "limits": {"ppl_relative_regression": 0.02, "instruction_point_regression": 2.0, "collapse": 0},
        "changes": {
            "general_ppl_relative": general_change,
            "japanese_ppl_relative": japanese_change,
            "instruction_points": instruction_delta,
        },
        "composite_score": composite,
        "failed_gates": failed_gates,
        "accepted": not failed_gates,
    }


__all__ = ["compare_quality_metrics"]
