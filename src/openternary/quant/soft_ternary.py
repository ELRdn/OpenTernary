"""Differentiable ternary assignments and deterministic hardening.

The soft path is intended for calibration only.  Persisted weights must always
be produced by :func:`harden_soft_codes` (or an equivalent hard quantizer).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

TemperatureSchedule = Literal["linear", "cosine", "exponential"]


def temperature_at_step(
    step: int,
    total_steps: int,
    *,
    start: float = 1.0,
    end: float = 0.05,
    schedule: TemperatureSchedule = "linear",
    hard_fraction: float = 0.1,
) -> float:
    """Return the annealed temperature, using zero for the final hard region."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= step < total_steps:
        raise ValueError("step must satisfy 0 <= step < total_steps")
    if not math.isfinite(start) or not math.isfinite(end) or start < end or end <= 0:
        raise ValueError("temperatures must be finite and satisfy start >= end > 0")
    if not 0 <= hard_fraction < 1:
        raise ValueError("hard_fraction must satisfy 0 <= hard_fraction < 1")
    if schedule not in {"linear", "cosine", "exponential"}:
        raise ValueError(f"unsupported temperature schedule: {schedule}")

    hard_steps = max(1, math.ceil(total_steps * hard_fraction)) if hard_fraction > 0 else 0
    hard_start = total_steps - hard_steps
    if step >= hard_start:
        return 0.0

    # Reach ``end`` on the last soft step while keeping the first step at
    # ``start``.  A one-step soft region is therefore exactly ``start``.
    progress = 0.0 if hard_start <= 1 else step / (hard_start - 1)
    if schedule == "linear":
        return start + (end - start) * progress
    if schedule == "cosine":
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return start * math.exp(math.log(end / start) * progress)


def _validate_weights(weights: Tensor) -> None:
    if not weights.is_floating_point():
        raise TypeError("weights must be a floating-point tensor")
    if weights.numel() == 0:
        raise ValueError("weights must not be empty")
    if not torch.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")


def _expanded_reference_scale(
    weights: Tensor,
    reference_scale: float | Tensor,
    group_size: int | None,
) -> Tensor:
    _validate_weights(weights)
    if isinstance(reference_scale, Tensor):
        if group_size is None or group_size < 1:
            raise ValueError("group_size is required for tensor reference_scale")
        if reference_scale.ndim != 1 or not reference_scale.is_floating_point():
            raise ValueError("tensor reference_scale must be a floating-point vector")
        if not torch.isfinite(reference_scale).all() or (reference_scale < 0).any():
            raise ValueError("reference_scale must be finite and non-negative")
        from openternary.quant.threshold import _expand_per_group

        return _expand_per_group(reference_scale.to(weights.device), tuple(weights.shape), group_size)
    if not math.isfinite(reference_scale) or reference_scale < 0:
        raise ValueError("reference_scale must be finite and non-negative")
    return torch.full_like(weights, reference_scale)


def harden_soft_codes(
    weights: Tensor,
    *,
    reference_scale: float | Tensor,
    group_size: int | None = None,
) -> Tensor:
    """Map weights deterministically to ``{-1, 0, 1}`` using midpoint ties."""

    expanded_scale = _expanded_reference_scale(weights, reference_scale, group_size)
    nonzero = expanded_scale > 0
    normalized = torch.where(nonzero, weights / expanded_scale.clamp_min(torch.finfo(weights.dtype).tiny), 0.0)
    return torch.where(
        nonzero & (normalized.abs() > 0.5),
        normalized.sign(),
        torch.zeros_like(normalized),
    ).to(torch.int8)


def soft_ternary_codes(
    weights: Tensor,
    *,
    reference_scale: float | Tensor,
    group_size: int | None = None,
    temperature: float,
    zero_logit_bias: float = 0.0,
) -> Tensor:
    """Return differentiable expected ternary codes for calibration.

    ``zero_logit_bias`` is the only modulation: positive values increase the
    probability of the zero code without changing the hardening contract.
    """

    expanded_scale = _expanded_reference_scale(weights, reference_scale, group_size)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(zero_logit_bias):
        raise ValueError("zero_logit_bias must be finite")
    nonzero = expanded_scale > 0
    normalized = torch.where(nonzero, weights / expanded_scale.clamp_min(torch.finfo(weights.dtype).tiny), 0.0)
    codebook = normalized.new_tensor((-1.0, 0.0, 1.0))
    logits = -(normalized.unsqueeze(-1) - codebook).square() / temperature
    if zero_logit_bias:
        logits[..., 1] = logits[..., 1] + zero_logit_bias
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities * codebook).sum(dim=-1) * nonzero


__all__ = [
    "TemperatureSchedule",
    "harden_soft_codes",
    "soft_ternary_codes",
    "temperature_at_step",
]
