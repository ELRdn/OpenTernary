"""Experimental straight-through ternary assignment from trainable code logits."""

from __future__ import annotations

import torch


def hard_logit_codes(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Map logits to {-1, 0, 1}; equality stays zero."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    return torch.where(logits > threshold, 1.0, torch.where(logits < -threshold, -1.0, 0.0))


def ste_logit_codes(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Use hard codes in forward and a clipped identity surrogate in backward."""
    hard = hard_logit_codes(logits, threshold)
    surrogate = logits.clamp(-1.0, 1.0)
    return hard.detach() - surrogate.detach() + surrogate
