"""Numerical contract for the BF16-to-rotated-hard-G128 pilot."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch

from openternary.quant.grouping import quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim


@pytest.fixture
def pilot(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_gemma4_global_rotation_q1")


def test_cayley_inverse_preserves_bf16_linear_function(pilot: Any) -> None:
    generator = torch.Generator().manual_seed(42)
    raw = torch.randn(128, 128, generator=generator) * 0.01
    rotation = pilot.cayley_from_raw(raw)
    identity = torch.eye(128)
    skew = torch.linalg.solve(identity + rotation, identity - rotation)
    recovered = pilot.cayley_from_raw((skew - skew.T) / 4)
    assert torch.allclose(rotation, recovered, atol=1e-5)
    assert torch.allclose(rotation.T @ rotation, identity, atol=1e-5)
    weights = torch.randn(4, 256, generator=generator)
    inputs = torch.randn(3, 256, generator=generator)
    transformed = apply_block_rotation(hadamard_last_dim(weights, 128), rotation)
    rotated_inputs = apply_block_rotation(hadamard_last_dim(inputs, 128), rotation)
    assert torch.allclose(rotated_inputs @ transformed.T, inputs @ weights.T, atol=1e-4)


def test_hard_g128_matches_canonical_codes_and_has_gradient(pilot: Any) -> None:
    generator = torch.Generator().manual_seed(7)
    weights = torch.nn.Parameter(torch.randn(4, 256, generator=generator))
    reconstructed, codes, scales = pilot.hard_g128(weights, ste=True)
    canonical = quantize_groupwise(weights.detach(), 128)
    assert torch.equal(codes, canonical.codes)
    assert torch.equal(scales, canonical.scales)
    assert torch.equal(reconstructed.detach(), codes.float() * scales.repeat_interleave(128).reshape_as(weights))
    reconstructed.square().mean().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert weights.grad.abs().sum() > 0
