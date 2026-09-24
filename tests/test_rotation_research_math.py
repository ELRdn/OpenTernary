"""Mathematical checks for the isolated rotation research probes."""

import torch
import torch.nn.functional as functional

from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from scripts.probe_gemma4_ternary_scales import lloyd_reconstruct
from scripts.train_gemma4_per_block_rotation import apply_per_block, rotations


def test_per_block_orthogonal_map_preserves_linear_function() -> None:
    generator = torch.Generator().manual_seed(37)
    raw = torch.randn(2, 128, 128, generator=generator) * 0.01
    matrices = rotations(raw)
    inputs = torch.randn(4, 256, generator=generator)
    weights = torch.randn(5, 256, generator=generator)
    reference = functional.linear(inputs, weights)
    transformed = functional.linear(apply_per_block(inputs, matrices), apply_per_block(weights, matrices))
    assert torch.allclose(transformed, reference, atol=1e-4)


def test_lloyd_scales_do_not_increase_weight_mse() -> None:
    generator = torch.Generator().manual_seed(41)
    weights = torch.randn(7, 256, generator=generator)
    baseline = dequantize_groupwise(quantize_groupwise(weights, 128))
    optimized = lloyd_reconstruct(weights, 10)
    assert (optimized - weights).square().mean() <= (baseline - weights).square().mean() + 1e-7
