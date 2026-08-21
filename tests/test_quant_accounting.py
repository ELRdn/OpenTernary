"""Accounting tests — Phase 2 per-tensor ceil sum + ideal bits."""

from __future__ import annotations

import math

import pytest

from openternary.quant.accounting import (
    LOG2_3,
    byte_padding_bits_for_n,
    estimate_whole_model,
    ideal_ternary_bits,
    logical_2bit_bits,
    packed_weight_bits_for_n,
    scale_overhead_bits,
)


def test_ideal_ternary_bits() -> None:
    assert abs(ideal_ternary_bits(1) - LOG2_3) < 1e-12
    assert abs(ideal_ternary_bits(100) - 100 * LOG2_3) < 1e-9
    assert ideal_ternary_bits(0) == 0.0


def test_logical_2bit() -> None:
    assert logical_2bit_bits(1) == 2
    assert logical_2bit_bits(4) == 8
    assert logical_2bit_bits(0) == 0


def test_packed_formula() -> None:
    assert packed_weight_bits_for_n(0) == 0
    assert packed_weight_bits_for_n(1) == 8
    assert packed_weight_bits_for_n(4) == 8
    assert packed_weight_bits_for_n(5) == 16
    assert packed_weight_bits_for_n(8) == 16


def test_byte_padding() -> None:
    assert byte_padding_bits_for_n(1) == 6  # 8-2
    assert byte_padding_bits_for_n(4) == 0  # 8-8
    assert byte_padding_bits_for_n(5) == 6  # 16-10


def test_scale_overhead() -> None:
    assert scale_overhead_bits(0) == 0
    assert scale_overhead_bits(1, "fp32") == 32
    assert scale_overhead_bits(2, "fp32") == 64
    assert scale_overhead_bits(1, "bf16") == 16
    with pytest.raises(ValueError):
        scale_overhead_bits(1, "unknown")


def test_packed_per_tensor_sum_not_total_ceil() -> None:
    # P0バグ回帰: 2 tensors 各1 weight → total ceil(2/4)=8 ではなく 16
    classified = [
        {"name": "a", "param_count": 1, "dtype": "BF16", "quantizable": True},
        {"name": "b", "param_count": 1, "dtype": "BF16", "quantizable": True},
    ]
    est = estimate_whole_model(classified, scale_dtype="fp32")
    assert est["packed_weight_bits"] == 16  # 8+8, not 8
    assert est["logical_2bit_bits"] == 4
    assert est["byte_padding_bits"] == 12  # 16-4


def test_estimate_whole_model_basic() -> None:
    classified = [
        {"name": "model.layers.0.self_attn.q_proj.weight", "param_count": 100, "dtype": "BF16", "quantizable": True},
        {"name": "model.embed_tokens.weight", "param_count": 200, "dtype": "BF16", "quantizable": False},
    ]
    est = estimate_whole_model(classified, scale_dtype="fp32")
    assert est["quantizable_params"] == 100
    assert est["num_quantizable_tensors"] == 1
    assert est["packed_weight_bits"] == packed_weight_bits_for_n(100)
    assert est["ideal_ternary_bits"] == pytest.approx(100 * LOG2_3)
    assert est["ternary_alphabet_bits"] == est["ideal_ternary_bits"]
    assert est["theoretical_entropy_bits"] == est["ideal_ternary_bits"]
    assert est["scale_overhead_bits"] == 32
    assert est["excluded_original_bits"] == 200 * 16  # BF16 16bit
    assert est["actual_file_size_bytes"] is None


def test_estimate_total_consistency() -> None:
    classified = [
        {"name": "a", "param_count": 5, "dtype": "BF16", "quantizable": True},
        {"name": "b", "param_count": 7, "dtype": "BF16", "quantizable": True},
        {"name": "c", "param_count": 10, "dtype": "BF16", "quantizable": False},
    ]
    est = estimate_whole_model(classified, scale_dtype="fp32")
    expected_packed = packed_weight_bits_for_n(5) + packed_weight_bits_for_n(7)  # per-tensor
    assert est["packed_weight_bits"] == expected_packed
    expected_total = expected_packed + 2 * 32 + 10 * 16
    assert est["estimated_total_bits"] == expected_total
    assert est["estimated_total_bytes"] == math.ceil(expected_total / 8)


def test_actual_file_null() -> None:
    est = estimate_whole_model([], scale_dtype="fp32")
    assert est["actual_file_size_bytes"] is None


def test_empty_classified() -> None:
    est = estimate_whole_model([], scale_dtype="fp32")
    assert est["quantizable_params"] == 0
    assert est["packed_weight_bits"] == 0
    assert est["ideal_ternary_bits"] == 0.0
