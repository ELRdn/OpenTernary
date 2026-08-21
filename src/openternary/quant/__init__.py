"""Quant package — Phase 2.

Torch-dependent modules (ternary/packing/metrics)はlazy importのみ。
accountingのみstdlibで即import可能なのでheader-only inspectからtorchなしで使える。
"""

from __future__ import annotations

# stdlibのみ — eager import OK（torch不要）
from openternary.quant.accounting import (
    LOG2_3,
    byte_padding_bits_for_n,
    estimate_whole_model,
    ideal_ternary_bits,
    logical_2bit_bits,
    packed_weight_bits_for_n,
    scale_overhead_bits,
    ternary_alphabet_bits,
    theoretical_entropy_bits,
)

__all__ = [
    "LOG2_3",
    "ideal_ternary_bits",
    "ternary_alphabet_bits",
    "theoretical_entropy_bits",
    "logical_2bit_bits",
    "packed_weight_bits_for_n",
    "byte_padding_bits_for_n",
    "scale_overhead_bits",
    "estimate_whole_model",
]


# Torch-dependent — lazy accessors（import時にtorchを要求しない）
def __getattr__(name: str) -> object:
    if name in {"TernaryTensor", "quantize_absmean", "dequantize"}:
        from openternary.quant import ternary as _ternary

        return getattr(_ternary, name)
    if name in {"pack_ternary", "unpack_ternary", "pack_bytes"}:
        from openternary.quant import packing as _packing

        return getattr(_packing, name)
    if name == "tensor_metrics":
        from openternary.quant import metrics as _metrics

        return getattr(_metrics, name)
    if name in {"GroupwiseResult", "quantize_groupwise", "dequantize_groupwise", "num_groups_for_shape"}:
        from openternary.quant import grouping as _grouping

        return getattr(_grouping, name)
    if name in {"convert_snapshot", "MAX_SHARD_SIZE"}:
        from openternary.quant import fake_quant as _fq

        return getattr(_fq, name)
    raise AttributeError(f"module 'openternary.quant' has no attribute {name!r}")
