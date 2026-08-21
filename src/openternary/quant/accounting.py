"""Size accounting — Phase 2 2-bit packing + ideal ternary accounting.

stdlibのみ。header-only inspectからtorchなしでimport可能にすることが
G18の要件。torchに依存する計算は行わない。
"""

from __future__ import annotations

import math

# log2(3) ≈ 1.584962500721156 — 三値alphabetの理想情報量（等確率時の最大値）
# 実データのShannon entropyではない。
LOG2_3: float = math.log2(3.0)  # 1.584962500721156

_SCALE_DTYPE_BITS: dict[str, int] = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
}

# header由来のdtype → 1要素あたりバイト数
_DTYPE_NBYTES: dict[str, int] = {
    "F32": 4,
    "F64": 8,
    "F16": 2,
    "BF16": 2,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}


def ideal_ternary_bits(n: int) -> float:
    """N個の三値重みを区別するのに必要な理想情報量 (N * log2(3))."""
    if n < 0:
        raise ValueError(f"n must be >=0, got {n}")
    return float(n) * LOG2_3


# 互換エイリアス — 名称変更前の呼び出しに対応
ternary_alphabet_bits = ideal_ternary_bits

# 旧名称は非推奨だが互換のため残す（docsではideal_ternary_bitsを推奨）
theoretical_entropy_bits = ideal_ternary_bits  # type: ignore[assignment]


def logical_2bit_bits(n: int) -> int:
    """2bit/weightの論理サイズ（paddingなし）."""
    if n < 0:
        raise ValueError(f"n must be >=0, got {n}")
    return n * 2


def packed_weight_bits_for_n(n: int) -> int:
    """Phase 2 2-bit packing物理サイズ: ceil(N/4)*8 bits."""
    if n < 0:
        raise ValueError(f"n must be >=0, got {n}")
    return ((n + 3) // 4) * 8 if n > 0 else 0


def byte_padding_bits_for_n(n: int) -> int:
    """1 tensorにおけるbyte境界padding bits = packed - logical_2bit."""
    return packed_weight_bits_for_n(n) - logical_2bit_bits(n)


def scale_overhead_bits(num_tensors: int, scale_dtype: str = "fp32") -> int:
    """Per-tensor scaleのoverhead bits。canonicalはfp32=32bit/tensor."""
    if num_tensors < 0:
        raise ValueError(f"num_tensors must be >=0, got {num_tensors}")
    key = scale_dtype.lower()
    if key not in _SCALE_DTYPE_BITS:
        raise ValueError(f"Unsupported scale_dtype '{scale_dtype}', supported: {sorted(_SCALE_DTYPE_BITS)}")
    return num_tensors * _SCALE_DTYPE_BITS[key]


def _excluded_original_bits(classified: list[dict[str, object]]) -> int:
    """非quantizable tensorの元表現bits（header dtypeに基づく）."""
    total = 0
    for c in classified:
        if c.get("quantizable"):
            continue
        dtype = str(c.get("dtype", "BF16"))
        nbytes = _DTYPE_NBYTES.get(dtype.upper(), 2)
        param_count = int(c.get("param_count", 0))  # type: ignore
        total += param_count * nbytes * 8
    return total


def estimate_whole_model(
    classified: list[dict[str, object]],
    scale_dtype: str = "fp32",
) -> dict[str, object]:
    """Whole-model ternary size見積もり.

    packedはper-tensor ceilの総和で計算する（totalで一括ceilしない）。
    """
    quantizable = [c for c in classified if c.get("quantizable")]
    num_quantizable = len(quantizable)
    quant_params = sum(int(c.get("param_count", 0)) for c in quantizable)  # type: ignore

    # per-tensor packed sum — P0バグ修正
    packed_bits = sum(packed_weight_bits_for_n(int(c.get("param_count", 0))) for c in quantizable)  # type: ignore
    logical_bits = logical_2bit_bits(quant_params)
    padding_bits = packed_bits - logical_bits
    ideal_bits = ideal_ternary_bits(quant_params)
    scale_bits = scale_overhead_bits(num_quantizable, scale_dtype)
    excluded_bits = _excluded_original_bits(classified)

    estimated_total_bits = packed_bits + scale_bits + excluded_bits
    estimated_total_bytes = math.ceil(estimated_total_bits / 8) if estimated_total_bits else 0

    return {
        "scheme": "absmean-per-tensor",
        "packing": "2bit-v1",
        "scale_dtype": scale_dtype,
        "num_quantizable_tensors": num_quantizable,
        "quantizable_params": quant_params,
        "ideal_ternary_bits": ideal_bits,
        # 互換: 旧名称も併記（docsではidealを推奨）
        "ternary_alphabet_bits": ideal_bits,
        "theoretical_entropy_bits": ideal_bits,
        "logical_2bit_bits": logical_bits,
        "packed_weight_bits": packed_bits,
        "byte_aligned_packed_bits": packed_bits,
        "byte_padding_bits": padding_bits,
        "scale_overhead_bits": scale_bits,
        "scale_overhead_bytes": math.ceil(scale_bits / 8) if scale_bits else 0,
        "excluded_original_bits": excluded_bits,
        "excluded_original_bytes": math.ceil(excluded_bits / 8) if excluded_bits else 0,
        "estimated_total_bits": estimated_total_bits,
        "estimated_total_bytes": estimated_total_bytes,
        "actual_file_size_bytes": None,
    }
