"""Tensor-level quantization metrics — Phase 2 diagnostics.

Language-model quality metrics (perplexity/MMLU)ではない。tensor復元の診断用。
"""

from __future__ import annotations

import math

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for metrics. Install with: uv sync --extra ml")


def tensor_metrics(
    orig: torch.Tensor,  # type: ignore[type-arg]
    recon: torch.Tensor,  # type: ignore[type-arg]
    codes: torch.Tensor,  # type: ignore[type-arg]
    packed: torch.Tensor,  # type: ignore[type-arg]
    scale: float,
) -> dict[str, object]:
    """単一tensorの復元diagnosticsを計算.

    Args:
        orig: 元tensor (float)
        recon: 復元tensor (dequantized)
        codes: 三値codes (int8, -1/0/+1)
        packed: packed bytes (uint8, CPU)
        scale: 使用されたscale

    Returns:
        metrics dict
    """
    _require_torch()

    # flattenしてfloat32で計算
    o = orig.detach().to(torch.float32).reshape(-1)  # type: ignore[union-attr]
    r = recon.detach().to(torch.float32).reshape(-1)  # type: ignore[union-attr]
    diff = o - r

    mae = float(diff.abs().mean().item()) if o.numel() > 0 else 0.0
    mse = float((diff * diff).mean().item()) if o.numel() > 0 else 0.0
    rmse = math.sqrt(mse)
    max_abs = float(diff.abs().max().item()) if o.numel() > 0 else 0.0

    # cosine — zero vectorなら null (None)
    cosine: float | None
    cos_defined = False
    orig_norm = float(torch.norm(o).item()) if o.numel() > 0 else 0.0  # type: ignore[union-attr]
    recon_norm = float(torch.norm(r).item()) if r.numel() > 0 else 0.0  # type: ignore[union-attr]
    if orig_norm == 0.0 or recon_norm == 0.0:
        cosine = None
    else:
        # cosine = dot / (norm_orig * norm_recon)
        dot = float((o * r).sum().item())  # type: ignore[union-attr]
        cosine = dot / (orig_norm * recon_norm)
        # 数値誤差で±1を超える場合をclamp
        cosine = max(-1.0, min(1.0, cosine))
        cos_defined = True  # noqa: F841 — 将来拡張用に保持

    # ratio
    n = int(codes.numel())  # type: ignore[union-attr]
    flat_codes = codes.reshape(-1)  # type: ignore[union-attr]
    zero_ratio = float((flat_codes == 0).float().mean().item()) if n > 0 else 0.0  # type: ignore[union-attr]
    neg_ratio = float((flat_codes == -1).float().mean().item()) if n > 0 else 0.0  # type: ignore[union-attr]
    pos_ratio = float((flat_codes == 1).float().mean().item()) if n > 0 else 0.0  # type: ignore[union-attr]

    # bytes
    orig_nbytes = int(o.numel() * 2)  # BF16前提の元サイズを2byte/paramで計算（header由来ではない簡易値）
    # 実際のheader dtypeは別だが、metricsではpackedとの比較のため2byteを仮定
    # より正確には呼び出し元がheader dtypeを渡すべきだが、Phase2では簡易的にBF16=2とする
    packed_bytes = int(packed.numel())  # type: ignore[union-attr]
    scale_bytes = 4  # FP32

    if packed_bytes > 0:
        weight_only_ratio = orig_nbytes / packed_bytes
        effective_ratio = orig_nbytes / (packed_bytes + scale_bytes)
    else:
        weight_only_ratio = 0.0
        effective_ratio = 0.0

    # byte padding
    logical_bits = n * 2
    packed_bits = packed_bytes * 8
    padding_bits = packed_bits - logical_bits

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "cosine_similarity": cosine,
        "zero_ratio": zero_ratio,
        "negative_ratio": neg_ratio,
        "positive_ratio": pos_ratio,
        "original_bytes": orig_nbytes,
        "packed_weight_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "weight_only_compression_ratio": weight_only_ratio,
        "effective_compression_ratio": effective_ratio,
        "effective_includes_scale": True,
        "logical_2bit_bits": logical_bits,
        "packed_weight_bits": packed_bits,
        "byte_padding_bits": padding_bits,
        "scale": scale,
        "numel": n,
    }
