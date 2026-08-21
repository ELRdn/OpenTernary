"""AbsMean ternary quantization — Phase 2 canonical baseline.

Formula:
    scale = mean(abs(W))  # FP32 accumulation
    if scale == 0:
        Q = zeros_like(W)
    else:
        Q = clamp(round(W / scale), -1, +1)  # ties-to-even via torch.round

Device-preserving: codes.device == w.device. No implicit .cpu().
"""

from __future__ import annotations

import dataclasses

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class TernaryTensor:
    """三値化結果の中間表現.

    scaleはPython floatだがcanonical serialized formは IEEE FP32。
    codesは torch.int8, values ∈ {-1,0,1}, deviceは入力deviceを保持。
    """

    codes: torch.Tensor  # type: ignore[type-arg]
    scale: float
    shape: tuple[int, ...]
    orig_dtype: str
    scale_dtype: str = "float32"
    scheme: str = "absmean"
    version: str = "1"


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for ternary quantization. Install with: uv sync --extra ml")


def quantize_absmean(w: torch.Tensor) -> TernaryTensor:  # type: ignore[type-arg]
    """Per-tensor AbsMean三値化.

    Args:
        w: 浮動小数点tensor（CPU/CUDAいずれも可）

    Returns:
        TernaryTensor (codesはinput device上に残る)

    Raises:
        ImportError: torch未導入
        ValueError: empty / non-floating / complex / NaN/Inf
    """
    _require_torch()

    # 入力契約 — G19
    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    if not w.is_floating_point():
        raise ValueError(f"quantize_absmean requires floating tensor, got dtype={w.dtype}")
    if w.is_complex():
        raise ValueError(f"quantize_absmean does not support complex tensor, got dtype={w.dtype}")
    if not torch.isfinite(w).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in input tensor")

    # FP32へ上げてからAbsMean — G16
    # deviceは保持される: to(torch.float32) はdtype変換のみ
    wf = w.detach().to(torch.float32)
    scale_t = wf.abs().mean()
    scale = float(scale_t.item())

    if scale == 0.0:
        codes = torch.zeros(w.shape, dtype=torch.int8, device=w.device)
        return TernaryTensor(
            codes=codes,
            scale=0.0,
            shape=tuple(w.shape),
            orig_dtype=str(w.dtype),
            scale_dtype="float32",
            scheme="absmean",
            version="1",
        )

    # roundはties-to-even (PyTorch仕様) — docs明記
    q = torch.round(wf / scale).clamp(-1, 1).to(torch.int8)
    # q.device == w.device であることを保証（wfはw.device上のfloat32）
    return TernaryTensor(
        codes=q,
        scale=scale,
        shape=tuple(w.shape),
        orig_dtype=str(w.dtype),
        scale_dtype="float32",
        scheme="absmean",
        version="1",
    )


def dequantize(tt: TernaryTensor) -> torch.Tensor:  # type: ignore[type-arg]
    """TernaryTensorをW_hat = Q * scaleへ復元.

    deviceはcodes.device上に維持。
    """
    _require_torch()
    if tt.scale == 0.0:
        return torch.zeros(tt.shape, dtype=torch.float32, device=tt.codes.device)
    # codes(int8) → float32 → scale乗算
    return tt.codes.to(torch.float32) * tt.scale
