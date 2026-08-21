"""Group-wise ternary quantization — LD-RW canonical.

Grouping scheme: last-dim row-wise v1.
- W.reshape(-1, W.shape[-1]) で各row独立にGごと分割。
- tailは各rowの末尾で独立group（smaller）。
"""

from __future__ import annotations

import dataclasses

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class GroupwiseResult:
    """Group-wise三値化結果."""

    codes: torch.Tensor  # type: ignore  # int8, same shape
    scales: torch.Tensor  # type: ignore  # float32 1-D [num_groups]
    shape: tuple[int, ...]
    orig_dtype: str
    group_size: int
    grouping_scheme: str = "last-dim-rowwise-v1"
    scale_dtype: str = "float32"


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for grouping. Install with: uv sync --extra ml")


def num_groups_for_shape(
    shape: tuple[int, ...] | list[int], group_size: int, scheme: str = "last-dim-rowwise-v1"
) -> int:
    """Shapeに対するgroup数を計算（LD-RW）。"""
    if scheme != "last-dim-rowwise-v1":
        raise ValueError(f"Unsupported grouping_scheme '{scheme}'")
    if group_size <= 0:
        raise ValueError(f"group_size must be >0, got {group_size}")
    if not shape:
        return 1
    last_dim = int(shape[-1])
    if last_dim == 0:
        return 0
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    n_full = last_dim // group_size
    tail = last_dim % group_size
    per_row = n_full + (1 if tail else 0)
    return rows * per_row


def quantize_groupwise(
    w: torch.Tensor,  # type: ignore
    group_size: int,
    grouping_scheme: str = "last-dim-rowwise-v1",
) -> GroupwiseResult:
    """Last-dim row-wise group-wise AbsMean三値化（vectorized, zero-branch）."""
    _require_torch()

    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    if not w.is_floating_point():
        raise ValueError(f"groupwise requires floating tensor, got {w.dtype}")
    if w.is_complex():
        raise ValueError(f"complex not supported, got {w.dtype}")
    if not torch.isfinite(w).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected")
    if group_size <= 0:
        raise ValueError(f"group_size must be >0, got {group_size}")
    if grouping_scheme != "last-dim-rowwise-v1":
        raise ValueError(f"Unsupported grouping_scheme '{grouping_scheme}'")

    orig_shape = tuple(w.shape)
    last_dim = int(w.shape[-1]) if w.dim() > 0 else 1
    rows = w.numel() // last_dim if w.dim() > 0 else 1

    # FP32で作業、device保持
    wf = w.detach().to(torch.float32)
    # reshape to [R, D]
    if w.dim() == 0:
        # scalar treated as [1,1]
        flat_row = wf.reshape(1, 1)
        # group_size 1 → single group
        n_full = 0
        tail = 1
        # scalar case: single group scale = abs value
        scale = float(flat_row.abs().mean().item())
        if scale == 0:
            codes = torch.zeros(orig_shape, dtype=torch.int8, device=w.device)
            scales = torch.zeros(1, dtype=torch.float32, device=w.device)
        else:
            codes = torch.round(flat_row / scale).clamp(-1, 1).to(torch.int8).reshape(orig_shape).to(w.device)
            scales = torch.tensor([scale], dtype=torch.float32, device=w.device)
        return GroupwiseResult(
            codes=codes,
            scales=scales,
            shape=orig_shape,
            orig_dtype=str(w.dtype),
            group_size=group_size,
            grouping_scheme=grouping_scheme,
        )

    flat_row = wf.reshape(rows, last_dim)
    n_full = last_dim // group_size
    tail = last_dim % group_size

    # prepare outputs
    codes_row = torch.empty_like(flat_row, dtype=torch.int8)
    # per-row interleaved scales: each row's full groups + tail contiguous
    # we collect per-row then flatten
    scales_full = None
    scales_tail = None
    if n_full > 0:
        full = flat_row[:, : n_full * group_size].reshape(rows, n_full, group_size)
        scales_full = full.abs().mean(dim=-1)  # [R, n_full]
        zero_full = scales_full == 0
        safe_full = torch.where(zero_full, torch.ones_like(scales_full), scales_full)
        codes_full = torch.round(full / safe_full.unsqueeze(-1)).clamp(-1, 1).to(torch.int8)
        codes_full = torch.where(zero_full.unsqueeze(-1), torch.zeros_like(codes_full), codes_full)
        codes_row[:, : n_full * group_size] = codes_full.reshape(rows, n_full * group_size).to(torch.int8)

    if tail > 0:
        tail_tensor = flat_row[:, n_full * group_size :]  # [R, tail]
        scales_tail = tail_tensor.abs().mean(dim=-1)  # [R]
        zero_tail = scales_tail == 0
        safe_tail = torch.where(zero_tail, torch.ones_like(scales_tail), scales_tail)
        codes_tail = torch.round(tail_tensor / safe_tail.unsqueeze(-1)).clamp(-1, 1).to(torch.int8)
        codes_tail = torch.where(zero_tail.unsqueeze(-1), torch.zeros_like(codes_tail), codes_tail)
        codes_row[:, n_full * group_size :] = codes_tail.to(torch.int8)

    # build scales in per-row interleaved order: [row0_g0, row0_g1, ..., row0_tail, row1_g0, ...]
    if n_full > 0 and tail > 0:
        assert scales_full is not None and scales_tail is not None
        scales = torch.cat([scales_full, scales_tail.unsqueeze(1)], dim=1).reshape(-1).to(torch.float32)
    elif n_full > 0:
        assert scales_full is not None
        scales = scales_full.reshape(-1).to(torch.float32)
    elif tail > 0:
        assert scales_tail is not None
        scales = scales_tail.to(torch.float32)
    else:
        scales = torch.empty(0, dtype=torch.float32, device=w.device)

    # device-preserving: codes_row is on wf.device (same as w.device)
    codes = codes_row.reshape(orig_shape).to(torch.int8).to(w.device)
    # scales should also be on w.device
    scales = scales.to(w.device)

    return GroupwiseResult(
        codes=codes,
        scales=scales,
        shape=orig_shape,
        orig_dtype=str(w.dtype),
        group_size=group_size,
        grouping_scheme=grouping_scheme,
    )


def dequantize_groupwise(res: GroupwiseResult) -> torch.Tensor:  # type: ignore
    """Group-wise dequantize: each group codes * scale (vectorized)."""
    _require_torch()
    if res.scales.numel() == 0:
        return torch.zeros(res.shape, dtype=torch.float32, device=res.codes.device)

    last_dim = res.shape[-1] if len(res.shape) > 0 else 1
    rows = 1
    for d in res.shape[:-1]:
        rows *= d
    if rows == 0:
        return torch.zeros(res.shape, dtype=torch.float32, device=res.codes.device)

    codes_flat = res.codes.reshape(rows, last_dim).to(torch.float32)
    n_full = last_dim // res.group_size
    tail = last_dim % res.group_size

    recon = torch.empty_like(codes_flat, dtype=torch.float32)
    # scales are per-row interleaved: [row0_g0, row0_g1, ..., row0_tail, row1_g0, ...]
    if tail > 0:
        per_row = n_full + 1
        scales_2d = res.scales.reshape(rows, per_row)  # [R, per_row]
        if n_full > 0:
            scales_full = scales_2d[:, :n_full]  # [R, n_full]
            scales_expanded = (
                scales_full.unsqueeze(-1).expand(-1, -1, res.group_size).reshape(rows, n_full * res.group_size)
            )
            recon[:, : n_full * res.group_size] = codes_flat[:, : n_full * res.group_size] * scales_expanded
        tail_scales = scales_2d[:, -1]  # [R]
        recon[:, n_full * res.group_size :] = codes_flat[:, n_full * res.group_size :] * tail_scales.unsqueeze(-1)
    else:
        if n_full > 0:
            scales_full = res.scales.reshape(rows, n_full)
            scales_expanded = (
                scales_full.unsqueeze(-1).expand(-1, -1, res.group_size).reshape(rows, n_full * res.group_size)
            )
            recon[:, : n_full * res.group_size] = codes_flat[:, : n_full * res.group_size] * scales_expanded

    return recon.reshape(res.shape).to(torch.float32).to(res.codes.device)
