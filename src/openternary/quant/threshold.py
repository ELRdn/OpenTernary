"""Threshold-based ternary quantization — Phase 4.2.

Phase 4.1 では暗黙の threshold_ratio=0.5 (round) を使用。
Phase 4.2 では reference_scale × threshold_ratio で hard codes を決定し、
clipped STE で threshold_ratio へ gradient を流す。

責務分離:
* reference_scale: frozen, code assignment 用
* threshold_ratio: learnable ∈ (eps, 1-eps)
* reconstruction_scale: learnable, magnitude 用 (optimizer.py 側)

Hard forward:
    u = abs(W) / reference_scale
    hard_gate = (u > threshold_ratio)
    hard_code = sign(W) * hard_gate

Surrogate (clipped STE):
    margin = u - threshold_ratio
    surrogate = clamp(0.5 + margin / (2*ste_width), 0, 1)
    gate_ste = hard_gate.detach() - surrogate.detach() + surrogate
    codes_ste = sign(W) * gate_ste  (forward == hard_code, backward via surrogate)

Per-tensor: scalar ref_scale / thr_ratio broadcast
Per-group : LD-RW per-row interleaved と同様に expand
"""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for threshold quantization. Install with: uv sync --extra ml")


def _expand_per_group(
    per_group: torch.Tensor,  # type: ignore[type-arg]
    shape: tuple[int, ...],
    group_size: int,
    grouping_scheme: str = "last-dim-rowwise-v1",
) -> torch.Tensor:  # type: ignore[type-arg]
    """per-group 1-D tensor を W shape に broadcast.

    LD-RW: rows = prod(shape[:-1]), last_dim = shape[-1]
    per_group は rows * per_row 個、per_row = ceil(last_dim / group_size)
    """
    _require_torch()
    if not shape:
        return per_group.view(1).expand(1).reshape(shape)  # scalar
    last_dim = int(shape[-1])  # type: ignore[arg-type]
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    if rows == 0 or last_dim == 0:
        # empty
        return torch.zeros(shape, dtype=torch.float32, device=per_group.device)
    n_full = last_dim // group_size
    tail = last_dim % group_size
    per_row = n_full + (1 if tail else 0)
    # per_group: [rows * per_row]
    pg_2d = per_group.reshape(rows, per_row)
    out = torch.empty(rows, last_dim, dtype=per_group.dtype, device=per_group.device)
    if n_full > 0:
        # each full group's value repeated group_size times along last_dim
        full_vals = pg_2d[:, :n_full]  # [R, n_full]
        expanded = full_vals.unsqueeze(-1).expand(-1, -1, group_size).reshape(rows, n_full * group_size)
        out[:, : n_full * group_size] = expanded
    if tail > 0:
        tail_vals = pg_2d[:, -1]  # [R]
        out[:, n_full * group_size :] = tail_vals.unsqueeze(-1).expand(-1, tail)
    return out.reshape(shape)


def hard_threshold_codes(
    w: torch.Tensor,  # type: ignore[type-arg]
    reference_scale: torch.Tensor | float,  # type: ignore[type-arg]
    threshold_ratio: torch.Tensor | float,  # type: ignore[type-arg]
    group_size: int | None = None,
    grouping_scheme: str = "last-dim-rowwise-v1",
) -> torch.Tensor:  # type: ignore[type-arg]
    """Hard ternary codes {-1,0,1} を threshold で決定.

    Args:
        w: 重み tensor (floating)
        reference_scale: per_tensor なら float/scalar tensor, per_group なら 1-D tensor
        threshold_ratio: 同上、(eps, 1-eps) の範囲を想定するが clamp は行わない
        group_size: per_group の場合に必要。None なら per_tensor として扱う

    Returns:
        codes: int8 tensor, same shape as w, values ∈ {-1,0,1}
    """
    _require_torch()
    if w.numel() == 0:
        raise ValueError("cannot quantize empty tensor")
    if not w.is_floating_point():
        raise ValueError(f"hard_threshold_codes requires floating tensor, got {w.dtype}")
    if not torch.isfinite(w).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf in W")
    device = w.device
    # threshold abs
    if group_size is None:
        # per-tensor
        if isinstance(reference_scale, float):
            ref = float(reference_scale)
        else:
            ref = float(reference_scale.item() if hasattr(reference_scale, "item") else reference_scale)  # type: ignore[union-attr]
        if isinstance(threshold_ratio, float):
            thr = float(threshold_ratio)
        else:
            thr = float(threshold_ratio.item() if hasattr(threshold_ratio, "item") else threshold_ratio)  # type: ignore[union-attr]
        if ref == 0:
            return torch.zeros_like(w, dtype=torch.int8)
        thr_abs = ref * thr
        # hard gate
        hard_gate = w.abs() > thr_abs
        sign = torch.sign(w).to(torch.int8)
        # sign 0 -> 0, multiply
        codes = torch.where(hard_gate, sign, torch.zeros_like(sign))
        return codes
    else:
        # per-group: reference_scale and threshold_ratio are 1-D tensors [num_groups]
        if isinstance(reference_scale, float):
            raise ValueError("per-group reference_scale must be tensor")
        if isinstance(threshold_ratio, float):
            raise ValueError("per-group threshold_ratio must be tensor")
        ref_t = reference_scale.to(device)  # type: ignore[union-attr]
        thr_t = threshold_ratio.to(device)  # type: ignore[union-attr]
        # expand to shape
        ref_exp = _expand_per_group(ref_t, tuple(w.shape), group_size, grouping_scheme)
        thr_ratio_exp = _expand_per_group(thr_t, tuple(w.shape), group_size, grouping_scheme)
        thr_abs = ref_exp * thr_ratio_exp  # type: ignore[assignment]
        # zero ref groups -> always zero code (avoid div by zero)
        # but thr_abs will be 0 for those groups, we mask
        # For ref==0 groups, we want codes 0 regardless of W
        # Detect via ref_exp ==0 mask
        zero_ref_mask = ref_exp == 0
        hard_gate = (w.abs() > thr_abs) & (~zero_ref_mask)
        sign = torch.sign(w).to(torch.int8)
        codes = torch.where(hard_gate, sign, torch.zeros_like(sign))
        return codes


class _ClippedSTE(torch.autograd.Function):  # type: ignore[misc]
    """Clipped STE autograd."""

    @staticmethod
    def forward(ctx, hard_gate: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(surrogate)  # not needed but keep
        return hard_gate

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[no-untyped-def]
        # grad to hard_gate is ignored, grad to surrogate is grad_output * surrogate grad
        # surrogate = clamp(0.5 + margin/(2*width),0,1)
        # d surrogate / d threshold_ratio = -1/(2*width) inside linear region, 0 outside
        # But autograd already computes dL/d surrogate via chain, we just need to pass grad_output to surrogate
        # Since gate_ste = hard - surr.detach() + surr, d gate_ste / d surr =1
        # So dL/d surr = grad_output
        # Return (None for hard_gate, grad_output for surrogate)
        return None, grad_output  # type: ignore[return-value]


def ste_threshold_codes(
    w: torch.Tensor,  # type: ignore[type-arg]
    reference_scale: torch.Tensor | float,  # type: ignore[type-arg]
    threshold_ratio: torch.Tensor,  # type: ignore[type-arg]  # must be tensor with requires_grad
    group_size: int | None = None,
    grouping_scheme: str = "last-dim-rowwise-v1",
    ste_width: float = 0.1,
) -> torch.Tensor:  # type: ignore[type-arg]
    """STE で threshold_ratio へ gradient を流す ternary codes.

    Forward は hard_threshold_codes と一致。
    Backward は clipped surrogate 経由で threshold_ratio に勾配が流れる。

    Args:
        w: frozen weight (no grad needed, but may have grad)
        reference_scale: frozen reference scale
        threshold_ratio: learnable ratio tensor (requires_grad True) or scalar tensor
        group_size: per_group の場合指定
        ste_width: clipped STE の width (default 0.1)

    Returns:
        codes_ste: float tensor? 実際は int8 相当だが STE のため float で返す。
                   値は hard codes と同じ {-1,0,1} を float で持つ。
                   gradient は threshold_ratio に流れる。
                   使用時は reconstruction_scale と掛ける前に float へ。
    """
    _require_torch()
    if w.numel() == 0:
        raise ValueError("empty tensor")
    if ste_width <= 0:
        raise ValueError(f"ste_width must be >0, got {ste_width}")
    if not isinstance(threshold_ratio, torch.Tensor):  # type: ignore[union-attr]
        raise ValueError("threshold_ratio must be Tensor for STE")
    device = w.device
    # Compute per-element threshold and normalized magnitude
    if group_size is None:
        # per-tensor scalar
        if isinstance(reference_scale, float):
            ref_val = float(reference_scale)
            ref_tensor = torch.tensor(ref_val, dtype=torch.float32, device=device)
        elif isinstance(reference_scale, torch.Tensor):  # type: ignore[union-attr]
            ref_tensor = reference_scale.to(device).float()  # type: ignore[union-attr]
            if ref_tensor.numel() != 1:
                raise ValueError("per-tensor reference_scale must be scalar")
            ref_val = float(ref_tensor.item())
        else:
            raise ValueError("invalid reference_scale")
        thr_ratio_val = threshold_ratio.to(device).float()  # type: ignore[union-attr]
        if thr_ratio_val.numel() != 1:
            raise ValueError("per-tensor threshold_ratio must be scalar")
        # expand
        if ref_val == 0:
            # all zeros: no gradient needed, return zeros with STE connection that is zero grad
            # need to keep graph: return zeros that still depend on thr? But if ref==0, codes always 0 so no grad is correct
            hard_gate = torch.zeros_like(w, dtype=torch.float32)
            surrogate = torch.zeros_like(w, dtype=torch.float32)
            gate_ste = _ClippedSTE.apply(hard_gate, surrogate)
            sign = torch.sign(w).float()
            return sign * gate_ste  # type: ignore[no-any-return]
        # u = |W| / ref
        u = w.abs().float() / ref_val  # type: ignore[union-attr]
        hard_gate = (u > thr_ratio_val.item()).float()  # bool to float for STE
        # surrogate
        margin = u - thr_ratio_val.view(1).expand_as(u)  # broadcast
        surrogate = torch.clamp(0.5 + margin / (2 * ste_width), 0, 1)
        gate_ste = _ClippedSTE.apply(hard_gate, surrogate)
        sign = torch.sign(w).float()
        codes_ste = sign * gate_ste
        return codes_ste  # type: ignore[no-any-return]
    else:
        # per-group
        if isinstance(reference_scale, float):
            raise ValueError("per-group reference_scale must be tensor")
        ref_t = reference_scale.to(device).float()  # type: ignore[union-attr]
        thr_t = threshold_ratio.to(device).float()  # type: ignore[union-attr]
        # expand per-group to per-element
        ref_exp = _expand_per_group(ref_t, tuple(w.shape), group_size, grouping_scheme)  # type: ignore[arg-type]
        thr_exp = _expand_per_group(thr_t, tuple(w.shape), group_size, grouping_scheme)  # type: ignore[arg-type]
        # zero ref mask: gate always 0
        zero_mask = ref_exp == 0
        # u = |W| / ref  (where ref==0, u is set to 0 to avoid inf, gate will be 0)
        # avoid div by zero: use where
        safe_ref = torch.where(zero_mask, torch.ones_like(ref_exp), ref_exp)
        u = w.abs().float() / safe_ref
        hard_gate_f = (w.abs().float() > (ref_exp * thr_exp)).float()
        hard_gate_f = torch.where(zero_mask, torch.zeros_like(hard_gate_f), hard_gate_f)
        margin = u - thr_exp
        # For zero ref groups, margin should be large negative so surrogate 0 and no grad? But thr doesn't affect those groups, so we mask surrogate to 0
        surrogate = torch.clamp(0.5 + margin / (2 * ste_width), 0, 1)
        surrogate = torch.where(zero_mask, torch.zeros_like(surrogate), surrogate)
        gate_ste = _ClippedSTE.apply(hard_gate_f, surrogate)
        sign = torch.sign(w).float()
        codes_ste = sign * gate_ste
        return codes_ste  # type: ignore[no-any-return]


__all__ = ["hard_threshold_codes", "ste_threshold_codes", "_expand_per_group"]
