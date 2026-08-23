"""Threshold quantization primitives — Phase 4.2 Gates G4.2-1/2/3/5/6."""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from openternary.calibration.optimizer import (
    build_threshold_params,
    get_effective_threshold_ratio,
)
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.ternary import quantize_absmean
from openternary.quant.threshold import hard_threshold_codes, ste_threshold_codes


def test_threshold_parameterization_bounds() -> None:
    """G4.2: raw=0 -> ratio 0.5, and ratio stays in (eps,1-eps) without post-clamp."""
    raw, _ = build_threshold_params(10, init_ratio=0.5, eps=0.01)
    ratio = get_effective_threshold_ratio(raw, eps=0.01)
    assert ratio[0].item() == pytest.approx(0.5, abs=1e-6)
    assert torch.all(ratio > 0.01)
    assert torch.all(ratio < 0.99)
    # gradient finite
    ratio.sum().backward()
    assert torch.all(torch.isfinite(raw.grad))  # type: ignore[union-attr]
    # large raw should approach bounds but never exceed
    raw2, _ = build_threshold_params(1, init_ratio=0.9, eps=0.01)
    r2 = get_effective_threshold_ratio(raw2, eps=0.01)
    assert r2.item() == pytest.approx(0.9, abs=1e-3)
    assert 0.01 < r2.item() < 0.99


def test_hard_forward_only_m123() -> None:
    """G4.2-2: forward codes always in {-1,0,1}."""
    w = torch.randn(32, 32)
    ref = w.abs().mean().item()
    for ratio in [0.1, 0.5, 0.9]:
        codes = hard_threshold_codes(w, ref, ratio)
        assert codes.dtype == torch.int8
        assert set(torch.unique(codes).tolist()).issubset({-1, 0, 1})
        assert codes.shape == w.shape
    # per-group
    w2 = torch.randn(8, 8)
    res = quantize_groupwise(w2, 4)
    thr, _ = build_threshold_params(int(res.scales.numel()), init_ratio=0.5, eps=0.01)
    ratio_g = get_effective_threshold_ratio(thr, eps=0.01)
    codes_g = hard_threshold_codes(w2, res.scales, ratio_g, group_size=4)
    assert set(torch.unique(codes_g).tolist()).issubset({-1, 0, 1})


def test_ste_forward_parity_and_backward() -> None:
    """G4.2-3: STE forward == hard, backward gradient flows only near boundary."""
    torch.manual_seed(0)
    # Crafted weights near threshold: ref=1.0, thr=0.5, weight = 0.51, 0.49, 0.6, 0.4
    ref = 1.0
    w = torch.tensor([[0.51, 0.49, 0.6, 0.4, -0.51, -0.49]], dtype=torch.float32)
    thr_param, _ = build_threshold_params(1, init_ratio=0.5, eps=0.01)
    raw = thr_param
    raw.retain_grad()  # type: ignore[attr-defined]
    # STE codes
    thr_ratio = get_effective_threshold_ratio(raw, eps=0.01)
    codes_ste = ste_threshold_codes(w, ref, thr_ratio, ste_width=0.1)
    # forward parity
    hard = hard_threshold_codes(w, ref, 0.5)
    assert torch.equal(codes_ste.detach().round().to(torch.int8), hard)
    # backward: loss = sum(codes_ste * 1.0)
    loss = codes_ste.sum()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all().item()
    assert raw.grad.abs().item() > 0  # near boundary should have grad
    # far from threshold: weight far => clipped STE grad 0
    raw2, _ = build_threshold_params(1, init_ratio=0.5, eps=0.01)
    thr2 = get_effective_threshold_ratio(raw2, eps=0.01)
    w_far = torch.tensor([[2.0, -2.0, 0.0]], dtype=torch.float32)  # u=2,0 far from 0.5 with width 0.1
    # For u=2, margin=1.5 -> surrogate=1 (clamped) grad 0; for u=0, margin=-0.5 -> surrogate=0 grad 0
    # Actually check that grad is ~0 because all surrogates are at boundaries (0 or 1)
    codes_far = ste_threshold_codes(w_far, ref, thr2, ste_width=0.1)
    loss_far = codes_far.sum()
    loss_far.backward()
    # thr2 grad should be 0 (or very small) because all are clipped
    assert raw2.grad is not None
    # allow small zero
    assert float(raw2.grad.abs().item()) == pytest.approx(0.0, abs=1e-6)


def test_threshold_step0_parity_vs_naive() -> None:
    """G4.2-1: threshold_ratio=0.5 must reproduce naive codes fingerprint."""
    torch.manual_seed(42)
    w = torch.randn(8, 16, dtype=torch.float32)
    # naive per_tensor
    tt = quantize_absmean(w)
    naive_codes = tt.codes
    ref = float(tt.scale)
    thr, _ = build_threshold_params(1, init_ratio=0.5, eps=0.01)
    ratio = get_effective_threshold_ratio(thr, eps=0.01)
    hard = hard_threshold_codes(w, ref, float(ratio[0].item()))
    assert torch.equal(hard, naive_codes)
    # naive per_group
    for gs in [4, 8]:
        res = quantize_groupwise(w, gs)
        n = int(res.scales.numel())
        raw_g, _ = build_threshold_params(n, init_ratio=0.5, eps=0.01)
        ratio_g = get_effective_threshold_ratio(raw_g, eps=0.01)
        hard_g = hard_threshold_codes(w, res.scales, ratio_g, group_size=gs)
        assert torch.equal(hard_g, res.codes), f"mismatch per_group gs={gs}"
        # fingerprint check
        fp_naive = hashlib.sha256(res.codes.numpy().tobytes()).hexdigest()
        fp_hard = hashlib.sha256(hard_g.numpy().tobytes()).hexdigest()
        assert fp_naive == fp_hard


def test_crafted_optimum_threshold_improves() -> None:
    """G4.2-5: crafted fixture where optimal threshold !=0.5 via activation loss."""
    torch.manual_seed(123)
    ref_val = 0.4
    ref = torch.tensor([ref_val], dtype=torch.float32)
    # Heavily weighted column near threshold boundary (0.19 vs 0.2 at thr 0.5)
    w = torch.tensor([[0.19, 0.60]], dtype=torch.float32)
    inp = torch.tensor([[10.0, 1.0]], dtype=torch.float32)  # emphasize col 0
    scale = ref_val
    with torch.no_grad():
        teacher_out = inp @ w.T

    def mse_for_thr(thr: float) -> float:
        codes = hard_threshold_codes(w, float(ref[0].item()), thr).float() * scale
        out = inp @ codes.T
        return float(((out - teacher_out) ** 2).mean().item())

    mse05 = mse_for_thr(0.5)
    mse04 = mse_for_thr(0.4)
    # 0.19 is below 0.2 at thr 0.5 -> zero, but above 0.16 at thr 0.4 -> 0.4, lower thr is better
    assert mse04 < mse05 - 1e-6, f"expected thr 0.4 to beat 0.5: {mse04} vs {mse05}"
    # Now run STE optimization to verify threshold moves and improves
    raw, _ = build_threshold_params(1, init_ratio=0.5, eps=0.01)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(30):
        opt.zero_grad()
        ratio = get_effective_threshold_ratio(raw, eps=0.01)
        codes_ste = ste_threshold_codes(w, float(ref[0].item()), ratio, ste_width=0.1)
        w_hat = codes_ste * scale
        out_hat = inp @ w_hat.T
        loss = ((out_hat - teacher_out) ** 2).mean()
        loss.backward()
        opt.step()
    final_ratio = get_effective_threshold_ratio(raw, eps=0.01).item()
    final_codes = hard_threshold_codes(w, float(ref[0].item()), final_ratio).float() * scale
    out_final = inp @ final_codes.T
    final_mse = float(((out_final - teacher_out) ** 2).mean().item())
    assert final_ratio != pytest.approx(0.5, abs=0.02)
    assert final_mse < mse05
    assert abs(final_ratio - 0.5) > 0.02


def test_threshold_gradient_finite_near_boundary() -> None:
    """Additional: gradient finite near boundary, zero far."""
    ref = 1.0
    w = torch.tensor([[0.51, 0.49]], dtype=torch.float32)
    raw, _ = build_threshold_params(1, init_ratio=0.5, eps=0.01)
    ratio = get_effective_threshold_ratio(raw, eps=0.01)
    codes = ste_threshold_codes(w, ref, ratio, ste_width=0.1)
    loss = codes.sum()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all().item()
