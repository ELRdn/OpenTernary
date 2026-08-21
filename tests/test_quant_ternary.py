"""Ternary quantizer tests — Phase 2 canonical AbsMean baseline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openternary.quant.ternary import dequantize, quantize_absmean  # noqa: E402


def test_all_zero() -> None:
    w = torch.zeros(4, 4)
    tt = quantize_absmean(w)
    assert tt.scale == 0.0
    assert tt.codes.dtype == torch.int8
    assert (tt.codes == 0).all().item()
    assert tt.shape == (4, 4)
    recon = dequantize(tt)
    assert torch.allclose(recon, torch.zeros(4, 4))


def test_positive_only() -> None:
    w = torch.tensor([[1.0, 2.0, 3.0]])
    tt = quantize_absmean(w)
    assert tt.scale > 0
    assert set(tt.codes.unique().tolist()).issubset({-1, 0, 1})
    # positive values → codes >=0
    assert (tt.codes >= 0).all().item()


def test_negative_only() -> None:
    w = torch.tensor([[-1.0, -2.0, -3.0]])
    tt = quantize_absmean(w)
    assert set(tt.codes.unique().tolist()).issubset({-1, 0, 1})
    assert (tt.codes <= 0).all().item()


def test_mixed_signs() -> None:
    w = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    tt = quantize_absmean(w)
    assert set(tt.codes.unique().tolist()).issubset({-1, 0, 1})
    assert tt.shape == (5,)


def test_small_values_around_threshold() -> None:
    # scale固定ベクトル: abs mean = 1.0 のため境界が ±0.5
    w_below = torch.tensor([0.49, 1.51])
    w_exact = torch.tensor([0.50, 1.50])
    w_above = torch.tensor([0.51, 1.49])
    # すべてscale=1.0
    for w in [w_below, w_exact, w_above]:
        tt = quantize_absmean(w)
        assert abs(tt.scale - 1.0) < 1e-6

    tt_below = quantize_absmean(w_below)
    # 0.49 → 0 (below threshold)
    assert int(tt_below.codes[0].item()) == 0
    tt_exact = quantize_absmean(w_exact)
    # 0.50 → 0 (ties-to-even: round(0.5)=0)
    assert int(tt_exact.codes[0].item()) == 0
    tt_above = quantize_absmean(w_above)
    # 0.51 → 1
    assert int(tt_above.codes[0].item()) == 1

    # 負側
    w_neg_below = torch.tensor([-0.49, -1.51])
    w_neg_exact = torch.tensor([-0.50, -1.50])
    w_neg_above = torch.tensor([-0.51, -1.49])
    assert int(quantize_absmean(w_neg_below).codes[0].item()) == 0
    assert int(quantize_absmean(w_neg_exact).codes[0].item()) == 0
    assert int(quantize_absmean(w_neg_above).codes[0].item()) == -1


def test_random_deterministic() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    t1 = quantize_absmean(w)
    t2 = quantize_absmean(w)
    assert torch.equal(t1.codes, t2.codes)
    assert t1.scale == t2.scale


def test_single_element() -> None:
    w = torch.tensor([3.14])
    tt = quantize_absmean(w)
    assert tt.codes.numel() == 1
    assert tt.shape == (1,)
    assert int(tt.codes[0].item()) in (-1, 0, 1)


def test_q_values_in_set() -> None:
    torch.manual_seed(42)
    w = torch.randn(100)
    tt = quantize_absmean(w)
    vals = set(tt.codes.unique().tolist())
    assert vals.issubset({-1, 0, 1})


def test_shape_preserved() -> None:
    w = torch.randn(2, 3, 4)
    tt = quantize_absmean(w)
    assert tt.shape == (2, 3, 4)
    assert tt.codes.shape == torch.Size([2, 3, 4])
    recon = dequantize(tt)
    assert recon.shape == torch.Size([2, 3, 4])


def test_scale_fp32_accumulation() -> None:
    # BF16入力でもFP32 accumulationで同じscaleになることを確認
    w_bf16 = torch.randn(32, 32, dtype=torch.bfloat16)
    w_fp32 = w_bf16.float()
    tt_bf16 = quantize_absmean(w_bf16)
    # 直接float32で計算したscaleと一致
    expected_scale = float(w_fp32.abs().mean().item())
    assert abs(tt_bf16.scale - expected_scale) < 1e-6


def test_zero_branch_no_eps() -> None:
    # 極小scaleでも epsで歪まない
    w = torch.tensor([1e-13, -1e-13, 1e-13])
    tt = quantize_absmean(w)
    # scale = mean(abs) ≈ 1e-13, ゼロではないので正しく三値化
    assert tt.scale != 0.0
    # ゼロtensorはscale 0で分岐
    w_zero = torch.zeros(3)
    tt_zero = quantize_absmean(w_zero)
    assert tt_zero.scale == 0.0
    assert (tt_zero.codes == 0).all().item()


def test_empty_rejected() -> None:
    w = torch.empty(0)
    try:
        quantize_absmean(w)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_int_rejected() -> None:
    w = torch.tensor([1, 2, 3], dtype=torch.int32)
    try:
        quantize_absmean(w)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "floating" in str(e).lower()


def test_nan_rejected() -> None:
    w = torch.tensor([1.0, float("nan")])
    try:
        quantize_absmean(w)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "nan" in str(e).lower() or "inf" in str(e).lower()


def test_inf_rejected() -> None:
    w = torch.tensor([1.0, float("inf")])
    try:
        quantize_absmean(w)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "inf" in str(e).lower() or "nan" in str(e).lower()


def test_device_preserving() -> None:
    w = torch.randn(4, 4)
    tt = quantize_absmean(w)
    assert tt.codes.device == w.device
    recon = dequantize(tt)
    assert recon.device == tt.codes.device
    if torch.cuda.is_available():
        w_cuda = torch.randn(4, 4, device="cuda")
        tt_cuda = quantize_absmean(w_cuda)
        assert str(tt_cuda.codes.device).startswith("cuda")


def test_determinism() -> None:
    torch.manual_seed(123)
    w = torch.randn(16, 16)
    a = quantize_absmean(w)
    b = quantize_absmean(w)
    assert torch.equal(a.codes, b.codes)
    assert a.scale == b.scale
    assert torch.equal(dequantize(a), dequantize(b))


def test_dequant_scale_zero() -> None:
    w = torch.zeros(3, 3)
    tt = quantize_absmean(w)
    recon = dequantize(tt)
    assert torch.equal(recon, torch.zeros(3, 3))
