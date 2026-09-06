"""BF16 F.linear operand dtype一致テスト — Phase 4.2-G 実機バグ再現."""
import torch
import torch.nn.functional as F
import pytest


def test_bf16_linear_both_operands_bf16():
    """BF16 pathは input と weight 共に BF16でなければならない."""
    inp = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(8, 8, dtype=torch.bfloat16, requires_grad=True)
    # 正しい呼び出し: 両方 BF16
    y = F.linear(inp, w)
    assert y.dtype == torch.bfloat16 or y.dtype == torch.float32  # some impl may upcast
    # input BF16 / weight FP32 はエラーになるべき
    inp_bf16 = torch.randn(4, 8, dtype=torch.bfloat16)
    w_fp32 = torch.randn(8, 8, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same dtype|expected.*dtype"):
        F.linear(inp_bf16, w_fp32)


def test_fp32_fallback_both_fp32():
    """Fallbackは両方 FP32."""
    inp = torch.randn(4, 8, dtype=torch.float32)
    w = torch.randn(8, 8, dtype=torch.float32)
    y = F.linear(inp, w)
    assert y.dtype == torch.float32


def test_bf16_path_in_runner_uses_same_dtype():
    """runner.py の該当行が両方 BF16かチェック (静的)."""
    import pathlib
    p = pathlib.Path("src/openternary/calibration/runner.py")
    text = p.read_text(encoding="utf-8")
    # 修正後は y_hat = F.linear(inp_dev.to(torch.bfloat16), w_hat.to(torch.bfloat16))
    assert "w_hat.to(torch.bfloat16).to(torch.float32)" not in text, "BF16→FP32 混在バグが残っている"
    assert "F.linear(inp_dev.to(torch.bfloat16), w_hat.to(torch.bfloat16))" in text
    # fallbackは両方 FP32
    assert text.count("F.linear(inp_dev.to(torch.float32), w_hat.to(torch.float32))") >= 2


def test_threshold_gradient_reaches_raw_params():
    """BF16 forwardでも raw_scale/raw_threshold まで勾配が届く."""
    from openternary.quant.threshold import ste_threshold_codes
    # ダミー weight と scale/threshold
    torch.manual_seed(0)
    w = torch.randn(8, 16, requires_grad=False)
    ref_scale = torch.tensor([0.5], dtype=torch.float32)
    # raw_threshold -> effective 0.5
    raw_thr = torch.nn.Parameter(torch.tensor([0.0]))  # softplus(0)=0.693, sigmoid? 実際は inverse
    # 簡易: raw_scale
    raw_scale = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
    # threshold 付近で STE — threshold_ratio は Tensor のまま渡す
    thr_ratio = torch.sigmoid(raw_thr)  # 近似
    codes = ste_threshold_codes(w, ref_scale, thr_ratio)
    scale = torch.nn.functional.softplus(raw_scale) + 1e-6
    w_hat = codes * scale[0]
    inp = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=False)
    # BF16 linear (両方 BF16)
    y_hat = F.linear(inp.to(torch.bfloat16), w_hat.to(torch.bfloat16))
    tout = torch.randn(4, 8, dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(y_hat.to(torch.float32), tout)
    loss.backward()
    assert raw_scale.grad is not None, "raw_scale に勾配が届かない"
    assert not torch.isnan(raw_scale.grad).any()
    assert not torch.isinf(raw_scale.grad).any()
    # raw_thr も勾配が届くか (STE経由)
    raw_thr2 = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))
    thr2 = torch.sigmoid(raw_thr2)
    codes2 = ste_threshold_codes(w, ref_scale, thr2)
    # codes2 は thr2 に依存 (STEで graphあり)
    # get_effective_threshold_ratio 経由でも raw_thr が graph に入る
    from openternary.calibration.optimizer import get_effective_threshold_ratio
    thr_eff = get_effective_threshold_ratio(raw_thr2)
    assert thr_eff.requires_grad
    # NaN/Inf なし
    assert not torch.isnan(thr_eff).any()
    assert not torch.isinf(thr_eff).any()
    # STE の backward で raw_thr2 に勾配が届くか確認 (簡易)
    loss2 = (codes2.to(torch.float32) * 0.1).sum()
    loss2.backward()
    assert raw_thr2.grad is not None
    assert not torch.isnan(raw_thr2.grad).any()
    assert not torch.isinf(raw_thr2.grad).any()


def test_no_nan_inf_after_bf16_forward():
    inp = torch.randn(4, 8, dtype=torch.bfloat16)
    w = torch.randn(8, 8, dtype=torch.bfloat16)
    y = F.linear(inp, w)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    # FP32 fallbackも同様
    y2 = F.linear(inp.to(torch.float32), w.to(torch.float32))
    assert not torch.isnan(y2).any()
    assert not torch.isinf(y2).any()
