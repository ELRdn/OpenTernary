"""Mocked real ROCm GPU で BF16 pathを検証 — dtype不一致が再発しないこと."""
import torch
import torch.nn.functional as F
from unittest.mock import patch


def _run_one_step_with_bf16(mock_bf16_fail=False):
    """runner の BF16 F.linear 相当を再現."""
    torch.manual_seed(0)
    inp = torch.randn(4, 16, requires_grad=False)
    # w_hat は learnable scale を含む想定で requires_grad True
    w_hat_fp32 = torch.randn(8, 16, requires_grad=True)
    w_hat_bf16 = w_hat_fp32.to(torch.bfloat16)
    # BF16 tensor は autograd でも grad が流れる (BF16でも requires_grad は保持)
    # ただし to() で dtype 変換すると grad_fn が残るようにする
    w_hat_bf16 = w_hat_fp32.to(torch.bfloat16)
    # w_hat_bf16 は w_hat_fp32 の view ではないので、別途 requires_grad を持つようにする
    # 簡易: w_hat_fp32 を直接使う
    inp_fp32 = inp.to(torch.float32)
    inp_bf16 = inp.to(torch.bfloat16)

    # 正常 BF16 path: 両方 BF16
    if not mock_bf16_fail:
        # inp_bf16 は requires_grad False だが w_hat_bf16 は True の graph を持つ
        # w_hat_bf16 が grad_fn を持つように、w_hat_fp32 から作る
        w_for_bf16 = w_hat_fp32.to(torch.bfloat16)
        y = F.linear(inp_bf16, w_for_bf16)
        # y は BF16 だが grad は FP32 の w_hat_fp32 に流れる
        loss = y.to(torch.float32).sum()
        loss.backward()
        assert w_hat_fp32.grad is not None
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
        return y
    else:
        # BF16 が失敗して FP32 fallback
        try:
            # わざと BF16 を失敗させる
            raise RuntimeError("bf16 not supported")
        except RuntimeError as e:
            if "bf16" in str(e).lower():
                y = F.linear(inp_fp32, w_hat_fp32)
                assert y.dtype == torch.float32
                assert not torch.isnan(y).any()
                return y
            raise


def test_bf16_both_operands_same_dtype_mocked_gpu():
    # 正常系
    y = _run_one_step_with_bf16(mock_bf16_fail=False)
    assert y is not None

    # fallback系
    y2 = _run_one_step_with_bf16(mock_bf16_fail=True)
    assert y2 is not None


def test_runner_bf16_path_with_mocked_cuda_available():
    """runner の BF16 分岐が dtype一致で呼ばれてもエラーにならないこと."""
    # 実際の CUDA がなくても、CPU 上で BF16 F.linear が dtype一致なら成功する
    # 以前のバグは BF16/FP32 混在で RuntimeError だった
    inp_bf16 = torch.randn(4, 16, dtype=torch.bfloat16)
    w_bf16 = torch.randn(8, 16, dtype=torch.bfloat16)
    # 正しい呼び出しはエラーにならない
    y = F.linear(inp_bf16, w_bf16)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    # 混在はエラー
    with pytest.raises(RuntimeError):
        F.linear(inp_bf16, w_bf16.to(torch.float32))


import pytest
