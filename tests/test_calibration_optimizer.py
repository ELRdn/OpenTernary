"""Tests for calibration optimizer helpers."""

import pytest

torch = pytest.importorskip("torch")

from openternary.calibration.optimizer import build_scale_params, get_effective_scales, inverse_softplus  # noqa: E402


def test_inverse_softplus_roundtrip() -> None:
    orig = torch.tensor([0.5, 1.0, 2.0, 0.1])
    eps = 1e-6
    raw = inverse_softplus(orig - eps)
    eff = torch.nn.functional.softplus(raw) + eps  # noqa: E402
    assert torch.allclose(eff, orig, atol=1e-6)


def test_build_scale_params_initial() -> None:
    orig = torch.tensor([0.0, 0.5, 1.0])
    zero_mask = orig == 0
    raw, zm = build_scale_params(orig, zero_mask)
    eff = get_effective_scales(raw, zm)
    assert eff[0].item() == pytest.approx(0.0, abs=1e-9)
    assert eff[1].item() == pytest.approx(0.5, abs=1e-6)
    assert eff[2].item() == pytest.approx(1.0, abs=1e-6)
    # only non-zero should be trainable (but Parameter itself is trainable, zero is masked in effective)
    assert raw.requires_grad


def test_zero_group_exact() -> None:
    orig = torch.tensor([0.0, 0.0, 1.0])
    raw, zm = build_scale_params(orig)
    eff = get_effective_scales(raw, zm)
    assert eff[0].item() == 0.0
    assert eff[1].item() == 0.0
    assert eff[2].item() != 0


def test_per_tensor_single() -> None:
    orig = torch.tensor([0.02])
    raw, zm = build_scale_params(orig)
    eff = get_effective_scales(raw, zm)
    assert eff.item() == pytest.approx(0.02, abs=1e-6)


def test_get_effective_scales_clamp() -> None:
    orig = torch.tensor([0.5])
    raw, zm = build_scale_params(orig)
    # after optimizer step, scales should stay >0
    raw.data = torch.tensor([-100.0])  # very negative
    eff = get_effective_scales(raw, zm)
    assert eff.item() > 0
    assert eff.item() == pytest.approx(1e-6, rel=1e-3)  # softplus(-100) ~0 + eps
