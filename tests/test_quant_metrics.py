"""Metrics tests — cosine null, ratio split."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openternary.quant.metrics import tensor_metrics  # noqa: E402
from openternary.quant.packing import pack_ternary  # noqa: E402
from openternary.quant.ternary import dequantize, quantize_absmean  # noqa: E402


def test_mae_mse_finite() -> None:
    torch.manual_seed(0)
    w = torch.randn(32)
    tt = quantize_absmean(w)
    recon = dequantize(tt)
    packed = pack_ternary(tt.codes)
    m = tensor_metrics(w, recon, tt.codes, packed, tt.scale)
    assert m["mae"] >= 0
    assert m["mse"] >= 0
    assert m["rmse"] >= 0
    assert m["max_abs_error"] >= 0
    assert m["mae"] != float("inf")
    assert m["mse"] != float("inf")


def test_cosine_null_on_zero() -> None:
    w = torch.zeros(5)
    tt = quantize_absmean(w)
    recon = dequantize(tt)
    packed = pack_ternary(tt.codes)
    m = tensor_metrics(w, recon, tt.codes, packed, tt.scale)
    assert m["cosine_similarity"] is None


def test_cosine_nonzero() -> None:
    w = torch.tensor([1.0, 2.0, 3.0])
    tt = quantize_absmean(w)
    recon = dequantize(tt)
    packed = pack_ternary(tt.codes)
    m = tensor_metrics(w, recon, tt.codes, packed, tt.scale)
    assert m["cosine_similarity"] is not None
    assert -1.0 <= float(m["cosine_similarity"]) <= 1.0  # type: ignore[arg-type]


def test_zero_ratio() -> None:
    w = torch.zeros(10)
    tt = quantize_absmean(w)
    recon = dequantize(tt)
    packed = pack_ternary(tt.codes)
    m = tensor_metrics(w, recon, tt.codes, packed, tt.scale)
    assert m["zero_ratio"] == pytest.approx(1.0)
    assert m["negative_ratio"] == pytest.approx(0.0)
    assert m["positive_ratio"] == pytest.approx(0.0)


def test_compression_split() -> None:
    w = torch.randn(100)
    tt = quantize_absmean(w)
    recon = dequantize(tt)
    packed = pack_ternary(tt.codes)
    m = tensor_metrics(w, recon, tt.codes, packed, tt.scale)
    # original 100*2=200 bytes (BF16仮定), packed 25 bytes
    assert m["original_bytes"] == 200
    assert m["packed_weight_bytes"] == 25
    assert m["scale_bytes"] == 4
    assert m["weight_only_compression_ratio"] == pytest.approx(200 / 25)
    assert m["effective_compression_ratio"] == pytest.approx(200 / 29)
    assert m["effective_includes_scale"] is True


def test_byte_padding() -> None:
    w = torch.randn(5)
    tt = quantize_absmean(w)
    recon = dequantize(tt)
    packed = pack_ternary(tt.codes)
    m = tensor_metrics(w, recon, tt.codes, packed, tt.scale)
    assert m["logical_2bit_bits"] == 10
    assert m["packed_weight_bits"] == 16
    assert m["byte_padding_bits"] == 6
