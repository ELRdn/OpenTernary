"""Group-wise LD-RW quantization tests — P0 fixes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openternary.quant.grouping import (  # noqa: E402
    dequantize_groupwise,
    num_groups_for_shape,
    quantize_groupwise,
)


def test_num_groups_last_dim_rowwise() -> None:
    # [2,5] G=2 -> each row: 2 full + tail 1 => 3 per row => total 6
    assert num_groups_for_shape([2, 5], 2) == 6
    assert num_groups_for_shape([2, 5], 5) == 2
    assert num_groups_for_shape([6144, 1536], 128) == 6144 * 12
    # G >= D case: [10,64] G=128 -> 10 groups (not 1)
    assert num_groups_for_shape([10, 64], 128) == 10
    assert num_groups_for_shape([3, 5], 128) == 3
    assert num_groups_for_shape([4, 6], 3) == 4 * 2  # 6/3=2 per row


def test_quantize_groupwise_interleaved_order_and_determinism() -> None:
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]])
    r1 = quantize_groupwise(w, 2)
    r2 = quantize_groupwise(w, 2)
    # deterministic
    assert torch.equal(r1.codes, r2.codes)
    assert torch.equal(r1.scales, r2.scales)
    # interleaved per-row: [1.5,3.5,5, 1.5,3.5,5]
    expected = torch.tensor([1.5, 3.5, 5.0, 1.5, 3.5, 5.0])
    assert torch.allclose(r1.scales, expected)
    # codes ∈ {-1,0,1}
    assert set(r1.codes.unique().tolist()).issubset({-1, 0, 1})


def test_groupwise_zero_branch() -> None:
    w = torch.zeros(2, 4)
    res = quantize_groupwise(w, 2)
    # all scales 0 -> codes 0, recon 0, no NaN
    assert torch.all(res.scales == 0)
    assert torch.all(res.codes == 0)
    recon = dequantize_groupwise(res)
    assert torch.all(recon == 0)
    assert not torch.isnan(recon).any()

    # mixed: one group zero, one non-zero
    w2 = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    res2 = quantize_groupwise(w2, 2)
    # first group scales 0 -> codes 0, second group non-zero
    assert res2.scales[0].item() == 0.0
    assert res2.scales[1].item() != 0.0
    assert res2.codes[0, 0].item() == 0 and res2.codes[0, 1].item() == 0
    assert not torch.isnan(dequantize_groupwise(res2)).any()


def test_groupwise_tail_handling() -> None:
    # D not divisible: [2,5] G=2 -> groups 3 per row, tail 1
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0, 10.0], [1.0, 2.0, 3.0, 4.0, 20.0]])
    res = quantize_groupwise(w, 2)
    assert res.scales.shape[0] == 6
    # tail scales are per-row: last of each row
    # row0 tail mean =10, row1 tail mean=20
    assert res.scales[2].item() == pytest.approx(10.0)
    assert res.scales[5].item() == pytest.approx(20.0)
    recon = dequantize_groupwise(res)
    assert recon.shape == w.shape
    assert not torch.isnan(recon).any()


def test_groupwise_g_ge_d() -> None:
    # G >= D => one group per row
    w = torch.randn(3, 5)
    res = quantize_groupwise(w, 128)
    assert res.scales.shape[0] == 3  # not 1
    assert res.codes.shape == w.shape
    recon = dequantize_groupwise(res)
    assert recon.shape == w.shape


def test_groupwise_vectorized_vs_naive() -> None:
    # Compare vectorized result vs naive per-group loop for correctness
    torch.manual_seed(0)
    w = torch.randn(4, 6)
    res = quantize_groupwise(w, 3)
    # Naive: for each row, for each group compute scale and codes
    last_dim = 6
    g = 3
    rows = 4
    n_full = last_dim // g
    tail = last_dim % g
    naive_codes = torch.empty_like(w, dtype=torch.int8)
    naive_scales = []
    flat = w.reshape(rows, last_dim).float()
    for r in range(rows):
        for gi in range(n_full):
            grp = flat[r, gi * g : (gi + 1) * g]
            scale = grp.abs().mean().item()
            naive_scales.append(scale)
            if scale == 0:
                naive_codes[r, gi * g : (gi + 1) * g] = 0
            else:
                naive_codes[r, gi * g : (gi + 1) * g] = torch.round(grp / scale).clamp(-1, 1).to(torch.int8)
        if tail:
            grp = flat[r, n_full * g :]
            scale = grp.abs().mean().item()
            naive_scales.append(scale)
            if scale == 0:
                naive_codes[r, n_full * g :] = 0
            else:
                naive_codes[r, n_full * g :] = torch.round(grp / scale).clamp(-1, 1).to(torch.int8)
    naive_scales_t = torch.tensor(naive_scales, dtype=torch.float32)
    # Our scales are interleaved per row, naive above also per row interleaved => compare
    assert torch.equal(res.codes, naive_codes)
    assert torch.allclose(res.scales, naive_scales_t)


def test_groupwise_dequantize_roundtrip_no_nan() -> None:
    torch.manual_seed(1)
    w = torch.randn(2, 8)
    res = quantize_groupwise(w, 3)
    recon = dequantize_groupwise(res)
    assert recon.shape == w.shape
    assert not torch.isnan(recon).any()
    assert not torch.isinf(recon).any()
