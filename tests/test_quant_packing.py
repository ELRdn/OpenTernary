"""Packing tests — 2bit-v1 canonical."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openternary.quant.packing import pack_ternary, unpack_ternary  # noqa: E402


def test_len_div4() -> None:
    codes = torch.tensor([1, 0, -1, 1], dtype=torch.int8)
    packed = pack_ternary(codes)
    assert packed.numel() == 1
    unpacked = unpack_ternary(packed, 4)
    assert torch.equal(unpacked, codes)


def test_len_not_div4() -> None:
    for n in [1, 2, 3, 5, 6, 7, 9]:
        codes = torch.randint(-1, 2, (n,), dtype=torch.int8)
        # ensure only -1,0,1
        codes = torch.clamp(codes, -1, 1)
        packed = pack_ternary(codes)
        assert packed.numel() == (n + 3) // 4
        unpacked = unpack_ternary(packed, n)
        assert torch.equal(unpacked, codes), f"n={n}"


def test_single_element() -> None:
    for v in [-1, 0, 1]:
        codes = torch.tensor([v], dtype=torch.int8)
        packed = pack_ternary(codes)
        assert packed.numel() == 1
        # tail padding must be 00
        bits = int(packed[0].item())
        # last byte low 6 bits should be 00
        assert (bits & 0b00111111) == 0
        assert torch.equal(unpack_ternary(packed, 1), codes)


def test_roundtrip_random() -> None:
    torch.manual_seed(1)
    n = 1000
    codes = torch.randint(-1, 2, (n,), dtype=torch.int8)
    codes = torch.clamp(codes, -1, 1)
    packed = pack_ternary(codes)
    unpacked = unpack_ternary(packed, n)
    assert torch.equal(unpacked, codes)


def test_reserved_11_rejected() -> None:
    # 0b11 を含む packed を作る
    packed = torch.tensor([0b11111111], dtype=torch.uint8)
    try:
        unpack_ternary(packed, 4)
        raise AssertionError("should have raised ValueError for reserved 11")
    except ValueError as e:
        assert "reserved" in str(e).lower() or "0b11" in str(e)


def test_invalid_code_rejected() -> None:
    codes = torch.tensor([2, 0, 1], dtype=torch.int8)
    try:
        pack_ternary(codes)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "invalid" in str(e).lower()


def test_packed_len_ceil() -> None:
    for n in [0, 1, 4, 5, 8, 100]:
        codes = torch.empty(0, dtype=torch.int8) if n == 0 else torch.zeros(n, dtype=torch.int8)  # noqa: SIM108
        packed = pack_ternary(codes)
        expected = ((n + 3) // 4) if n else 0
        assert packed.numel() == expected


def test_bits_match_len() -> None:
    codes = torch.tensor([1, -1, 0, 1, 0], dtype=torch.int8)
    packed = pack_ternary(codes)
    assert packed.numel() * 8 == ((5 + 3) // 4) * 8


def test_tail_padding_zero_strict() -> None:
    # N=1 なら paddingは00、非00ならstrictエラー
    codes = torch.tensor([1], dtype=torch.int8)
    packed = pack_ternary(codes)
    # 正しいpackedは tail 00
    unpack_ternary(packed, 1, strict=True)  # no error
    # paddingを破壊
    bad = packed.clone()
    bad[0] |= 0b00000001  # set low bits
    try:
        unpack_ternary(bad, 1, strict=True)
        raise AssertionError("should have raised for non-zero padding")
    except ValueError as e:
        assert "padding" in str(e).lower()


def test_strict_len_required() -> None:
    codes = torch.tensor([1], dtype=torch.int8)
    packed = pack_ternary(codes)
    assert packed.numel() == 1
    # 余分なbyteを付けたartifactはstrictでreject
    extended = torch.cat([packed, torch.tensor([0], dtype=torch.uint8)])
    try:
        unpack_ternary(extended, 1, strict=True)
        raise AssertionError("should have raised for non-minimal length")
    except ValueError as e:
        assert "strict" in str(e).lower()


def test_empty_storage() -> None:
    codes = torch.empty(0, dtype=torch.int8)
    packed = pack_ternary(codes)
    assert packed.numel() == 0
    unpacked = unpack_ternary(packed, 0)
    assert unpacked.numel() == 0


def test_determinism() -> None:
    torch.manual_seed(7)
    codes = torch.randint(-1, 2, (100,), dtype=torch.int8)
    codes = torch.clamp(codes, -1, 1)
    p1 = pack_ternary(codes)
    p2 = pack_ternary(codes)
    assert torch.equal(p1, p2)


def test_unpack_numel_range() -> None:
    packed = torch.tensor([0], dtype=torch.uint8)
    # numel > capacity
    try:
        unpack_ternary(packed, 5)
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "exceeds" in str(e).lower() or "capacity" in str(e).lower()
