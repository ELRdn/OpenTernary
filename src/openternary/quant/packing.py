"""2-bit ternary packing — Phase 2 canonical 2bit-v1.

Mapping:
    00 = -1
    01 =  0
    10 = +1
    11 = reserved / invalid

Byte layout: byte = (c0<<6)|(c1<<4)|(c2<<2)|c3 where c0 is first element.
Tail padding: unused slots MUST be 00.
"""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# code value → 2bit
_TERNARY_TO_BITS: dict[int, int] = {-1: 0b00, 0: 0b01, 1: 0b10}
_BITS_TO_TERNARY: dict[int, int] = {0b00: -1, 0b01: 0, 0b10: 1}
_RESERVED: int = 0b11


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for packing. Install with: uv sync --extra ml")


def pack_ternary(codes: torch.Tensor) -> torch.Tensor:  # type: ignore[type-arg]
    """Ternary codesを2bit packingしてuint8 bytesへ.

    Args:
        codes: int8 tensor, values ∈ {-1,0,1}, 任意shape（flattenしてpack）

    Returns:
        torch.Tensor dtype=uint8, 1D, CPU上, len==ceil(N/4)

    Raises:
        ValueError: codesが{-1,0,1}以外を含む / int8以外
    """
    _require_torch()

    if codes.dtype != torch.int8:  # type: ignore[union-attr]
        raise ValueError(f"pack_ternary requires int8 codes, got {codes.dtype}")

    flat = codes.reshape(-1).cpu()
    n = flat.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.uint8)

    valid_mask = (flat == -1) | (flat == 0) | (flat == 1)
    if not bool(valid_mask.all().item()):
        invalid = flat[~valid_mask]
        raise ValueError(f"pack_ternary: invalid ternary values {invalid[:5].tolist()} (must be -1,0,+1)")

    # vectorized: -1→0, 0→1, 1→2  == flat+1
    bits = (flat.to(torch.int16) + 1).to(torch.uint8)  # type: ignore[union-attr]
    # pad to multiple of 4 with 00 (0)
    pad_len = (4 - (n % 4)) % 4
    if pad_len:
        bits = torch.cat([bits, torch.zeros(pad_len, dtype=torch.uint8)])

    # reshape to (num_bytes, 4)
    bits4 = bits.view(-1, 4)
    packed = (bits4[:, 0] << 6) | (bits4[:, 1] << 4) | (bits4[:, 2] << 2) | bits4[:, 3]
    return packed.to(torch.uint8)


def unpack_ternary(
    packed: torch.Tensor,  # type: ignore[type-arg]
    numel: int,
    strict: bool = True,
) -> torch.Tensor:  # type: ignore[type-arg]
    """Packed bytesをternary codesへ復元.

    Args:
        packed: uint8 tensor, 1D, CPU想定（GPUでもcpu化してdecode）
        numel: 元の要素数（padding除去に必要）
        strict: Trueなら reserved 11検出/非minimal長/padding非00でエラー

    Returns:
        torch.Tensor dtype=int8, shape=(numel,), CPU

    Raises:
        ValueError: reserved 11 / numel範囲外 / strict違反
    """
    _require_torch()

    if numel < 0:
        raise ValueError(f"numel must be >=0, got {numel}")
    if numel == 0:
        if strict and packed.numel() != 0:
            raise ValueError(f"strict: numel=0 requires empty packed, got len={packed.numel()}")
        return torch.empty(0, dtype=torch.int8)

    if packed.dtype != torch.uint8:  # type: ignore[union-attr]
        raise ValueError(f"unpack_ternary requires uint8 packed, got {packed.dtype}")

    packed_cpu = packed.reshape(-1).cpu()
    expected_len = (numel + 3) // 4

    if numel > packed_cpu.numel() * 4:
        raise ValueError(f"numel {numel} exceeds packed capacity {packed_cpu.numel() * 4}")

    if strict and packed_cpu.numel() != expected_len:
        raise ValueError(f"strict: packed len {packed_cpu.numel()} != ceil({numel}/4)={expected_len}")

    # vectorized decode: expand each byte to 4 2bit values
    # Use torch ops for speed (9M elements → ~2M bytes)
    # Compute bits for all bytes, then truncate to numel
    b0 = (packed_cpu >> 6) & 0b11
    b1 = (packed_cpu >> 4) & 0b11
    b2 = (packed_cpu >> 2) & 0b11
    b3 = packed_cpu & 0b11
    bits_matrix = torch.stack([b0, b1, b2, b3], dim=1)  # (num_bytes, 4)
    bits_flat = bits_matrix.reshape(-1)[:numel]

    # reserved check on used bits
    if (bits_flat == _RESERVED).any().item():
        # find first index for error message
        idx = int((bits_flat == _RESERVED).nonzero(as_tuple=False)[0].item())
        byte_idx = idx // 4
        raise ValueError(f"unpack_ternary: reserved code 0b11 at index {idx} (byte {byte_idx})")

    # tail padding check
    if strict and (numel % 4 != 0):
        # bits beyond numel in last byte must be 00
        last_byte_bits = bits_matrix[expected_len - 1]
        remaining = numel % 4
        for pos in range(remaining, 4):
            if int(last_byte_bits[pos].item()) != 0b00:
                raise ValueError(
                    f"strict: tail padding at byte {expected_len - 1} pos {pos} must be 00, got {int(last_byte_bits[pos].item()):02b}"
                )

    # bits 0→-1, 1→0, 2→1  == bits-1
    codes = (bits_flat.to(torch.int16) - 1).to(torch.int8)
    return codes


def pack_bytes(codes: torch.Tensor) -> bytes:  # type: ignore[type-arg]
    """pack_ternaryのbytes版ヘルパ."""
    packed = pack_ternary(codes)
    return packed.numpy().tobytes()  # type: ignore[union-attr]
