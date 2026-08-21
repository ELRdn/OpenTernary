"""Safetensors raw header parser — Python標準ライブラリのみ.

Safetensors spec: [8 bytes LE U64: header_len][header_len bytes: JSON][tensor data]
Header JSON: {"tensor.name": {"dtype": "BF16", "shape": [...], "data_offsets": [begin, end]}, "__metadata__": {...}}
data_offsetsは tensor data buffer基準の相対offset (ファイル先頭からの絶対offsetではない).
Header-only inspectではpayloadを読まず、header JSONのみで名前/shape/dtype/param_countを取得する.
"""

from __future__ import annotations

import json
import pathlib
import struct

# 既知dtype→1要素あたりバイト数 (検証用, 未知はスキップ)
_DTYPE_NBYTES: dict[str, int] = {
    "F32": 4,
    "F64": 8,
    "F16": 2,
    "BF16": 2,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}

# header_lenが異常に巨大な場合の上限 (100MB). 実Gemma 4 E2B headerは数MB.
_MAX_HEADER_LEN = 100 * 1024 * 1024


def parse_safetensors_header(
    path: str | pathlib.Path,
) -> dict[str, dict[str, object]]:
    """Safetensors headerをraw parseして {tensor_name: {dtype, shape, data_offsets}} を返す.

    Raises:
        FileNotFoundError: ファイルが存在しない
        ValueError: headerが不正/検証失敗
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Safetensors file not found: {p}")
    file_size = p.stat().st_size
    if file_size < 8:
        raise ValueError(f"File too small to be safetensors (size={file_size}): {p}")

    with p.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Failed to read 8-byte header length from {p}")
        header_len = struct.unpack("<Q", raw_len)[0]

        if header_len == 0:
            raise ValueError(f"Invalid header_len=0: {p}")
        if header_len > _MAX_HEADER_LEN:
            raise ValueError(f"header_len too large ({header_len} > {_MAX_HEADER_LEN}): {p}")
        if header_len > file_size - 8:
            raise ValueError(f"header_len {header_len} exceeds file_size-8 {file_size - 8}: {p}")

        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Failed to read header JSON ({len(header_bytes)} != {header_len}): {p}")
        try:
            header_str = header_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Header is not valid UTF-8: {p}") from e
        try:
            header = json.loads(header_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Header is not valid JSON: {p}") from e
        if not isinstance(header, dict):
            raise ValueError(f"Header JSON must be an object, got {type(header)}: {p}")

        # payloadサイズ (buffer基準の上限)
        payload_size = file_size - 8 - header_len
        result: dict[str, dict[str, object]] = {}
        warnings: list[str] = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(value, dict):
                raise ValueError(f"Tensor entry '{key}' must be an object, got {type(value)}")
            dtype = value.get("dtype")
            shape = value.get("shape")
            offsets = value.get("data_offsets")
            if not isinstance(dtype, str):
                raise ValueError(f"Tensor '{key}' dtype must be str, got {type(dtype)}")
            if not isinstance(shape, list):
                raise ValueError(f"Tensor '{key}' shape must be list, got {type(shape)}")
            if not all(isinstance(x, int) and x >= 0 for x in shape):
                raise ValueError(f"Tensor '{key}' shape elements must be non-negative ints: {shape}")
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) for x in offsets):
                raise ValueError(f"Tensor '{key}' data_offsets must be [int,int], got {offsets}")
            begin, end = offsets
            if begin > end:
                raise ValueError(f"Tensor '{key}' data_offsets begin > end: {offsets}")
            if begin < 0 or end < 0:
                raise ValueError(f"Tensor '{key}' data_offsets must be >=0: {offsets}")
            if end > payload_size:
                raise ValueError(f"Tensor '{key}' data_offsets end {end} exceeds payload_size {payload_size}: {p}")
            # dtypeサイズ検証 (不一致はwarning扱い: QAT等で特殊dtypeの可能性)
            if dtype in _DTYPE_NBYTES:
                nbytes = _DTYPE_NBYTES[dtype]
                # prod(shape) == 0 の場合は0バイトも許容 (稀だが)
                import math

                numel = math.prod(shape) if shape else 1
                expected = numel * nbytes
                actual = end - begin
                if expected != actual:
                    warnings.append(
                        f"Tensor '{key}' dtype {dtype} shape {shape} expects {expected} bytes but offsets span {actual}"
                    )
            result[key] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
        # warningsは現時点では破棄しないが、呼び出し元で利用する場合は返すAPIを別途用意
        # ここでは検証warningは握り潰さないよう、必要ならログに残す
        return result


def tensor_param_count(shape: list[int]) -> int:
    """shapeからパラメータ数を計算. 空shapeはスカラー1."""
    if not shape:
        return 1
    import math

    return math.prod(shape)
