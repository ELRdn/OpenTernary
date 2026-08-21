"""Tests for raw safetensors header parser (stdlib only)."""

from __future__ import annotations

import json
import pathlib
import struct
import uuid


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Write a minimal safetensors file with given tensors (name -> (dtype, shape))."""
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        import math

        numel = math.prod(shape) if shape else 1
        nbytes = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1}.get(dtype, 2)
        size = numel * nbytes
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header_json = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * offset)


def test_header_parse_basic() -> None:
    import pathlib
    import shutil

    tmp = pathlib.Path.cwd() / f"test_hdr_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        path = tmp / "model.safetensors"
        _write_safetensors(
            path,
            {
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [1536, 1536]),
                "model.embed_tokens.weight": ("BF16", [262144, 1536]),
            },
        )
        from openternary.utils.safetensors_header import parse_safetensors_header, tensor_param_count

        header = parse_safetensors_header(path)
        assert "model.layers.0.self_attn.q_proj.weight" in header
        assert header["model.layers.0.self_attn.q_proj.weight"]["shape"] == [1536, 1536]
        assert header["model.layers.0.self_attn.q_proj.weight"]["dtype"] == "BF16"
        assert tensor_param_count([1536, 1536]) == 2359296
        assert tensor_param_count([]) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_header_parse_validation_begin_gt_end() -> None:
    import pathlib
    import shutil

    tmp = pathlib.Path.cwd() / f"test_hdr2_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        path = tmp / "model.safetensors"
        header = {"a.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [10, 5]}}
        header_json = json.dumps(header).encode("utf-8")
        with path.open("wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 20)
        from openternary.utils.safetensors_header import parse_safetensors_header

        try:
            parse_safetensors_header(path)
            raise AssertionError("should have raised ValueError")
        except ValueError as e:
            assert "begin > end" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_header_parse_offsets_exceed_payload() -> None:
    import pathlib
    import shutil

    tmp = pathlib.Path.cwd() / f"test_hdr3_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        path = tmp / "model.safetensors"
        header = {"a.weight": {"dtype": "BF16", "shape": [10], "data_offsets": [0, 1000]}}
        header_json = json.dumps(header).encode("utf-8")
        with path.open("wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 20)
        from openternary.utils.safetensors_header import parse_safetensors_header

        try:
            parse_safetensors_header(path)
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "exceeds payload" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
