"""Read and validate artifact tensors without instantiating an architecture."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openternary.services.artifacts import member_path, validate_artifact
from openternary.services.errors import CapabilityError


def _packed_tensors(root: Path) -> Iterator[tuple[str, Any]]:
    from safetensors.torch import load_file

    from openternary.export.packed import decode_tensor

    info = json.loads((root / "packed.json").read_text(encoding="utf-8"))
    if (info.get("schema_version"), info.get("packing"), info.get("codebook")) != (1, "2bit-v1", [-1, 0, 1]):
        raise ValueError("unsupported packed tensor schema")
    if info.get("bit_layout") != "(c0<<6)|(c1<<4)|(c2<<2)|c3" or info.get("padding") != "00":
        raise ValueError("unsupported packed bit layout or padding")
    for row in info["tensors"]:
        payload = load_file(str(member_path(root, row["file"])))
        if row["storage"] == "ternary":
            tensor = decode_tensor(payload["codes"], payload["scales"], row)
        elif row["storage"] == "original":
            tensor = payload["weight"]
        else:
            raise ValueError("unknown packed storage")
        if list(tensor.shape) != row["shape"] or str(tensor.dtype).removeprefix("torch.") != row["dtype"]:
            raise ValueError("packed tensor shape/dtype mismatch")
        yield row["name"], tensor


def validate_tensors(root: Path) -> dict[str, Any]:
    manifest = validate_artifact(root)
    if manifest["format"] not in {"safetensors", "ternary-packed", "torchao"}:
        raise CapabilityError("tensor validation supports safetensors, ternary-packed and torchao")
    import torch

    found: dict[str, dict[str, Any]] = {}

    def inspect(name: str, tensor: Any) -> None:
        if not isinstance(name, str) or not name or name in found:
            raise ValueError("invalid or duplicate tensor name")
        # TorchAO tensor subclasses expose their dense representation for finite checks.
        dense = tensor.dequantize() if hasattr(tensor, "dequantize") else tensor
        if tensor.numel() == 0 or not torch.isfinite(dense).all():
            raise ValueError(f"empty or non-finite tensor: {name}")
        found[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}

    if manifest["format"] == "safetensors":
        from openternary.services.snapshot import SnapshotReader

        with SnapshotReader(root) as reader:
            for name in reader.keys():  # noqa: SIM118 - SnapshotReader is not a mapping
                inspect(name, reader.get_tensor(name))
    else:
        if manifest["format"] == "torchao":
            from openternary.backends.torchao import TorchAOBackend

            rows = TorchAOBackend().load_weights(root)
        else:
            rows = _packed_tensors(root)
        for name, tensor in rows:
            inspect(name, tensor)
    if not found:
        raise ValueError("artifact has no tensors")
    report_file = root / "quantization.json"
    if report_file.is_file():
        entries = json.loads(report_file.read_text(encoding="utf-8")).get("per_tensor")
        if not isinstance(entries, list) or len(entries) != len(found) or {r["name"] for r in entries} != set(found):
            raise ValueError("quantization report and tensor inventory differ")
        for row in entries:
            if "shape" in row and row["shape"] != found[row["name"]]["shape"]:
                raise ValueError("quantization tensor shape mismatch")
            if str(row.get("dtype", "")).removeprefix("torch.") != found[row["name"]]["dtype"]:
                raise ValueError("quantization tensor dtype mismatch")
        targets = manifest.get("provenance", {}).get("targets")
        if targets is not None:
            actual = {row["name"] for row in entries if row.get("quantizable")}
            expected_targets = {row["name"] for row in targets if row.get("quantizable")}
            if expected_targets != actual:
                raise ValueError("manifest targets and quantization inventory differ")
    return {
        "artifact": str(root.resolve()),
        "artifact_validation": "tensor_integrity",
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "tensor_count": len(found),
        "model_reload": "not_run",
        "quality_acceptance": "not_run",
    }
