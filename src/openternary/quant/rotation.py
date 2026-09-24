"""Orthogonal block Hadamard transform for bounded rotation research."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import torch


def hadamard_last_dim(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply a normalized Hadamard matrix to every block on the final axis."""
    if block_size < 2 or block_size & (block_size - 1) or x.shape[-1] % block_size:
        raise ValueError("block_size must be a power of two dividing the input dimension")
    original_shape = x.shape
    y = x.reshape(-1, block_size)
    width = 1
    while width < block_size:
        pairs = y.reshape(-1, block_size // (2 * width), 2, width)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        y = torch.stack((left + right, left - right), dim=2).reshape(-1, block_size)
        width *= 2
    return (y / math.sqrt(block_size)).reshape(original_shape)


def cayley_orthogonal(skew: torch.Tensor) -> torch.Tensor:
    """Map a finite skew-symmetric matrix to an orthogonal matrix."""
    if skew.ndim != 2 or skew.shape[0] != skew.shape[1]:
        raise ValueError("skew must be square")
    if not torch.isfinite(skew).all():
        raise ValueError("skew must be finite")
    if not torch.allclose(skew, -skew.T, atol=1e-5, rtol=1e-5):
        raise ValueError("skew must be antisymmetric")
    identity = torch.eye(skew.shape[0], dtype=skew.dtype, device=skew.device)
    return cast(torch.Tensor, torch.linalg.solve(identity + skew, identity - skew))


def apply_block_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply the same orthogonal block rotation to the final dimension."""
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("rotation must be square")
    block_size = rotation.shape[0]
    if block_size < 1 or x.shape[-1] % block_size:
        raise ValueError("rotation block size must divide input dimension")
    return (x.reshape(-1, block_size) @ rotation).reshape(x.shape)


def load_rotation_plan(path: Path, source: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Verify a learned-rotation plan and return CPU matrices and portable metadata."""
    from safetensors.torch import load_file

    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("rotation plan requires schema_version=1")
    if Path(plan.get("source", "")).resolve() != source.resolve():
        raise ValueError("rotation plan source does not match BF16 input")
    entries = plan.get("rotations")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("rotation plan requires nonempty rotations")
    matrices: dict[str, torch.Tensor] = {}
    portable: dict[str, object] = {}
    for module_name, entry in sorted(entries.items()):
        if not isinstance(module_name, str) or not isinstance(entry, dict):
            raise ValueError("invalid rotation plan entry")
        if not isinstance(entry.get("file"), str) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"rotation file and sha256 are required: {module_name}")
        raw_path = Path(entry["file"])
        tensor_path = raw_path if raw_path.is_absolute() else path.parent / raw_path
        digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"rotation file hash mismatch: {module_name}")
        report_path = tensor_path.with_suffix(".json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("module") != module_name or Path(report.get("source", "")).resolve() != source.resolve():
            raise ValueError(f"rotation training provenance mismatch: {module_name}")
        if report.get("group_size") != 128 or report.get("block_size") != 128:
            raise ValueError(f"unsupported rotation geometry: {module_name}")
        saved = load_file(str(tensor_path), device="cpu")
        if set(saved) != {"cayley_rotation"}:
            raise ValueError(f"rotation file must contain only cayley_rotation: {module_name}")
        matrix = saved["cayley_rotation"]
        if matrix.shape != (128, 128) or matrix.dtype != torch.float32 or not torch.isfinite(matrix).all():
            raise ValueError(f"invalid rotation matrix: {module_name}")
        error = float((matrix.T @ matrix - torch.eye(128)).abs().max().item())
        if error > 1e-4:
            raise ValueError(f"rotation is not orthogonal: {module_name}")
        matrices[module_name] = matrix
        portable[module_name] = {
            "source_file_sha256": digest,
            "training_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "block_size": 128,
            "group_size": 128,
            "orthogonality_max_abs": error,
        }
    return matrices, {"schema_version": 1, "method": "hadamard-learned-cayley", "rotations": portable}
