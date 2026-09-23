"""Lossless packing of already-ternary snapshots; no model construction or inference."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

from openternary.experiment.metadata import write_json
from openternary.services.artifacts import MANIFEST, member_path
from openternary.services.reporting import progress


def _copy_interface(source: Path, destination: Path) -> None:
    for file in source.iterdir():
        if (
            file.is_file()
            and file.suffix in {".json", ".model", ".txt", ".jinja"}
            and file.name not in {MANIFEST, "packed.json", "quantization.json"}
            and not file.name.endswith(".index.json")
        ):
            shutil.copyfile(file, destination / file.name)


def encode_tensor(tensor: Any, group_size: int, granularity: str) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional

    from openternary.quant.packing import pack_ternary

    if not tensor.is_floating_point() or not tensor.numel() or not torch.isfinite(tensor).all():
        raise ValueError("packed ternary requires non-empty finite floating weights")
    if granularity not in {"per_tensor", "per_group"} or group_size < 1:
        raise ValueError("invalid packed grouping")
    rows = tensor.reshape(1, -1) if granularity == "per_tensor" else tensor.reshape(-1, tensor.shape[-1])
    width = rows.shape[-1]
    group = width if granularity == "per_tensor" else group_size
    padded = functional.pad(rows.abs().float(), (0, (-width) % group))
    scales = padded.reshape(rows.shape[0], -1, group).amax(-1).to(tensor.dtype)
    expanded = scales.repeat_interleave(group, dim=1)[:, :width]
    codes = rows.sign().to(torch.int8)
    rebuilt = (codes.float() * expanded.float()).to(tensor.dtype).reshape(tensor.shape)
    if not torch.equal(rebuilt.view(torch.uint8), tensor.contiguous().view(torch.uint8)):
        raise ValueError("snapshot is not exactly ternary for the recorded groups; refusing lossy repacking")
    return pack_ternary(codes), scales.reshape(-1).contiguous()


def decode_tensor(packed: Any, scales: Any, row: dict[str, Any]) -> Any:
    import torch

    from openternary.quant.packing import unpack_ternary

    shape = row["shape"]
    if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("invalid packed shape")
    dtype = row["dtype"]
    if dtype not in {"float32", "float16", "bfloat16"} or str(scales.dtype) != f"torch.{dtype}":
        raise ValueError("packed scale/original dtype mismatch")
    if (
        row.get("granularity") not in {"per_tensor", "per_group"}
        or type(row.get("group_size")) is not int
        or row["group_size"] <= 0
    ):
        raise ValueError("invalid packed grouping metadata")
    n = math.prod(shape)
    codes = unpack_ternary(packed, n, strict=True)
    width = n if row["granularity"] == "per_tensor" else shape[-1]
    rows = n // width
    group = width if row["granularity"] == "per_tensor" else row["group_size"]
    groups = (width + group - 1) // group
    if scales.numel() != rows * groups or not torch.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("invalid packed scales")
    expanded = scales.reshape(rows, groups).repeat_interleave(group, dim=1)[:, :width]
    return (codes.reshape(rows, width).float() * expanded.float()).to(getattr(torch, dtype)).reshape(shape)


def pack_snapshot(source: Path, destination: Path) -> None:
    from safetensors.torch import save_file

    from openternary.services.snapshot import SnapshotReader

    report = json.loads((source / "quantization.json").read_text(encoding="utf-8"))
    if report.get("codebook") != [-1, 0, 1]:
        raise ValueError("packing requires a ternary quantization report")
    entries = report.get("per_tensor")
    if not isinstance(entries, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("name"), str) for row in entries
    ):
        raise ValueError("quantization report requires per_tensor inventory")
    targets = {row["name"]: row for row in entries}
    if len(targets) != len(entries):
        raise ValueError("duplicate quantization target")
    _copy_interface(source, destination)
    (destination / "tensors").mkdir()
    rows: list[dict[str, Any]] = []
    with SnapshotReader(source) as reader:
        keys = reader.keys()
        if set(keys) != set(targets):
            raise ValueError("quantization report and payload inventory differ")
        total_tensors = len(keys)
        for index, name in enumerate(keys):
            tensor = reader.get_tensor(name)
            entry = targets[name]
            filename = f"tensors/{index:06d}.safetensors"
            row: dict[str, Any] = {
                "name": name,
                "file": filename,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            if type(entry.get("quantizable")) is not bool:
                raise ValueError("quantizable must be boolean")
            if entry["quantizable"]:
                group = int(entry["group_size"])
                granularity = entry["scale_granularity"]
                codes, scales = encode_tensor(tensor, group, granularity)
                save_file({"codes": codes, "scales": scales}, str(destination / filename))
                row.update(
                    storage="ternary", group_size=group, granularity=granularity, scale_dtype=row["dtype"], padding="00"
                )
            else:
                save_file({"weight": tensor.contiguous()}, str(destination / filename))
                row["storage"] = "original"
            rows.append(row)
            completed = index + 1
            if completed == 1 or completed == total_tensors or completed % 25 == 0:
                progress("pack_tensors", completed, total_tensors)
                logging.getLogger("openternary.export").info("pack tensors: %d/%d", completed, total_tensors)
    write_json(
        destination / "packed.json",
        {
            "schema_version": 1,
            "packing": "2bit-v1",
            "codebook": [-1, 0, 1],
            "bit_layout": "(c0<<6)|(c1<<4)|(c2<<2)|c3",
            "mapping": {"00": -1, "01": 0, "10": 1},
            "reserved": "11",
            "padding": "00",
            "grouping": "last-dim-rowwise-v1",
            "tensors": rows,
        },
    )
    shutil.copyfile(source / "quantization.json", destination / "quantization.json")


def unpack_snapshot(source: Path, destination: Path) -> None:
    from safetensors.torch import load_file, save_file

    metadata = json.loads((source / "packed.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("packing") != "2bit-v1"
        or metadata.get("codebook") != [-1, 0, 1]
        or metadata.get("grouping") != "last-dim-rowwise-v1"
    ):
        raise ValueError("unsupported packed format")
    rows = metadata.get("tensors")
    if not isinstance(rows, list) or not rows:
        raise ValueError("packed artifact has no tensors")
    _copy_interface(source, destination)
    weight_map = {}
    total_tensors = len(rows)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or row["name"] in weight_map:
            raise ValueError("invalid or duplicate packed tensor")
        payload = load_file(str(member_path(source, row["file"])))
        if row["storage"] == "ternary":
            tensor = decode_tensor(payload["codes"], payload["scales"], row)
        elif row["storage"] == "original":
            tensor = payload["weight"]
        else:
            raise ValueError("unknown packed tensor storage")
        if list(tensor.shape) != row["shape"] or str(tensor.dtype).removeprefix("torch.") != row["dtype"]:
            raise ValueError("packed tensor metadata mismatch")
        filename = f"model-{index + 1:05d}-of-{len(rows):05d}.safetensors"
        save_file({row["name"]: tensor}, str(destination / filename))
        weight_map[row["name"]] = filename
        completed = index + 1
        if completed == 1 or completed == total_tensors or completed % 25 == 0:
            progress("unpack_tensors", completed, total_tensors)
            logging.getLogger("openternary.export").info("unpack tensors: %d/%d", completed, total_tensors)
    write_json(destination / "model.safetensors.index.json", {"metadata": {}, "weight_map": weight_map})
    if (source / "quantization.json").is_file():
        shutil.copyfile(source / "quantization.json", destination / "quantization.json")
