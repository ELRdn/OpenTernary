"""Explicit external llama.cpp bridge. Runtime loading remains a separate acceptance gate."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

from openternary.adapters.registry import read_object
from openternary.services.artifacts import (
    MANIFEST,
    atomic_artifact,
    file_hash,
    snapshot_from_run,
    validate_artifact,
    write_manifest,
)
from openternary.services.errors import CapabilityError
from openternary.services.process import run_process
from openternary.services.snapshot import tensor_inventory


def export_gguf(
    source: Path,
    destination: Path,
    converter: Path,
    *,
    output_dtype: str = "f16",
    timeout_seconds: float = 3600,
    converter_python: Path | None = None,
) -> dict[str, Any]:
    source = snapshot_from_run(source).resolve()
    converter = converter.resolve()
    if converter.name != "convert_hf_to_gguf.py" or not converter.is_file():
        raise ValueError("provide a local llama.cpp convert_hf_to_gguf.py")
    if output_dtype not in {"f16", "bf16", "f32"} or timeout_seconds <= 0:
        raise ValueError("GGUF bridge requires f16/bf16/f32 and a positive timeout")
    raw = read_object(source / "config.json")
    if raw.get("model_type") not in {"llama", "mistral", "qwen2"}:
        raise CapabilityError(
            "GGUF bridge currently admits llama/mistral/qwen2 metadata only; this is not runtime certification"
        )
    tensor_inventory(source)
    prior = validate_artifact(source) if (source / MANIFEST).is_file() else None
    if prior and prior["format"] != "safetensors":
        raise CapabilityError("GGUF bridge requires a safetensors source")
    if destination.resolve().is_relative_to(source):
        raise ValueError("GGUF destination must be outside the source")
    with atomic_artifact(destination) as staged:
        output = staged / "model.gguf"
        python = str(converter_python.resolve()) if converter_python else sys.executable
        command = [python, str(converter), str(source), "--outfile", str(output), "--outtype", output_dtype]
        with (staged / "converter.log").open("w", encoding="utf-8") as log:
            run_process(command, timeout=timeout_seconds, stdout=log)
        with output.open("rb") as stream:
            header = stream.read(24)
        if len(header) != 24:
            raise ValueError("converter did not produce a GGUF header")
        magic, version, tensors, _ = struct.unpack("<4sIQQ", header)
        if magic != b"GGUF" or version not in {2, 3} or not tensors:
            raise ValueError("invalid GGUF magic, version or tensor count")
        manifest = write_manifest(
            staged,
            format="gguf",
            provenance={
                **(prior.get("provenance", {}) if prior else {}),
                "source": str(source),
                "parent_manifest": prior["manifest_fingerprint"] if prior else None,
                "source_files": {
                    p.name: file_hash(p)
                    for p in source.iterdir()
                    if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}
                },
                "converter": str(converter),
                "converter_sha256": file_hash(converter),
                "converter_python": python,
                "output_dtype": output_dtype,
                "derivation": "floating-point conversion; not native ternary encoding",
                "validation_scope": "GGUF header and file hashes only; target runtime not executed",
            },
        )
    return {"artifact": str(destination.resolve()), "manifest": manifest, "artifact_validation": "file_integrity"}
