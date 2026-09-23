"""Conversion service shared by CLI, Python API and bounded search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openternary.adapters.registry import resolve_component
from openternary.backends import get_backend, validate_backend
from openternary.config.schema import AppConfig
from openternary.services.artifacts import MANIFEST, atomic_artifact, file_hash, write_manifest
from openternary.services.errors import CapabilityError
from openternary.services.identity import snapshot_identity
from openternary.services.passes import plan_passes
from openternary.services.planning import build_plan


def convert_artifact(source: Path, destination: Path, config: AppConfig) -> Any:
    config = AppConfig.model_validate(config.model_dump())
    spec = validate_backend(config, require_installed=True)
    if spec.name == "ternary" and config.device == "cuda":
        raise CapabilityError("ternary snapshot conversion runs on CPU; choose --device cpu or auto")
    source = source.resolve()
    if destination.resolve() == source or destination.resolve().is_relative_to(source):
        raise ValueError("conversion destination must be outside the source")
    selected = resolve_component(source, config)
    previous = selected / "quantization.json"
    if previous.is_file():
        applied = json.loads(previous.read_text(encoding="utf-8")).get("passes", [])
        plan_passes(config.quantization.passes, spec.name, [row["name"] for row in applied])
    planned_config = config.model_copy(deep=True)
    planned_config.model.id = str(source)
    planned = build_plan(planned_config)
    if planned["target_count"] == 0:
        raise CapabilityError("no supported quantization targets")
    source_files = {
        p.name: file_hash(p)
        for p in selected.iterdir()
        if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"} and p.name != MANIFEST
    }
    with atomic_artifact(destination) as staged:
        report = get_backend(config).convert(selected, staged / "payload", config)
        payload = staged / "payload"
        for member in payload.iterdir():
            member.rename(staged / member.name)
        payload.rmdir()
        if planned["adapter"] == "diffusers" and spec.formats[0] == "safetensors":
            from openternary.adapters.runtime import normalize_diffusion_weights

            normalize_diffusion_weights(staged)
            report.file_hashes = {p.name: f"sha256:{file_hash(p)}" for p in staged.glob("*.safetensors")}
        write_manifest(
            staged,
            format=spec.formats[0],
            provenance={
                "source": str(source),
                "identity": snapshot_identity(source),
                "source_files": source_files,
                "config": config.model_dump(),
                "backend": spec.describe(),
                "adapter": planned["adapter"],
                "component": config.model.component,
                "targets": planned["targets"],
                "passes": planned["passes"],
                "actual_device": "cpu",
                "actual_dtype": "source dtype preserved",
            },
        )
    report.dst_snapshot = destination.resolve()
    return report
