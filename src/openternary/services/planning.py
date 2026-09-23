"""Offline planning and read-only environment diagnosis."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from openternary.adapters.registry import adapter_name, component_inventory, resolve_component, select_adapter
from openternary.backends import list_backends, validate_backend
from openternary.config.loader import load_config
from openternary.config.schema import AppConfig
from openternary.services.passes import plan_passes
from openternary.utils.safetensors_header import parse_safetensors_header


def build_plan(config: AppConfig, *, operation: str = "quantize") -> dict[str, Any]:
    config = AppConfig.model_validate(config.model_dump())
    spec = validate_backend(config)
    passes = plan_passes(config.quantization.passes, spec.name)
    source = Path(config.model.id)
    targets: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    adapter_id: str | None = None
    unknown = ["hardware/runtime execution", "artifact reload", "model quality"]
    if source.is_dir():
        if (source / "model_index.json").is_file():
            components = component_inventory(source)
        if not components or config.model.component:
            selected = resolve_component(source, config)
            adapter = select_adapter(config, selected)
            adapter_id = adapter_name(adapter)
            import dataclasses

            seen: set[str] = set()
            for file in sorted(selected.glob("*.safetensors")):
                for name, metadata in parse_safetensors_header(file).items():
                    if name in seen:
                        raise ValueError(f"duplicate tensor: {name}")
                    seen.add(name)
                    targets.append(dataclasses.asdict(adapter.classify(name, metadata["shape"], metadata["dtype"])))  # type: ignore[arg-type]
            if not targets:
                unknown.append("tensor inventory (no safetensors headers)")
    else:
        unknown.extend(["model adapter", "tensor inventory (source is not a local directory)"])
    target_names = {item["name"] for item in targets if item["quantizable"]}
    for name, precision in config.quantization.mixed_precision.items():
        if targets and name not in target_names:
            raise ValueError(f"mixed_precision key is not a supported target: {name}")
        for item in targets:
            if item["name"] == name and precision == "preserve":
                item.update(quantizable=False, exclude_reason="mixed_precision: preserve")
    if operation == "calibrate":
        if not config.runtime.low_vram:
            raise ValueError("calibration currently requires runtime.low_vram=true")
        if spec.name != "ternary" or config.quantization.passes or config.quantization.mixed_precision:
            raise ValueError("research calibration requires plain ternary configuration")
    return {
        "schema_version": 1,
        "operation": operation,
        "status": "planned",
        "execution": "not_run",
        "artifact_validation": "not_run",
        "quality_acceptance": "not_run",
        "config": config.model_dump(),
        "backend": spec.describe(),
        "adapter": adapter_id,
        "components": components,
        "targets": targets,
        "target_count": sum(item["quantizable"] for item in targets) if targets else None,
        "passes": passes,
        "actual_device": "cpu" if spec.name == "ternary" and operation == "quantize" else None,
        "actual_dtype": "source dtype preserved" if operation == "quantize" else None,
        "unknown": unknown,
        "side_effects": [],
    }


def plan(config_path: Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_plan(load_config(config_path, overrides))


def doctor() -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for name in ("openternary", "torch", "safetensors", "transformers", "diffusers", "torchao", "datasets"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependencies,
        "backends": list_backends(),
        "hardware": "not_probed",
        "model_validation": "not_run",
        "changes": [],
    }
