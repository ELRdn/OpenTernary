"""Model/component selection without importing Transformers or Diffusers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from openternary.adapters.base import ModelAdapter
from openternary.adapters.gemma4 import Gemma4Adapter
from openternary.adapters.generic import DiffusersAdapter, TransformersAdapter
from openternary.config.schema import AppConfig
from openternary.services.errors import CapabilityError


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def component_inventory(source: Path) -> list[dict[str, Any]]:
    index = read_object(source / "model_index.json")
    result = []
    for name, value in sorted(index.items()):
        if name.startswith("_") or not isinstance(value, list) or len(value) != 2 or value[0] is None:
            continue
        if Path(name).name != name or name in {".", ".."} or "\\" in name or ":" in name:
            raise ValueError(f"invalid pipeline component: {name}")
        folder = source / name
        role = (
            "denoiser"
            if name in {"transformer", "unet"}
            else "text_encoder"
            if name.startswith("text_encoder")
            else "vae"
            if name == "vae"
            else "auxiliary"
        )
        result.append(
            {
                "name": name,
                "library": value[0],
                "class": value[1],
                "role": role,
                "present": folder.is_dir(),
                "conversion": "unverified" if role in {"denoiser", "text_encoder"} else "unsupported",
            }
        )
    return result


def resolve_component(source: Path, config: AppConfig) -> Path:
    if not (source / "model_index.json").is_file():
        if config.model.component:
            raise CapabilityError("component selection requires a Diffusers model_index.json")
        return source
    components = component_inventory(source)
    selected = config.model.component
    if selected is None:
        raise CapabilityError(
            "Diffusers conversion requires model.component; whole-pipeline conversion is not yet validated"
        )
    match = next((item for item in components if item["name"] == selected), None)
    if match is None or match["conversion"] == "unsupported":
        raise CapabilityError(f"unsupported pipeline component: {selected}")
    path = (source / selected).resolve()
    if not path.is_relative_to(source.resolve()):
        raise ValueError("component path escapes pipeline")
    return path


def select_adapter(config: AppConfig, source: Path | None = None) -> ModelAdapter:
    requested = config.model.adapter
    raw = read_object(source / "config.json") if source and (source / "config.json").is_file() else {}
    model_type = str(raw.get("model_type", "")).lower()
    if requested == "auto":
        if model_type in {"gemma4", "gemma4_text"} or (not raw and Gemma4Adapter.supports(config.model.id)):
            requested = "gemma4"
        elif "_class_name" in raw:
            requested = "diffusers"
        elif model_type or raw.get("architectures"):
            requested = "transformers"
        else:
            raise CapabilityError("cannot identify model adapter from config.json; specify model.adapter")
    cls: Any = {"gemma4": Gemma4Adapter, "transformers": TransformersAdapter, "diffusers": DiffusersAdapter}[requested]
    return cast(ModelAdapter, cls(config.quantization.target.attention, config.quantization.target.mlp))


def adapter_name(adapter: ModelAdapter) -> str:
    return "gemma4" if isinstance(adapter, Gemma4Adapter) else str(getattr(adapter, "name", "unknown"))
