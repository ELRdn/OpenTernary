"""Ordered, explicit tensor transformations. Rotation requires a runtime adapter."""

from __future__ import annotations

from typing import Any

from openternary.config.schema import PassConfig
from openternary.services.errors import CapabilityError


def plan_passes(passes: list[PassConfig], backend: str, applied: list[str] | None = None) -> list[dict[str, Any]]:
    seen = set(applied or [])
    result = []
    for item in passes:
        if item.name in seen:
            raise CapabilityError(f"pass already applied or duplicated: {item.name}")
        if item.name not in {"noop", "clip"}:
            from openternary.services.plugins import load_plugin

            description = load_plugin("passes", item.name).plan(item.options, backend)
            if description.get("stage") != "pre_quantization" or description.get("runtime_state") is not None:
                raise CapabilityError("tensor-pass plugins must be stateless pre-quantization transformations")
        if backend != "ternary" and item.name != "noop":
            raise CapabilityError(f"{item.name} is not supported by {backend}")
        seen.add(item.name)
        result.append(
            {
                **item.model_dump(),
                "stage": "pre_quantization",
                "requires_calibration": False,
                "changes_function": item.name != "noop",
                "runtime_state": None,
            }
        )
    return result


def apply_passes(tensor: Any, passes: list[PassConfig]) -> Any:
    for item in passes:
        if item.name == "clip":
            tensor = tensor.clamp(-item.options["max_abs"], item.options["max_abs"])
        elif item.name != "noop":
            from openternary.services.plugins import load_plugin

            transformed = load_plugin("passes", item.name).apply(tensor, item.options)
            if (
                transformed.shape != tensor.shape
                or transformed.dtype != tensor.dtype
                or transformed.device != tensor.device
            ):
                raise ValueError("tensor pass must preserve shape, dtype and device")
            tensor = transformed
    return tensor
