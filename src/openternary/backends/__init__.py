"""Lazy backend registry. Listing never imports an ML runtime."""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass
from typing import Any

from openternary import __version__
from openternary.config.schema import AppConfig
from openternary.services.errors import CapabilityError, DependencyError


@dataclass(frozen=True)
class BackendSpec:
    name: str
    distribution: str | None
    schemes: tuple[str, ...]
    weight_dtypes: tuple[str, ...]
    formats: tuple[str, ...]
    implementation: str
    evidence: str = "model_validation_pending"
    api_version: int = 1

    def describe(self) -> dict[str, Any]:
        version: str | None = __version__
        if self.distribution:
            try:
                version = importlib.metadata.version(self.distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
        return {
            **asdict(self),
            "version": version,
            "installation": "installed" if version else "not_installed",
            "quality": "experimental",
            "native_kernel": "unverified",
            "operations": {
                "convert": "implemented_model_unverified",
                "save": "implemented",
                "tensor_reload": "implemented",
                "model_reload": "unverified",
                "native_low_bit_execution": "unsupported" if self.name == "ternary" else "unverified",
            },
            "required_version": "0.18.0" if self.name == "torchao" else None,
        }


BUILTINS = {
    "ternary": BackendSpec(
        "ternary",
        None,
        ("absmean",),
        ("ternary",),
        ("safetensors", "ternary-packed"),
        "openternary.backends.ternary:TernaryBackend",
    ),
    "torchao": BackendSpec(
        "torchao",
        "torchao",
        ("int8-weight-only",),
        ("int8",),
        ("torchao",),
        "openternary.backends.torchao:TorchAOBackend",
    ),
}


def list_backends() -> list[dict[str, Any]]:
    from openternary.services.plugins import discover_plugins

    return [spec.describe() for spec in BUILTINS.values()] + [
        {**row, "installation": "installed", "capabilities": "not_loaded"}
        for row in discover_plugins()
        if row["kind"] == "backends"
    ]


def get_spec(name: str) -> BackendSpec:
    if name not in BUILTINS:
        from openternary.services.plugins import load_plugin

        plugin = load_plugin("backends", name)
        spec = getattr(plugin, "spec", None)
        if not isinstance(spec, BackendSpec) or spec.name != name:
            raise CapabilityError("backend plugin must declare a matching BackendSpec")
        return spec
    return BUILTINS[name]


def validate_backend(config: AppConfig, *, require_installed: bool = False) -> BackendSpec:
    q = config.quantization
    spec = get_spec(q.backend)
    if q.scheme not in spec.schemes or q.weight_dtype not in spec.weight_dtypes:
        raise CapabilityError(f"{q.backend} supports schemes={spec.schemes}, weight_dtypes={spec.weight_dtypes}")
    if q.backend_options and q.backend in BUILTINS:
        raise CapabilityError(f"{q.backend}: no backend_options supported by API v1")
    if q.backend != "ternary" and q.mixed_precision:
        raise CapabilityError("mixed_precision is currently supported by ternary only")
    if require_installed and spec.describe()["installation"] == "not_installed":
        raise DependencyError(f"{q.backend} is not installed; use a separate compatible backend environment")
    return spec


def get_backend(config: AppConfig) -> Any:
    import importlib

    spec = validate_backend(config, require_installed=True)
    if spec.name not in BUILTINS:
        from openternary.services.plugins import load_plugin

        backend = load_plugin("backends", spec.name)()
        backend.validate(config)
        return backend
    module, name = spec.implementation.split(":")
    return getattr(importlib.import_module(module), name)()
