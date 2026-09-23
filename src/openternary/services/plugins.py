"""Versioned plugin discovery; code loads only after explicit selection."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from openternary.services.errors import CapabilityError

API_VERSION = 1
KINDS = ("backends", "passes", "exporters", "evaluators")
RESERVED = {
    "backends": {"ternary", "torchao"},
    "passes": {"noop", "clip"},
    "exporters": {"safetensors", "ternary-packed", "torchao", "gguf"},
    "evaluators": {"diffusion"},
}


def discover_plugins() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in KINDS:
        entries = importlib.metadata.entry_points(group=f"openternary.{kind}.v1")
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry.name] = counts.get(entry.name, 0) + 1
        for entry in entries:
            reserved = entry.name in RESERVED[kind]
            rows.append(
                {
                    "kind": kind,
                    "name": entry.name,
                    "entry_point": entry.value,
                    "distribution": entry.dist.name if entry.dist else None,
                    "version": entry.dist.version if entry.dist else None,
                    "status": "conflict" if counts[entry.name] > 1 or reserved else "discovered",
                    "api_version": API_VERSION,
                    "loaded": False,
                }
            )
    return rows


def load_plugin(kind: str, name: str) -> Any:
    if kind not in KINDS:
        raise CapabilityError(f"unknown plugin kind: {kind}")
    rows = [r for r in discover_plugins() if r["kind"] == kind and r["name"] == name]
    if len(rows) != 1 or rows[0]["status"] != "discovered":
        raise CapabilityError(f"missing or conflicting plugin: {kind}/{name}")
    entries = importlib.metadata.entry_points(group=f"openternary.{kind}.v1", name=name)
    plugin = next(iter(entries)).load()
    if getattr(plugin, "api_version", None) != API_VERSION:
        raise CapabilityError(f"plugin API version mismatch: {kind}/{name}")
    return plugin
