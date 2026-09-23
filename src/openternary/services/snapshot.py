"""Safetensors inventory and bounded tensor access for single or sharded snapshots."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from openternary.services.artifacts import member_path
from openternary.utils.safetensors_header import parse_safetensors_header


def tensor_inventory(source: Path) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    for path in sorted(source.glob("*.safetensors")):
        for name, entry in parse_safetensors_header(path).items():
            if name in tensors:
                raise ValueError(f"duplicate tensor across shards: {name}")
            tensors[name] = {**entry, "file": path.name}
    if not tensors:
        raise ValueError(f"no safetensors tensors in {source}")
    for index in source.glob("*.safetensors.index.json"):
        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict) or weight_map != {name: row["file"] for name, row in tensors.items()}:
            raise ValueError("safetensors index does not match tensor inventory")
        for name in weight_map.values():
            member_path(source, name)
    return tensors


class SnapshotReader:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.inventory = tensor_inventory(source)
        self.stack = contextlib.ExitStack()
        self.files: dict[str, Any] = {}

    def __enter__(self) -> SnapshotReader:
        from safetensors import safe_open

        try:
            for name in sorted({row["file"] for row in self.inventory.values()}):
                self.files[name] = self.stack.enter_context(
                    safe_open(str(self.source / name), framework="pt", device="cpu")
                )
        except BaseException:
            self.stack.close()
            raise
        return self

    def keys(self) -> list[str]:
        return sorted(self.inventory)

    def get_tensor(self, name: str) -> Any:
        return self.files[self.inventory[name]["file"]].get_tensor(name)

    def __exit__(self, *args: Any) -> None:
        self.stack.close()
