"""Versioned manifests, exact file inventories, and atomic artifact publication."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from filelock import FileLock

from openternary.experiment.metadata import write_json
from openternary.services.errors import CapabilityError

MANIFEST = "openternary-manifest.json"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def member_path(root: Path, name: str) -> Path:
    if (
        not name
        or "\\" in name
        or ":" in name
        or PurePosixPath(name).is_absolute()
        or any(p in {"..", "."} for p in name.split("/"))
    ):
        raise ValueError(f"unsafe artifact member: {name}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact member escapes root: {name}")
    return path


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact contains a symbolic link: {path}")
        if path.is_file() and path != root / MANIFEST:
            name = path.relative_to(root).as_posix()
            result[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return result


def write_manifest(root: Path, *, format: str, provenance: dict[str, Any]) -> dict[str, Any]:
    from openternary.utils.git import code_provenance

    files = inventory(root)
    if not files:
        raise ValueError("cannot publish an empty artifact")
    manifest = {
        "schema_version": 1,
        "format": format,
        "files": files,
        "provenance": {**provenance, "code": code_provenance()},
        "execution": "completed",
        "artifact_validation": "file_integrity",
        "model_reload": "not_run",
        "quality_acceptance": "not_run",
        "runtime": {"native_low_bit": False, "requires_unpack": format == "ternary-packed"},
    }
    manifest["manifest_fingerprint"] = canonical_hash(manifest)
    write_json(root / MANIFEST, manifest)
    return manifest


def validate_artifact(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported artifact manifest schema")
    recorded = manifest.pop("manifest_fingerprint", None)
    if recorded != canonical_hash(manifest):
        raise ValueError("manifest fingerprint mismatch")
    manifest["manifest_fingerprint"] = recorded
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest requires non-empty files")
    for name, expected in files.items():
        path = member_path(root, name)
        if not isinstance(expected, dict) or not path.is_file():
            raise ValueError(f"missing artifact member: {name}")
    if inventory(root) != files:
        raise ValueError("artifact file inventory, hash or size mismatch")
    return manifest


@contextlib.contextmanager
def atomic_artifact(destination: Path) -> Iterator[Path]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination.parent / f".{destination.name}.publish.lock")):
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            yield temporary
            validate_artifact(temporary)
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def snapshot_from_run(source: Path) -> Path:
    if (source / MANIFEST).is_file() or (source / "config.json").is_file():
        return source
    matches = [
        source / "artifacts" / name
        for name in ("calibrated_snapshot", "snapshot")
        if (source / "artifacts" / name / "config.json").is_file()
    ]
    if len(matches) != 1:
        raise ValueError("run must identify exactly one snapshot; pass the snapshot directory explicitly")
    return matches[0]


def export_artifact(source: Path, destination: Path, format: str = "safetensors") -> dict[str, Any]:
    plugin = None
    if format not in {"safetensors", "ternary-packed", "torchao"}:
        from openternary.services.plugins import load_plugin

        plugin = load_plugin("exporters", format)
    source = snapshot_from_run(source).resolve()
    if destination.resolve() == source or destination.resolve().is_relative_to(source):
        raise ValueError("export destination must be outside the source artifact")
    prior = validate_artifact(source) if (source / MANIFEST).is_file() else None
    source_format = prior["format"] if prior else "safetensors"
    if plugin is not None:
        plugin.validate(source_format)
    with atomic_artifact(destination) as staged:
        if plugin is not None:
            plugin.export(source, staged)
        elif format == "ternary-packed":
            if source_format != "safetensors":
                raise CapabilityError("packing requires a ternary fake-quant safetensors snapshot")
            from openternary.export.packed import pack_snapshot

            pack_snapshot(source, staged)
        elif source_format == "ternary-packed" and format == "safetensors":
            from openternary.export.packed import unpack_snapshot

            unpack_snapshot(source, staged)
        elif format == source_format:
            if prior:
                names = list(prior["files"])
            else:
                names = [
                    p.name
                    for p in source.iterdir()
                    if p.is_file()
                    and not p.name.startswith(".")
                    and p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}
                ]
            if format == "safetensors":
                from openternary.services.snapshot import tensor_inventory

                tensor_inventory(source)
            for name in names:
                target = member_path(staged, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                # HF cache files may be symlinks; published artifacts are standalone copies.
                shutil.copyfile(source / name, target)
        else:
            raise CapabilityError(f"cannot convert {source_format} to {format}")
        if format == "safetensors" and prior and prior.get("provenance", {}).get("adapter") == "diffusers":
            from openternary.adapters.runtime import normalize_diffusion_weights

            normalize_diffusion_weights(staged)
        manifest = write_manifest(
            staged,
            format=format,
            provenance={
                **(prior.get("provenance", {}) if prior else {}),
                "source": str(source),
                "parent_manifest": prior["manifest_fingerprint"] if prior else None,
                "parent_code": prior.get("provenance", {}).get("code") if prior else None,
                "source_files": {
                    p.relative_to(source).as_posix(): file_hash(p)
                    for p in source.rglob("*")
                    if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}
                },
            },
        )
    return {"artifact": str(destination.resolve()), "manifest": manifest, "artifact_validation": "file_integrity"}
