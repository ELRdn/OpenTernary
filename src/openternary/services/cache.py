"""Reference-aware removal of explicitly selected managed artifacts."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from openternary.services.artifacts import MANIFEST, validate_artifact


def list_artifacts(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError("cache root must be an existing directory")
    rows = []
    for manifest in sorted(root.rglob(MANIFEST)):
        if not manifest.resolve().is_relative_to(root.resolve()):
            raise ValueError("cache entry escapes root")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError(f"invalid cache manifest: {manifest}")
        rows.append(
            {
                "artifact": str(manifest.parent.resolve()),
                "format": data.get("format"),
                "source": data.get("provenance", {}).get("source"),
                "bytes": sum(row["bytes"] for row in data.get("files", {}).values()),
                "validation": "not_rehashed",
                "quality_acceptance": data.get("quality_acceptance", "not_run"),
            }
        )
    return rows


def remove_artifact(root: Path, target: Path, *, execute: bool = False) -> dict[str, Any]:
    root, target = root.resolve(), target.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("target must be a managed artifact strictly inside the cache root")

    def check() -> dict[str, Any]:
        manifest = validate_artifact(target)
        incoming = [
            row["artifact"]
            for row in list_artifacts(root)
            if row["artifact"] != str(target) and row["source"] and Path(row["source"]).resolve() == target
        ]
        # Search journals retain best candidates and completed resume/cache entries.
        for journal in root.rglob("search.json"):
            data = json.loads(journal.read_text(encoding="utf-8"))
            if any(
                (journal.parent / row["folder"] / "artifact").resolve() == target for row in data.get("candidates", [])
            ):
                incoming.append(str(journal))
        if incoming:
            raise ValueError(f"artifact is referenced by: {incoming}")
        return {
            "target": str(target),
            "root": str(root),
            "bytes": sum(row["bytes"] for row in manifest["files"].values()),
            "references_checked_within": str(root),
            "external_references": "unknown",
            "status": "planned",
        }

    if not execute:
        return check()
    with FileLock(str(target.parent / f".{target.name}.publish.lock"), timeout=0):
        result = check()
        tombstone = target.parent / f".{target.name}-deleting-{uuid.uuid4().hex}"
        if not tombstone.resolve().is_relative_to(root):
            raise ValueError("deletion staging path escapes cache root")
        target.rename(tombstone)
        shutil.rmtree(tombstone)
        return {**result, "status": "removed"}
