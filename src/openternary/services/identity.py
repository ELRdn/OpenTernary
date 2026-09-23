"""Path-independent source lineage and current interface evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openternary.services.artifacts import MANIFEST, canonical_hash, file_hash, validate_artifact


def snapshot_identity(source: Path | None) -> dict[str, Any]:
    if source is None or not source.is_dir():
        return {"status": "unresolved", "scope": "unknown"}
    source = source.resolve()
    files = {
        p.relative_to(source).as_posix(): file_hash(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.name != MANIFEST and p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}
    }
    interface = {
        name: digest
        for name, digest in files.items()
        if any(
            token in Path(name).name
            for token in ("tokenizer", "vocab", "merges", "processor", "special_tokens", "chat_template")
        )
    }
    identity: dict[str, Any] = {
        "status": "resolved" if files else "unresolved",
        "source_fingerprint": canonical_hash(files),
        "interface_fingerprint": canonical_hash(interface) if interface else None,
        "interface_files": interface,
        "scope": "synthetic" if (source / "openternary-fixture.json").is_file() else "model",
    }
    if (source / MANIFEST).is_file():
        origin = validate_artifact(source).get("provenance", {}).get("identity")
        if isinstance(origin, dict):
            identity.update(source_fingerprint=origin.get("source_fingerprint"), scope=origin.get("scope", "unknown"))
    # The marker is metadata, never permission to load arbitrary external weights.
    if (source / "openternary-fixture.json").is_file():
        marker = json.loads((source / "openternary-fixture.json").read_text(encoding="utf-8"))
        if marker.get("pretrained") is not False:
            raise ValueError("invalid synthetic fixture marker")
    return identity
