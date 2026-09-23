"""Install disposable entry-point wheels in an explicitly selected validation Python."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def wheel(root: Path, name: str, entries: str) -> Path:
    info = f"{name}-1.0.dist-info"
    files = {
        f"{name}.py": "class Good:\n    api_version=1\nclass Bad:\n    api_version=999\n",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: openternary-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{info}/entry_points.txt": "[openternary.passes.v1]\n" + entries,
    }
    rows = []
    for path, content in files.items():
        data = content.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([path, f"sha256={digest}", len(data)])
    rows.append([f"{info}/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record).writerows(rows)
    files[f"{info}/RECORD"] = record.getvalue()
    output = root / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(output, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--uv", required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    names = ["openternary_validation_plugin_a", "openternary_validation_plugin_b"]
    installed = {d.metadata["Name"].replace("-", "_") for d in importlib.metadata.distributions()}
    if set(names) & installed:
        raise ValueError("validation plugin already installed; refusing to replace it")
    first = wheel(
        args.root, names[0], f"validation = {names[0]}:Good\nbad_api = {names[0]}:Bad\nnoop = {names[0]}:Good\n"
    )
    second = wheel(args.root, names[1], f"validation = {names[1]}:Good\n")
    commands = []

    def execute(command: list[str]) -> None:
        with (args.root / "install.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        commands.append({"command": command, "exit_code": result.returncode})
        result.check_returncode()

    try:
        execute([args.uv, "pip", "install", "--python", sys.executable, "--no-deps", str(first)])
        from openternary.services.errors import CapabilityError
        from openternary.services.plugins import discover_plugins, load_plugin

        assert load_plugin("passes", "validation").api_version == 1
        assert any(r["name"] == "noop" and r["status"] == "conflict" for r in discover_plugins())
        try:
            load_plugin("passes", "bad_api")
        except CapabilityError as exc:
            assert "API version mismatch" in str(exc)
        else:
            raise AssertionError("bad API accepted")
        execute([args.uv, "pip", "install", "--python", sys.executable, "--no-deps", str(second)])
        rows = [r for r in discover_plugins() if r["name"] == "validation"]
        assert len(rows) == 2 and all(r["status"] == "conflict" for r in rows)
        try:
            load_plugin("passes", "validation")
        except CapabilityError:
            pass
        else:
            raise AssertionError("duplicate plugin accepted")
        status = "passed"
    finally:
        execute([args.uv, "pip", "uninstall", "--python", sys.executable, *names])
    from openternary.services.artifacts import file_hash

    report = {
        "status": status,
        "python": sys.executable,
        "prefix": sys.prefix,
        "checks": ["actual_entry_point", "reserved_conflict", "duplicate_conflict", "api_mismatch"],
        "commands": commands,
        "wheels": {p.name: file_hash(p) for p in (first, second)},
    }
    (args.root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
