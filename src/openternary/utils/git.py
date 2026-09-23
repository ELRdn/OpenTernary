"""Git metadata helpers."""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
from typing import Any


def get_git_commit(cwd: pathlib.Path | None = None) -> str:
    """現在の git commit hash を取得。失敗時は 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or pathlib.Path.cwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_branch(cwd: pathlib.Path | None = None) -> str:
    """現在の git branch を取得。失敗時は 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd or pathlib.Path.cwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def code_provenance() -> dict[str, Any]:
    """Hash only this package's code, independent of the invocation directory."""
    package = pathlib.Path(__file__).resolve().parents[1]
    root = package.parent.parent
    files = {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    dirty: bool | None = None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "src/openternary", "pyproject.toml", "uv.lock"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            dirty = bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"commit": get_git_commit(root), "dirty": dirty, "package_files": files}
