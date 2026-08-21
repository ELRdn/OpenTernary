"""Git metadata helpers."""

from __future__ import annotations

import pathlib
import subprocess


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
