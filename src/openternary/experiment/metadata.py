"""Experiment metadata collection (FR-06)."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import platform
import sys
import tempfile

import psutil

from openternary import __version__
from openternary.config.schema import AppConfig
from openternary.utils.git import code_provenance, get_git_branch, get_git_commit


def _uv_lock_hash() -> str:
    """uv.lock のハッシュを計算（再現性bundle用）."""
    try:
        p = pathlib.Path("uv.lock")
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    return "unknown"


def collect_environment(
    config: AppConfig,
    run_id: str,
) -> dict[str, object]:
    """environment.json 用のメタデータを収集."""
    hardware: dict[str, str] = {}
    try:
        hardware["platform"] = platform.platform()
        hardware["processor"] = platform.processor() or "unknown"
        hardware["ram_gb"] = str(round(psutil.virtual_memory().total / (1024**3), 2))
    except Exception:
        hardware["platform"] = "unknown"
        hardware["ram_gb"] = "unknown"

    # GPU
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            hardware["gpu_name"] = torch.cuda.get_device_name(0)
            hardware["cuda_available"] = True  # type: ignore[assignment]
        else:
            hardware["gpu_name"] = "none"
            hardware["cuda_available"] = False  # type: ignore[assignment]
    except ImportError:
        hardware["gpu_name"] = "not installed"
        hardware["cuda_available"] = False  # type: ignore[assignment]
    except Exception:
        hardware["gpu_name"] = "unknown"
        hardware["cuda_available"] = False  # type: ignore[assignment]

    env: dict[str, object] = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "openternary_version": __version__,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_manager": "uv",
        "uv_lock_hash": _uv_lock_hash(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "code_provenance": code_provenance(),
        "seed": config.seed,
        "PYTHONHASHSEED": __import__("os").environ.get("PYTHONHASHSEED", "not set (subprocess only)"),
        "config": config.model_dump(),
        "hardware": hardware,
    }
    return env


def write_json(path: pathlib.Path, data: dict[str, object]) -> None:
    """Atomically replace a complete JSON document, including after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = f.name
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            pathlib.Path(temporary).unlink(missing_ok=True)
