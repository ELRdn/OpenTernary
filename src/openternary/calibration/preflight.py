"""Fail-closed calibration preflight for the canonical Gemma 4 target."""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

from openternary.config.schema import AppConfig
from openternary.inspect.engine import run_inspection
from openternary.utils.hardware import get_hardware_info

CANONICAL_TARGET_COUNT = 205
CANONICAL_MODEL_ID = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
CANONICAL_MODEL_REVISION = "6befbaca7398925921802abd1f277b495b78b738"
CANONICAL_WEIGHT_SHA256 = "33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137"
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PROTOCOL_FILES = (
    "ROADMAP.md",
    "docs/PROJECT_SPEC.md",
    "docs/ARCHITECTURE.md",
    "docs/RESEARCH_PLAN.md",
    "docs/BENCHMARKS.md",
    "docs/plans/p0-p7-research-acceptance.md",
)
MAX_PHASE_GPU_HOURS = 24
MAX_ACTIVE_ARTIFACT_GB = 120


def _canonical_target_names() -> tuple[str, ...]:
    names: list[str] = []
    for layer_index in range(35):
        projections = [
            ("mlp", "down_proj"),
            ("mlp", "gate_proj"),
            ("mlp", "up_proj"),
            ("self_attn", "o_proj"),
            ("self_attn", "q_proj"),
        ]
        if layer_index < 15:
            projections.extend((("self_attn", "k_proj"), ("self_attn", "v_proj")))
        names.extend(
            f"model.language_model.layers.{layer_index}.{family}.{projection}.weight"
            for family, projection in projections
        )
    return tuple(sorted(names))


CANONICAL_TARGET_NAMES = _canonical_target_names()


def _revision_is_pinned(revision: str | None) -> bool:
    """Return whether revision names one immutable Git commit."""
    return isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None


def _revision_matches_canonical(revision: str | None) -> bool:
    """Return whether revision is the preregistered Gemma 4 source revision."""
    return revision == CANONICAL_MODEL_REVISION


def _weight_matches_canonical(source_files: list[dict[str, Any]]) -> bool:
    weight_files = [entry for entry in source_files if str(entry.get("name", "")).endswith(".safetensors")]
    return len(weight_files) == 1 and weight_files[0].get("sha256") == CANONICAL_WEIGHT_SHA256


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest(snapshot: pathlib.Path) -> list[dict[str, Any]]:
    files = [path for path in sorted(snapshot.iterdir()) if path.is_file()]
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                cwd=_PROJECT_ROOT,
            ).stdout.strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def _runtime_info(output: pathlib.Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hardware": get_hardware_info(),
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["hip"] = getattr(torch.version, "hip", None)
        info["cuda"] = getattr(torch.version, "cuda", None)
    except ImportError:
        info["torch"] = None
        info["hip"] = None
        info["cuda"] = None
    usage = shutil.disk_usage(output.parent)
    info["output_disk"] = {
        "path": str(output.parent.resolve()),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
    }
    return info


def _protocol_files() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for relative in _PROTOCOL_FILES:
        path = _PROJECT_ROOT / relative
        if path.exists():
            result.append({"path": relative, "sha256": _sha256_file(path)})
    return result


def _read_small_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size > 10 * 1024 * 1024:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _existing_artifacts(root: pathlib.Path, current_output: pathlib.Path) -> dict[str, Any]:
    if not root.exists():
        return {"root": str(root), "run_count": 0, "truncated": False, "runs": []}
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.resolve() != current_output.resolve()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    limit = 200
    runs: list[dict[str, Any]] = []
    for path in directories[:limit]:
        calibration = _read_small_json(path / "calibration.json")
        metrics = _read_small_json(path / "metrics.json")
        state_dir = path / "artifacts" / "calibration_state"
        checkpoint_dir = path / "artifacts" / "checkpoint"
        state_count = (
            len(list(state_dir.glob("*.safetensors"))) + len(list(state_dir.glob("*.pt"))) if state_dir.exists() else 0
        )
        checkpoint_count = len(list(checkpoint_dir.glob("*.pt"))) if checkpoint_dir.exists() else 0
        if calibration.get("used_dummy_simulation") is True:
            classification = "synthetic-fixture"
        elif state_count not in (0, CANONICAL_TARGET_COUNT):
            classification = "invalid-target-set"
        elif calibration or metrics or checkpoint_count or state_count:
            classification = "engineering-evidence"
        else:
            classification = "unclassified"
        runs.append(
            {
                "name": path.name,
                "classification": classification,
                "state_files": state_count,
                "checkpoint_files": checkpoint_count,
                "has_calibration": bool(calibration),
                "has_metrics": bool(metrics),
                "scientific_acceptance": False,
            }
        )
    return {
        "root": str(root.resolve()),
        "run_count": len(directories),
        "truncated": len(directories) > limit,
        "runs": runs,
    }


def build_preflight_report(
    config: AppConfig,
    snapshot: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    """Build a machine-readable P0 report without loading model payloads."""
    snapshot = snapshot.resolve()
    inspection_config = config.model_copy(
        update={"model": config.model.model_copy(update={"id": CANONICAL_MODEL_ID})},
    )
    inspection = run_inspection(inspection_config, snapshot_path=snapshot, load_weights=False)
    targets = [entry["name"] for entry in inspection["tensors"] if entry["quantizable"]]
    source_files = _source_manifest(snapshot)
    checks = {
        "revision_pinned": _revision_is_pinned(config.model.revision),
        "canonical_revision": _revision_matches_canonical(config.model.revision),
        "source_has_weights": any(item["name"].endswith(".safetensors") for item in source_files),
        "canonical_target_count": len(targets) == CANONICAL_TARGET_COUNT,
        "canonical_target_inventory": targets == list(CANONICAL_TARGET_NAMES),
        "canonical_weight_sha256": _weight_matches_canonical(source_files),
        "no_per_layer_targets": not any("per_layer" in name for name in targets),
        "protocol_files_present": len(_protocol_files()) == len(_PROTOCOL_FILES),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "acceptance_scope": "preflight",
        "status": status,
        "checks": checks,
        "model": {
            "id": config.model.id,
            "inspection_id": CANONICAL_MODEL_ID,
            "source_locator": str(snapshot),
            "revision": config.model.revision,
            "snapshot": str(snapshot),
            "inspection_fingerprint": inspection["inspection_fingerprint"],
        },
        "source": {
            "revision": config.model.revision,
            "files": source_files,
        },
        "target_inventory": {
            "count": len(targets),
            "expected_count": CANONICAL_TARGET_COUNT,
            "expected_names_sha256": hashlib.sha256("\n".join(CANONICAL_TARGET_NAMES).encode("utf-8")).hexdigest(),
            "names": targets,
        },
        "resource_budget": {
            "evaluation": "not_evaluated",
            "note": "Budget limits are recorded policy; preflight status applies only to checks above.",
            "max_gpu_hours_per_phase": MAX_PHASE_GPU_HOURS,
            "max_active_artifact_gb": MAX_ACTIVE_ARTIFACT_GB,
            "stop_new_candidates_at_fraction": 0.8,
        },
        "environment": _runtime_info(output),
        "git": _git_state(),
        "protocol_files": _protocol_files(),
        "existing_artifacts": _existing_artifacts(output.parent, output),
        "scientific_acceptance": False,
    }


def write_preflight_report(report: dict[str, Any], output: pathlib.Path) -> pathlib.Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preflight.json"
    if path.exists():
        raise FileExistsError(f"preflight report already exists: {path}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = ["build_preflight_report", "write_preflight_report"]
