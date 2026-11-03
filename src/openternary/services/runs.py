"""Read-only discovery of run directories (``runs/<date>_<slug>/``) and guarded deletion."""

from __future__ import annotations

import contextlib
import datetime
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A run does not record which command made it; the result file it wrote says so.
MARKERS = (
    ("search.json", "optimize"),
    ("calibration.json", "calibrate"),
    ("compare.json", "compare"),
    ("evaluation.json", "evaluate"),
    ("quality.json", "quality"),
    ("benchmark.json", "benchmark"),
    ("packed.json", "export"),
    ("quantization.json", "quantize"),
    ("inspection.json", "inspect"),
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


@dataclass
class RunInfo:
    path: Path
    name: str
    status: str
    kind: str | None
    model: str | None
    run_id: str | None
    started: datetime.datetime
    accepted: bool | None  # quality gate; None when no gate was evaluated
    error: str | None

    @property
    def quality(self) -> str:
        return "not_run" if self.accepted is None else "accepted" if self.accepted else "rejected"


def is_run(path: Path) -> bool:
    return path.is_dir() and (path / "metrics.json").is_file() and (path / "config.yaml").is_file()


def read_run(path: Path) -> RunInfo:
    metrics = _json(path / "metrics.json")
    environment = _json(path / "environment.json")
    gate = _json(path / "acceptance.json")
    try:
        started = datetime.datetime.fromisoformat(str(environment["timestamp"]))
    except (KeyError, ValueError):
        started = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.UTC)
    if started.tzinfo is None:
        started = started.replace(tzinfo=datetime.UTC)
    accepted = gate.get("accepted") if isinstance(gate.get("accepted"), bool) else None
    return RunInfo(
        path=path,
        name=path.name,
        status=str(metrics.get("status") or "unknown"),
        kind=next((kind for marker, kind in MARKERS if (path / marker).is_file()), None),
        model=_json(path / "model.json").get("id"),
        run_id=metrics.get("run_id") or environment.get("run_id"),
        started=started,
        accepted=accepted,
        error=metrics.get("error_message"),
    )


def list_runs(root: Path) -> list[RunInfo]:
    if not root.is_dir():
        return []
    runs = [read_run(path) for path in root.iterdir() if is_run(path)]
    return sorted(runs, key=lambda run: (run.started, run.name), reverse=True)


def files(path: Path, limit: int = 40) -> list[tuple[str, int]]:
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and not item.name.startswith("."):
            rows.append((str(item.relative_to(path)), item.stat().st_size))
            if len(rows) >= limit:
                break
    return rows


def delete_run(root: Path, target: Path) -> None:
    """Delete one run directory. Refuses anything that is not a run strictly inside ``root``,
    and a run another process is executing (its ``.execution.lock`` is held)."""
    from filelock import FileLock, Timeout

    root, target = root.resolve(), target.resolve()
    if target == root or target.parent != root:
        raise ValueError("only a run directly inside the runs folder can be deleted")
    if not is_run(target):
        raise ValueError(f"not a run directory: {target}")
    try:
        with FileLock(str(target / ".execution.lock"), timeout=0):
            pass
    except Timeout as exc:
        raise ValueError(f"run is in use by another process: {target}") from exc
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(target)
