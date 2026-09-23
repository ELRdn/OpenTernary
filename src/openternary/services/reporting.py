"""Context-local results and progress; no CLI dependency."""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class Operation:
    command: str
    result: dict[str, Any] = field(default_factory=dict)
    stream: TextIO | None = None
    sequence: int = 0

    def event(self, stage: str, completed: int | None = None, total: int | None = None) -> None:
        if self.stream is None:
            return
        self.sequence += 1
        row = {
            "schema_version": 1,
            "sequence": self.sequence,
            "command": self.command,
            "run_id": self.result.get("run_id"),
            "stage": stage,
            "completed": completed,
            "total": total,
        }
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()


current: contextvars.ContextVar[Operation | None] = contextvars.ContextVar("openternary_operation", default=None)


def publish(**result: Any) -> None:
    operation = current.get()
    if operation is not None:
        operation.result.update(result)


def progress(stage: str, completed: int, total: int | None = None) -> None:
    """Publish bounded progress when an operation has an explicit event stream."""
    operation = current.get()
    if operation is not None:
        operation.event(stage, completed, total)


def track_run(path: Path, run_id: str) -> None:
    publish(run_dir=str(path), run_id=run_id)
    operation = current.get()
    if operation is not None:
        operation.event("run_created")
