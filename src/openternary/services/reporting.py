"""Context-local results and progress; no CLI dependency.

Progress rows (``--events-jsonl``) keep ``schema_version`` 1. Every row has
``schema_version, sequence, command, run_id, stage, completed, total`` (the original keys)
plus the additive ``unit, elapsed_s, detail``. Library code reports progress with
:func:`progress`; it is a no-op when no event stream is open (Python API, tests, plain CLI).
"""

from __future__ import annotations

import contextvars
import json
import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO, TypeVar

T = TypeVar("T")

MIN_INTERVAL_S = 0.2  # at most ~5 rows per second per stage; first and last rows always pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, bool | int | str):
        return value
    return str(value)


@dataclass
class Operation:
    command: str
    result: dict[str, Any] = field(default_factory=dict)
    stream: TextIO | None = None
    sequence: int = 0
    started_at: float = field(default_factory=time.monotonic)
    error_context: str | None = None  # human-readable "what failed", set by the CLI before exiting
    _last_emit: dict[str, float] = field(default_factory=dict, repr=False)

    def event(
        self,
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        *,
        unit: str | None = None,
        detail: dict[str, Any] | None = None,
        throttle: bool = False,
    ) -> None:
        if self.stream is None:
            return
        now = time.monotonic()
        if throttle and not (completed is not None and total is not None and completed >= total):
            last = self._last_emit.get(stage)
            if last is not None and now - last < MIN_INTERVAL_S:
                return
        self._last_emit[stage] = now
        self.sequence += 1
        row = {
            "schema_version": 1,
            "sequence": self.sequence,
            "command": self.command,
            "run_id": self.result.get("run_id"),
            "stage": stage,
            "completed": completed,
            "total": total,
            "unit": unit,
            "elapsed_s": round(now - self.started_at, 3),
            "detail": _json_safe(detail or {}),
        }
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()


current: contextvars.ContextVar[Operation | None] = contextvars.ContextVar("openternary_operation", default=None)


def publish(**result: Any) -> None:
    operation = current.get()
    if operation is not None:
        operation.result.update(result)


def progress(
    stage: str,
    completed: int | None = None,
    total: int | None = None,
    *,
    unit: str | None = None,
    throttle: bool = True,
    **detail: Any,
) -> None:
    """Report progress of a long-running step. Throttled per stage unless ``throttle=False``."""
    operation = current.get()
    if operation is not None:
        operation.event(stage, completed, total, unit=unit, detail=detail, throttle=throttle)


def milestone(stage: str, **detail: Any) -> None:
    """Report a one-off step boundary (never throttled)."""
    progress(stage, throttle=False, **detail)


def track(
    items: Iterable[T], stage: str, total: int | None = None, *, unit: str | None = None, **detail: Any
) -> Iterator[T]:
    """Yield ``items`` while reporting ``stage`` progress after each one."""
    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
    progress(stage, 0, total, unit=unit, **detail)
    for done, item in enumerate(items, start=1):
        yield item
        progress(stage, done, total, unit=unit, **detail)


def track_run(path: Path, run_id: str) -> None:
    publish(run_dir=str(path), run_id=run_id)
    operation = current.get()
    if operation is not None:
        operation.event("run_created")
