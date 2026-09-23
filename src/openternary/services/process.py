"""Bounded subprocess execution with owned child-process cleanup."""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO


def stop_process_tree(process: subprocess.Popen[Any]) -> None:
    import psutil

    children: list[Any] = []
    with contextlib.suppress(psutil.NoSuchProcess):
        children = psutil.Process(process.pid).children(recursive=True)
    for child in reversed(children):
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    with contextlib.suppress(ProcessLookupError, OSError):
        process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    psutil.wait_procs(children, timeout=5)


def run_process(
    command: Sequence[str],
    *,
    timeout: float,
    stdout: TextIO,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    poll_check: Callable[[], None] | None = None,
) -> int:
    if timeout <= 0:
        raise ValueError("process timeout must be positive")
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        while process.poll() is None:
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"process exceeded {timeout:g} second deadline")
            if poll_check is not None:
                poll_check()
            time.sleep(0.05)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, list(command))
        return int(process.returncode)
    except BaseException:
        stop_process_tree(process)
        raise
