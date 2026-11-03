"""Background jobs: run ``openternary <command> --json --events-jsonl`` in a child process.

The TUI never runs heavy work in its own process. Each job is the ordinary CLI in a child process, so
the machine contract (JSON envelope, exit codes, events, run locks, ``last_failure.json``) is exactly
what scripts get, and an out-of-memory crash cannot take the interface down with it.

Every job keeps a small directory (``job.json``, ``events.jsonl``, ``stdout.json``, ``stderr.log``)
so a reopened TUI can show past jobs and keep monitoring ones that are still running.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RECORD, EVENTS, STDOUT, STDERR = "job.json", "events.jsonl", "stdout.json", "stderr.log"
BREAK_ENV = "OPENTERNARY_BREAK_IS_INTERRUPT"  # child turns Ctrl+Break into KeyboardInterrupt (Windows)
KEEP = 200  # finished job records kept on disk


def state_dir() -> Path:
    if override := os.environ.get("OPENTERNARY_STATE_DIR"):
        return Path(override) / "jobs"
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return root / "openternary" / "jobs"


def child_command(args: list[str], events: Path) -> list[str]:
    return [sys.executable, "-m", "openternary", *args, "--json", "--events-jsonl", str(events)]


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            BREAK_ENV: "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
            "OPENTERNARY_NO_UPDATE_CHECK": "1",
        }
    )
    return env


# --- progress -----------------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    first_s: float = 0.0  # elapsed_s of the first row of this stage
    first_completed: int = 0
    last_s: float = 0.0

    @property
    def fraction(self) -> float | None:
        if self.completed is None or not self.total:
            return None
        return min(1.0, self.completed / self.total)

    @property
    def done(self) -> bool:
        return self.total is not None and self.completed is not None and self.completed >= self.total

    def eta_s(self) -> float | None:
        """Remaining seconds from this stage's own average rate; None until it is measurable."""
        if self.completed is None or not self.total or self.done:
            return None
        progressed, spent = self.completed - self.first_completed, self.last_s - self.first_s
        if progressed <= 0 or spent <= 0:
            return None
        return (self.total - self.completed) * spent / progressed


LIFECYCLE = {"started", "run_created", "completed", "failed", "interrupted"}


@dataclass
class Progress:
    stages: dict[str, Stage] = field(default_factory=dict)
    current: str | None = None
    elapsed_s: float = 0.0
    rows: int = 0
    outcome: str | None = None  # completed / failed / interrupted, from the last lifecycle row

    def feed(self, row: dict[str, Any]) -> None:
        self.rows += 1
        self.elapsed_s = float(row.get("elapsed_s") or self.elapsed_s)
        name = str(row.get("stage"))
        if name in LIFECYCLE:
            if name in {"completed", "failed", "interrupted"}:
                self.outcome = name
            return
        stage = self.stages.get(name)
        if stage is None:
            stage = self.stages[name] = Stage(
                name, first_s=self.elapsed_s, first_completed=int(row.get("completed") or 0)
            )
        stage.completed, stage.total = row.get("completed"), row.get("total")
        stage.unit = row.get("unit") or stage.unit
        stage.detail = row.get("detail") or {}
        stage.last_s = self.elapsed_s
        self.current = name


class EventTail:
    """Incrementally read a JSONL file that another process is still appending to."""

    def __init__(self, path: Path) -> None:
        self.path, self.offset, self.partial = path, 0, b""

    def read(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
        except FileNotFoundError:
            return []
        self.offset += len(chunk)
        data = self.partial + chunk
        lines = data.split(b"\n")
        self.partial = lines.pop()  # an unfinished last line waits for the next read
        rows = []
        for line in lines:
            if line.strip():
                with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                    rows.append(json.loads(line.decode("utf-8")))
        return rows


# --- jobs ---------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    args: list[str]
    cwd: str
    started_at: float
    pid: int | None = None
    exit_code: int | None = None
    finished_at: float | None = None
    cancel_requested: bool = False
    path: Path = field(default=Path(), repr=False)

    @property
    def command(self) -> str:
        return " ".join(a for a in self.args[:2] if not a.startswith("-")) or "?"

    @property
    def events_path(self) -> Path:
        return self.path / EVENTS

    @property
    def stdout_path(self) -> Path:
        return self.path / STDOUT

    @property
    def stderr_path(self) -> Path:
        return self.path / STDERR

    def save(self) -> None:
        data = asdict(self)
        data.pop("path")
        tmp = self.path / f".{RECORD}.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path / RECORD)

    @classmethod
    def load(cls, path: Path) -> Job:
        data = json.loads((path / RECORD).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, path=path)

    def envelope(self) -> dict[str, Any] | None:
        try:
            lines = self.stdout_path.read_text(encoding="utf-8").strip().splitlines()
        except (FileNotFoundError, UnicodeDecodeError):
            return None
        for line in reversed(lines):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(line)
                if isinstance(value, dict) and "exit_code" in value:
                    return value
        return None

    def stderr_tail(self, lines: int = 40) -> list[str]:
        try:
            with self.stderr_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 64_000))
                text = handle.read().decode("utf-8", "replace")
        except FileNotFoundError:
            return []
        return text.splitlines()[-lines:]

    @property
    def status(self) -> str:
        if self.exit_code is None:
            return "running" if self.finished_at is None else "lost"
        if self.exit_code == 0:
            return "completed"
        return "interrupted" if self.exit_code == 130 else "failed"


def _alive(pid: int, started_at: float) -> bool:
    """True if ``pid`` is still the process we started (guards against pid reuse)."""
    import psutil

    try:
        process = psutil.Process(pid)
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and (process.create_time() <= started_at + 5)
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


class JobManager:
    """Start, track, cancel and list jobs. Owns the ``Popen`` handles of jobs it started."""

    def __init__(self, root: Path | None = None, command: Callable[[list[str], Path], list[str]] = child_command):
        self.root = root or state_dir()
        self.command = command
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def start(self, args: list[str], cwd: Path | None = None) -> Job:
        self.root.mkdir(parents=True, exist_ok=True)
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        path = self.root / job_id
        path.mkdir()
        job = Job(job_id, list(args), str((cwd or Path.cwd()).resolve()), time.time(), path=path)
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # target of CTRL_BREAK_EVENT
        else:
            kwargs["start_new_session"] = True  # survives the TUI closing; cancelled as a group
        with job.stdout_path.open("wb") as out, job.stderr_path.open("wb") as err:
            process = subprocess.Popen(
                self.command(job.args, job.events_path),
                cwd=job.cwd,
                env=child_env(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                **kwargs,
            )
        job.pid = process.pid
        self._processes[job_id] = process
        job.save()
        return job

    def refresh(self, job: Job) -> Job:
        """Record the exit of a job that has finished; cheap enough to call on every tick."""
        if job.exit_code is not None or job.finished_at is not None:
            return job
        process = self._processes.get(job.id)
        if process is not None:
            code = process.poll()
            if code is None:
                return job
            job.exit_code = 130 if code in (-signal.SIGINT, 0xC000013A) else code
            self._processes.pop(job.id, None)
        elif job.pid is not None and _alive(job.pid, job.started_at):
            return job
        else:  # started by an earlier session: the envelope is the only record of how it ended
            envelope = job.envelope()
            job.exit_code = int(envelope["exit_code"]) if envelope else None
        job.finished_at = time.time()
        job.save()
        return job

    def cancel(self, job: Job, force: bool = False) -> None:
        """First request: an interrupt (the CLI then exits 130 and records the failure). ``force``: kill."""
        if job.pid is None or job.status != "running":
            return
        job.cancel_requested = True
        job.save()
        try:
            if force:
                import psutil

                parent = psutil.Process(job.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            elif sys.platform == "win32":
                os.kill(job.pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                os.killpg(job.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def jobs(self, limit: int = 100) -> list[Job]:
        """Newest first; job ids start with a timestamp, so name order is start order."""
        if not self.root.is_dir():
            return []
        found = []
        for path in sorted(self.root.iterdir(), reverse=True)[:limit]:
            if (path / RECORD).is_file():
                with contextlib.suppress(json.JSONDecodeError, TypeError, OSError):
                    found.append(self.refresh(Job.load(path)))
        return found

    def running(self) -> list[Job]:
        return [job for job in self.jobs(limit=10_000) if job.status == "running"]

    def prune(self, keep: int = KEEP) -> int:
        import shutil

        finished = [job for job in self.jobs(limit=10_000) if job.status != "running"]
        removed = 0
        for job in finished[keep:]:
            shutil.rmtree(job.path, ignore_errors=True)
            removed += 1
        return removed


def preview(args: list[str], cwd: Path | None = None, timeout: float = 60.0) -> tuple[int, str]:
    """Run ``openternary <args> --dry-run`` and return (exit code, plain text) for a plan screen."""
    env = child_env()
    env.update({"TERM": "dumb", "COLUMNS": "100"})
    try:
        done = subprocess.run(
            [sys.executable, "-m", "openternary", *args, "--dry-run"],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"dry-run did not finish within {timeout:.0f}s"
    text = (done.stdout + done.stderr).decode("utf-8", "replace")
    return done.returncode, text.strip()
