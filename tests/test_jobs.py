"""Background jobs and run discovery used by the TUI (no Textual involved)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from filelock import FileLock

from openternary.services import jobs as jobsvc
from openternary.services import runs as runsvc

ROOT = Path(__file__).resolve().parents[1]

# Mimics the CLI: Ctrl+Break becomes KeyboardInterrupt when the job runner asks (Windows), events go
# to the file named on the command line, and the last stdout line is the JSON envelope.
FAKE = r"""
import json, os, signal, sys, time
if os.environ.get("OPENTERNARY_BREAK_IS_INTERRUPT") and hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)
events = open(sys.argv[1], "x", encoding="utf-8")
steps, delay = int(sys.argv[2]), float(sys.argv[3])
start = time.monotonic()
def ev(seq, stage, done=None, total=None, **detail):
    row = {"schema_version": 1, "sequence": seq, "command": "fake", "run_id": None, "stage": stage,
           "completed": done, "total": total, "unit": "steps", "elapsed_s": round(time.monotonic() - start, 3),
           "detail": detail}
    events.write(json.dumps(row) + "\n"); events.flush()
def envelope(code, error=None):
    print(json.dumps({"schema_version": 1, "command": "fake", "exit_code": code, "result": {}, "error": error}))
try:
    ev(1, "started")
    for i in range(steps + 1):
        ev(i + 2, "fake.steps", i, steps, loss=1.0 / (i + 1))
        print(f"step {i}", file=sys.stderr, flush=True)
        if i < steps:
            time.sleep(delay)
    ev(steps + 3, "completed")
    envelope(0)
except KeyboardInterrupt:
    ev(99, "interrupted"); envelope(130, "KeyboardInterrupt"); sys.exit(130)
"""


@pytest.fixture
def fake(tmp_path: Path) -> Path:
    script = tmp_path / "fake_job.py"
    script.write_text(FAKE, encoding="utf-8")
    return script


def manager(tmp_path: Path, script: Path, steps: int, delay: float) -> jobsvc.JobManager:
    return jobsvc.JobManager(
        tmp_path / "jobs",
        command=lambda args, events: [sys.executable, str(script), str(events), str(steps), str(delay)],
    )


def wait(predicate, timeout: float = 30.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.05)


def test_event_tail_reads_only_complete_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    tail = jobsvc.EventTail(path)
    assert tail.read() == []  # not created yet
    path.write_bytes(b'{"stage": "a"}\n{"stage": "b"')
    assert [r["stage"] for r in tail.read()] == ["a"]
    with path.open("ab") as handle:
        handle.write(b', "x": 1}\nnot json\n')
    assert tail.read() == [{"stage": "b", "x": 1}]  # the half-written row waited; garbage is skipped


def test_progress_tracks_stages_eta_and_outcome() -> None:
    progress = jobsvc.Progress()
    for row in (
        {"stage": "started", "elapsed_s": 0.0},
        {"stage": "convert.tensors", "completed": 0, "total": 100, "unit": "tensors", "elapsed_s": 1.0},
        {"stage": "convert.tensors", "completed": 25, "total": 100, "unit": "tensors", "elapsed_s": 6.0},
    ):
        progress.feed(row)
    stage = progress.stages["convert.tensors"]
    assert progress.current == "convert.tensors" and stage.fraction == 0.25
    assert stage.eta_s() == pytest.approx(15.0)  # 25 tensors in 5 s -> 75 more in 15 s
    progress.feed({"stage": "failed", "elapsed_s": 7.0})
    assert progress.outcome == "failed" and "failed" not in progress.stages


def test_job_runs_to_completion_with_events_and_envelope(tmp_path: Path, fake: Path) -> None:
    jobs = manager(tmp_path, fake, steps=3, delay=0.01)
    job = jobs.start(["calibrate", "--config", "x.yaml"])
    assert job.command == "calibrate" and (job.path / jobsvc.RECORD).is_file()
    wait(lambda: jobs.refresh(job).status != "running")
    assert job.status == "completed" and job.exit_code == 0
    assert job.envelope() == {"schema_version": 1, "command": "fake", "exit_code": 0, "result": {}, "error": None}
    progress = jobsvc.Progress()
    for row in jobsvc.EventTail(job.events_path).read():
        progress.feed(row)
    assert progress.stages["fake.steps"].done and progress.outcome == "completed"
    assert job.stderr_tail()[-1] == "step 3"
    assert [j.id for j in jobs.jobs()] == [job.id]


def test_cancel_interrupts_cleanly_and_reports_130(tmp_path: Path, fake: Path) -> None:
    jobs = manager(tmp_path, fake, steps=200, delay=0.05)
    job = jobs.start(["calibrate"])
    wait(lambda: len(jobsvc.EventTail(job.events_path).read()) >= 3)
    jobs.cancel(job)
    assert job.cancel_requested
    wait(lambda: jobs.refresh(job).status != "running")
    assert job.status == "interrupted" and job.exit_code == 130
    assert job.envelope()["error"] == "KeyboardInterrupt"  # type: ignore[index]


def test_force_stop_kills(tmp_path: Path, fake: Path) -> None:
    jobs = manager(tmp_path, fake, steps=200, delay=0.05)
    job = jobs.start(["calibrate"])
    wait(lambda: job.events_path.exists())
    jobs.cancel(job, force=True)
    wait(lambda: jobs.refresh(job).status != "running")
    assert job.status in {"failed", "interrupted"} and job.exit_code != 0


def test_a_job_from_an_earlier_session_is_read_from_its_record(tmp_path: Path) -> None:
    path = tmp_path / "jobs" / "20260101-000000-abcd"
    path.mkdir(parents=True)
    job = jobsvc.Job("20260101-000000-abcd", ["export"], str(tmp_path), time.time() - 60, pid=None, path=path)
    job.save()
    (path / jobsvc.STDOUT).write_text(json.dumps({"exit_code": 2, "error": "bad"}) + "\n", encoding="utf-8")
    (loaded,) = jobsvc.JobManager(tmp_path / "jobs").jobs()
    assert loaded.status == "failed" and loaded.exit_code == 2 and loaded.finished_at is not None


def test_real_cli_child_keeps_the_machine_contract(tmp_path: Path) -> None:
    jobs = jobsvc.JobManager(tmp_path / "jobs")
    job = jobs.start(["doctor"], cwd=ROOT)
    wait(lambda: jobs.refresh(job).status != "running", timeout=60)
    envelope = job.envelope()
    assert job.exit_code == 0 and envelope is not None
    assert envelope["command"] == "doctor" and envelope["result"]["backends"][0]["name"] == "ternary"
    stages = [row["stage"] for row in jobsvc.EventTail(job.events_path).read()]
    assert stages[0] == "started" and stages[-1] == "completed"


def test_preview_runs_the_commands_own_dry_run() -> None:
    code, text = jobsvc.preview(["quantize", "--config", "configs/gemma4-e2b.yaml"], cwd=ROOT)
    assert code == 0 and "dry-run" in text and "backend: ternary" in text
    code, text = jobsvc.preview(["quantize", "--config", "configs/missing.yaml"], cwd=ROOT)
    assert code == 2


def make_run(root: Path, name: str, status: str, marker: str | None = None, accepted: bool | None = None) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps({"status": status, "run_id": name[-4:]}), encoding="utf-8")
    (path / "model.json").write_text(json.dumps({"id": "org/model"}), encoding="utf-8")
    if marker:
        (path / marker).write_text("{}", encoding="utf-8")
    if accepted is not None:
        (path / "acceptance.json").write_text(json.dumps({"accepted": accepted}), encoding="utf-8")
    return path


def test_runs_are_listed_with_kind_status_and_quality(tmp_path: Path) -> None:
    make_run(tmp_path, "2026-11-01_a", "completed", "benchmark.json")
    make_run(tmp_path, "2026-11-02_b", "completed", "compare.json", accepted=False)
    make_run(tmp_path, "2026-11-03_c", "failed")
    (tmp_path / "not-a-run").mkdir()
    runs = {run.name: run for run in runsvc.list_runs(tmp_path)}
    assert set(runs) == {"2026-11-01_a", "2026-11-02_b", "2026-11-03_c"}
    assert runs["2026-11-01_a"].kind == "benchmark" and runs["2026-11-01_a"].quality == "not_run"
    assert runs["2026-11-02_b"].kind == "compare" and runs["2026-11-02_b"].quality == "rejected"
    assert runs["2026-11-03_c"].status == "failed" and runs["2026-11-03_c"].model == "org/model"
    assert runsvc.list_runs(tmp_path / "missing") == []


def test_delete_run_guards(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = make_run(root, "2026-11-01_a", "completed")
    with pytest.raises(ValueError):
        runsvc.delete_run(root, root)  # the folder itself
    with pytest.raises(ValueError):
        runsvc.delete_run(root, tmp_path)  # outside
    (root / "plain").mkdir()
    with pytest.raises(ValueError, match="not a run"):
        runsvc.delete_run(root, root / "plain")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time;from filelock import FileLock\n"
            "with FileLock(sys.argv[1]):\n print('held', flush=True); time.sleep(30)",
            str(run / ".execution.lock"),
        ],
        stdout=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == b"held"
        with pytest.raises(ValueError, match="in use"):
            runsvc.delete_run(root, run)
    finally:
        holder.kill()
        holder.wait()
    with FileLock(str(tmp_path / "unrelated.lock")):
        runsvc.delete_run(root, run)
    assert not run.exists()
