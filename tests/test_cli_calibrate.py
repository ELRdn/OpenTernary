"""CLI calibrate tests."""

import pathlib

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")  # noqa: F401, E402

from openternary.cli.main import app  # noqa: E402

runner = CliRunner()


def test_calibrate_dry_run_no_filesystem() -> None:
    import shutil
    import uuid

    tmp = pathlib.Path.cwd() / f"test_cli_{uuid.uuid4().hex[:6]}"
    out = tmp / "dry"
    result = runner.invoke(app, ["calibrate", "--dry-run", "--output", str(out)])
    try:
        assert result.exit_code == 0
        assert "calibrate: dry-run" in result.output
        assert "Target modules" in result.output
        assert "Dataset probe:" in result.output
        assert not out.exists()  # no run created
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_calibrate_dry_run_wiki_no_fallback_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import sys
    import uuid

    # make datasets import fail
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    orig_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    tmp = pathlib.Path.cwd() / f"test_cli_wiki_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cfg = tmp / "cfg.yaml"
        cfg.write_text(
            "calibration:\n  enabled: true\n  dataset: wiki-tiny\n  allow_dataset_fallback: false\n", encoding="utf-8"
        )
        result = runner.invoke(app, ["calibrate", "--config", str(cfg), "--dry-run"])
        assert result.exit_code == 2
        assert "Dataset probe failed" in result.output
        # no run created
        assert not any(tmp.glob("runs*"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_calibrate_window_validation() -> None:
    import shutil
    import uuid

    tmp = pathlib.Path.cwd() / f"test_cli2_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        # per-block should be rejected in 4.1
        cfg = tmp / "bad.yaml"
        cfg.write_text("calibration:\n  enabled: true\n  window: per-block\n", encoding="utf-8")
        result = runner.invoke(app, ["calibrate", "--config", str(cfg)])
        assert result.exit_code == 2
        assert "per-layer" in result.output or "per-block" in result.output
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_calibrate_help() -> None:
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    assert "calibrate" in result.output.lower()


def test_calibrate_resume_reuses_existing() -> None:
    import json
    import shutil
    import uuid

    tmp = pathlib.Path.cwd() / f"test_resume_cli_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        # create initial run via CLI
        cfg = tmp / "calib.yaml"
        cfg.write_text(
            "calibration:\n  enabled: true\n  dataset: synthetic\n  num_samples: 4\n  seq_len: 16\n  steps: 3\n  checkpoint_interval: 2\n",
            encoding="utf-8",
        )
        out = tmp / "run"
        result = runner.invoke(app, ["calibrate", "--config", str(cfg), "--output", str(out)])
        assert result.exit_code == 0
        assert (out / "calibration.json").exists()
        # modify config to have more steps and resume
        cfg2 = tmp / "calib2.yaml"
        cfg2.write_text(
            "calibration:\n  enabled: true\n  dataset: synthetic\n  num_samples: 4\n  seq_len: 16\n  steps: 5\n  checkpoint_interval: 2\n",
            encoding="utf-8",
        )
        result2 = runner.invoke(app, ["calibrate", "--config", str(cfg2), "--output", str(out), "--resume"])
        assert result2.exit_code == 0
        # should not have created -001
        assert not (pathlib.Path(str(out) + "-001")).exists()
        assert (out / "artifacts" / "checkpoint" / "step_00004.pt").exists()
        data = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
        assert data["steps"] == 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        # also clean possible -001
        for p in pathlib.Path.cwd().glob("test_resume_cli_*"):
            import shutil as _sh

            _sh.rmtree(p, ignore_errors=True)


def test_calibrate_resume_no_checkpoint_fails() -> None:
    import shutil
    import uuid

    tmp = pathlib.Path.cwd() / f"test_resume_nockpt_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cfg = tmp / "calib.yaml"
        cfg.write_text(
            "calibration:\n  enabled: true\n  dataset: synthetic\n  num_samples: 4\n  seq_len: 16\n  steps: 2\n",
            encoding="utf-8",
        )
        out = tmp / "empty_run"
        out.mkdir(parents=True, exist_ok=True)
        (out / "artifacts").mkdir(parents=True, exist_ok=True)  # no checkpoint
        (out / "environment.json").write_text('{"run_id":"test"}', encoding="utf-8")
        result = runner.invoke(app, ["calibrate", "--config", str(cfg), "--output", str(out), "--resume"])
        assert result.exit_code == 2
        assert "no resumable checkpoint found" in result.output
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
