"""Run directory and metadata tests."""

import json
import pathlib
import shutil
import uuid

from openternary.config.loader import load_config
from openternary.experiment.run import create_run


def _tmp_base() -> pathlib.Path:
    p = pathlib.Path.cwd() / f"test_run_{uuid.uuid4().hex[:6]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_create_run_generates_expected_files() -> None:
    tmp = _tmp_base()
    try:
        cfg = load_config()
        run_dir, run_id = create_run(cfg, base_dir=tmp / "runs")
        assert run_dir.exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "environment.json").exists()
        assert (run_dir / "model.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "logs").exists()
        assert (run_dir / "artifacts").exists()
        env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        for key in ["run_id", "timestamp", "python_version", "git_commit", "seed", "hardware"]:
            assert key in env, f"missing key {key}"
        assert env["run_id"] == run_id
        assert "seed: 42" in (run_dir / "config.yaml").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_run_output_override() -> None:
    tmp = _tmp_base()
    try:
        cfg = load_config(cli_overrides={"output": str(tmp / "my-run")})
        run_dir, _ = create_run(cfg, base_dir=tmp / "runs")
        assert run_dir == tmp / "my-run"
        assert run_dir.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_create_run_collision_suffix() -> None:
    tmp = _tmp_base()
    try:
        cfg = load_config()
        run_dir1, _ = create_run(cfg, base_dir=tmp / "runs")
        run_dir2, _ = create_run(cfg, base_dir=tmp / "runs")
        assert run_dir1 != run_dir2
        assert run_dir2.name.endswith("-001") or run_dir2.name != run_dir1.name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_environment_json_has_uv_lock_hash() -> None:
    tmp = _tmp_base()
    try:
        cfg = load_config()
        run_dir, _ = create_run(cfg, base_dir=tmp / "runs")
        env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        assert "uv_lock_hash" in env
        assert "python_manager" in env
        assert env["python_manager"] == "uv"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
