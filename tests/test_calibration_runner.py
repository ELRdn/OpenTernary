"""Runner synthetic tiny fixture."""

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from openternary.calibration.runner import run_calibration  # noqa: E402
from openternary.config.loader import load_config  # noqa: E402


def test_runner_synthetic_tiny() -> None:
    import shutil
    import uuid

    tmp = pathlib.Path.cwd() / f"test_runner_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cfg = load_config(
            cli_overrides={
                "calibration.enabled": True,
                "calibration.dataset": "synthetic",
                "calibration.num_samples": 4,
                "calibration.seq_len": 16,
                "calibration.steps": 5,
                "calibration.checkpoint_interval": 2,
            }
        )
        out = tmp / "calib_run"
        run_calibration(cfg, None, out)
        # Check artifacts
        assert (out / "calibration.json").exists()
        assert (out / "artifacts" / "checkpoint").exists()
        ckpts = list((out / "artifacts" / "checkpoint").glob("step_*.pt"))
        assert len(ckpts) >= 2  # interval 2 with 5 steps -> steps 1,3,4
        data = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
        assert data["best_loss"] < data["initial_loss"] - 1e-9
        assert data["final_loss"] < data["initial_loss"]
        assert data["mean_abs_scale_delta"] > 1e-9
        assert data["trainable_param_count"] > 0
        assert not data["contamination_train_smoke"]
        assert data["zero_group_exact"] is True
        # new fields for dataset abstraction
        assert data["requested_dataset"] == "synthetic"
        assert data["effective_dataset"] == "synthetic"
        assert data["dataset_fallback"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runner_wiki_tiny_fallback_true(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    import sys
    import uuid

    # mock datasets to be missing
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    orig_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    tmp = pathlib.Path.cwd() / f"test_wiki_fallback_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cfg = load_config(
            cli_overrides={
                "calibration.enabled": True,
                "calibration.dataset": "wiki-tiny",
                "calibration.allow_dataset_fallback": True,
                "calibration.num_samples": 4,
                "calibration.seq_len": 16,
                "calibration.steps": 2,
                "calibration.checkpoint_interval": 2,
            }
        )
        out = tmp / "calib_wiki"
        run_calibration(cfg, None, out)
        data = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
        assert data["requested_dataset"] == "wiki-tiny"
        assert data["effective_dataset"] == "synthetic"
        assert data["dataset_fallback"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runner_wiki_tiny_no_fallback_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    orig_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    cfg = load_config(
        cli_overrides={
            "calibration.enabled": True,
            "calibration.dataset": "wiki-tiny",
            "calibration.allow_dataset_fallback": False,
            "calibration.num_samples": 4,
            "calibration.seq_len": 16,
            "calibration.steps": 2,
        }
    )
    import pathlib as _pl

    tmp = _pl.Path.cwd() / "test_wiki_nofallback_dummy"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(ValueError, match="dataset wiki-tiny failed"):
            run_calibration(cfg, None, tmp / "out")
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_runner_resume() -> None:
    import shutil
    import uuid

    tmp = pathlib.Path.cwd() / f"test_resume_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cfg = load_config(
            cli_overrides={
                "calibration.enabled": True,
                "calibration.dataset": "synthetic",
                "calibration.num_samples": 4,
                "calibration.seq_len": 16,
                "calibration.steps": 5,
                "calibration.checkpoint_interval": 2,
            }
        )
        out = tmp / "calib_run2"
        # First run
        run_calibration(cfg, None, out)
        # Resume should pick latest and continue (but steps already 5, so resume does nothing extra)
        # Create a new run with more steps and resume
        cfg2 = load_config(
            cli_overrides={
                "calibration.enabled": True,
                "calibration.dataset": "synthetic",
                "calibration.num_samples": 4,
                "calibration.seq_len": 16,
                "calibration.steps": 7,
                "calibration.checkpoint_interval": 2,
            }
        )
        # Resume from previous out (simulate)
        run_calibration(cfg2, None, out, resume=True)
        # Should have continued from step 4 to 6
        assert (out / "artifacts" / "checkpoint" / "step_00006.pt").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
