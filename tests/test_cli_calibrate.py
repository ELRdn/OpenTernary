"""CLI calibrate tests."""

import hashlib
import json
import pathlib
import struct

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")  # noqa: F401, E402

from openternary.cli.main import app  # noqa: E402

runner = CliRunner()


@pytest.mark.parametrize(
    ("revision", "expected"),
    [
        (None, False),
        ("main", False),
        ("6befbaca7398925921802abd1f277b495b78b738", True),
    ],
)
def test_preflight_revision_pin_requires_commit_hash(revision, expected) -> None:
    from openternary.calibration.preflight import _revision_is_pinned

    assert _revision_is_pinned(revision) is expected


def test_preflight_revision_must_match_canonical_target() -> None:
    from openternary.calibration.preflight import CANONICAL_MODEL_REVISION, _revision_matches_canonical

    assert _revision_matches_canonical(CANONICAL_MODEL_REVISION) is True
    assert _revision_matches_canonical("0" * 40) is False


def test_preflight_canonical_inventory_and_weight_are_exact() -> None:
    from openternary.calibration.preflight import (
        CANONICAL_TARGET_NAMES,
        CANONICAL_WEIGHT_SHA256,
        _weight_matches_canonical,
    )

    assert len(CANONICAL_TARGET_NAMES) == 205
    assert len(set(CANONICAL_TARGET_NAMES)) == 205
    assert "model.language_model.layers.0.self_attn.k_proj.weight" in CANONICAL_TARGET_NAMES
    assert "model.language_model.layers.15.self_attn.k_proj.weight" not in CANONICAL_TARGET_NAMES
    assert _weight_matches_canonical([{"name": "model.safetensors", "sha256": CANONICAL_WEIGHT_SHA256}])
    assert not _weight_matches_canonical([{"name": "model.safetensors", "sha256": "0" * 64}])


def test_preflight_repo_contracts_do_not_depend_on_invocation_cwd(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openternary.calibration.preflight import _git_state, _protocol_files

    monkeypatch.chdir(tmp_path)

    protocol_files = _protocol_files()
    assert len(protocol_files) == 6
    assert all(len(item["sha256"]) == 64 for item in protocol_files)
    assert _git_state()["commit"] != "unknown"


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


def test_calibrate_preflight_writes_canonical_manifest(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import openternary.calibration.preflight as preflight_module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {"num_hidden_layers": 35},
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    target_names = [
        f"model.language_model.layers.{index}.{family}.{projection}.weight"
        for index in range(35)
        for family, projection in (
            [("mlp", "down_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj")]
            + [
                ("self_attn", name)
                for name in (["k_proj", "o_proj", "q_proj", "v_proj"] if index < 15 else ["o_proj", "q_proj"])
            ]
        )
    ]
    header = {
        name: {
            "dtype": "BF16",
            "shape": [1, 1],
            "data_offsets": [index * 2, index * 2 + 2],
        }
        for index, name in enumerate(target_names)
    }
    encoded = json.dumps(header).encode("utf-8")
    with (snapshot / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(b"\x00" * (205 * 2))
    monkeypatch.setattr(
        "openternary.calibration.preflight.CANONICAL_WEIGHT_SHA256",
        hashlib.sha256((snapshot / "model.safetensors").read_bytes()).hexdigest(),
    )
    inspected_model_ids: list[str] = []
    original_run_inspection = preflight_module.run_inspection

    def record_inspection_identity(config, **kwargs):
        inspected_model_ids.append(config.model.id)
        return original_run_inspection(config, **kwargs)

    monkeypatch.setattr(preflight_module, "run_inspection", record_inspection_identity)

    output = tmp_path / "preflight"
    result = runner.invoke(
        app,
        ["calibrate", str(snapshot), "--preflight", "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["acceptance_scope"] == "preflight"
    assert report["model"]["inspection_id"] == "google/gemma-4-E2B-it-qat-q4_0-unquantized"
    assert inspected_model_ids == [report["model"]["inspection_id"]]
    assert report["model"]["source_locator"] == str(snapshot.resolve())
    assert report["target_inventory"]["count"] == 205
    assert report["checks"]["canonical_target_inventory"] is True
    assert report["checks"]["canonical_weight_sha256"] is True
    assert report["source"]["revision"] == "6befbaca7398925921802abd1f277b495b78b738"
    assert all(len(item["sha256"]) == 64 for item in report["source"]["files"])
    assert "docs/plans/p0-p7-research-acceptance.md" in {item["path"] for item in report["protocol_files"]}
    assert "existing_artifacts" in report
    assert all("classification" in item for item in report["existing_artifacts"]["runs"])
    assert not (output / "artifacts/checkpoint").exists()


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
