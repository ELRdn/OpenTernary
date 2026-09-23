"""CLI contracts tested without model construction, inference or network access."""

from __future__ import annotations

import json
import os
import pathlib
import struct
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from openternary.cli.main import app
from openternary.config.loader import load_config
from openternary.experiment.run import run_lock

runner = CliRunner()


def header_fixture(folder: pathlib.Path, model_type: str = "llama") -> pathlib.Path:
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    header = {
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "model.embed_tokens.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [16, 32]},
    }
    encoded = json.dumps(header).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 32)
    return folder


@pytest.mark.parametrize(
    "arguments",
    [
        ["doctor"],
        ["backends", "list"],
        ["backends", "info", "ternary"],
        ["plan"],
        ["plugins"],
        ["calibrate", "--dry-run"],
    ],
)
def test_json_is_one_versioned_document(arguments, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [*arguments, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["quality_acceptance"] == "not_run"
    assert not list(tmp_path.iterdir())


def test_core_commands_do_not_import_ml(tmp_path):
    # Fresh process with a hard import guard catches eager imports even when ML happens to be installed.
    script = """import sys,importlib.abc,json
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','diffusers','torchao','datasets'}:
   raise AssertionError('ML import: '+fullname)
sys.meta_path.insert(0,Block())
from typer.testing import CliRunner
from openternary.cli.main import app
for args in [['--help'],['doctor','--json'],['backends','list','--json'],['calibrate','--dry-run','--json']]:
 r=CliRunner().invoke(app,args)
 assert r.exit_code==0,(r.output,r.exception)
print('core-only OK')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_plan_conservative_adapter_and_no_writes(tmp_path):
    source = header_fixture(tmp_path / "日本語 source")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = runner.invoke(app, ["plan", str(source), "--json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)["result"]
    assert plan["adapter"] == "transformers"
    assert plan["target_count"] == 1
    assert plan["unknown"]
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeeed": 1},
        {"quantization.codebook": [0, 1]},
        {"calibration.method": "recon-threshold"},
        {"calibration.threshold_lr": 0},
        {"runtime.low_vram_typo": False},
    ],
)
def test_settings_reject_unknown_or_conflicting_values(overrides):
    with pytest.raises(ValueError):
        load_config(cli_overrides=overrides)


def test_missing_model_never_becomes_synthetic_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate", str(tmp_path / "absent"), "--output", str(tmp_path / "run"), "--json"])
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["execution"] == "failed"
    assert "Calibrate completed" not in result.output
    assert not list(tmp_path.rglob("calibration.json"))
    assert json.loads((tmp_path / "run/metrics.json").read_text())["status"] == "failed"


def test_empty_comparison_fails_and_records_failure(tmp_path):
    baseline, candidate, output = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    baseline.mkdir()
    candidate.mkdir()
    result = runner.invoke(app, ["compare", str(baseline), str(candidate), "--output", str(output), "--json"])
    assert result.exit_code == 2, result.output
    assert json.loads((output / "metrics.json").read_text())["status"] == "failed"


def test_legacy_quality_never_accepts(tmp_path):
    from openternary.experiment.compare import compare_runs

    report = {
        "summary": {"general_ppl": 2, "japanese_ppl": 2, "instruction_score": 100, "collapse_count": 0},
        "protocol_fingerprint": "old",
        "dataset_fingerprint": "old",
    }
    directories = [tmp_path / "baseline", tmp_path / "candidate"]
    for directory in directories:
        directory.mkdir()
        (directory / "quality.json").write_text(json.dumps(report), encoding="utf-8")
    result = compare_runs(*directories)
    assert result["quality_gate"]["accepted"] is False
    invocation = runner.invoke(
        app, ["compare", *map(str, directories), "--require-acceptance", "--json", "--output", str(tmp_path / "out")]
    )
    assert invocation.exit_code == 6
    assert json.loads(invocation.stdout)["quality_acceptance"]["accepted"] is False


def test_parallel_creation_across_working_directories(tmp_path):
    import openternary

    source = pathlib.Path(openternary.__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(source)}
    directories = [tmp_path / "cwd1", tmp_path / "cwd2"]
    for directory in directories:
        directory.mkdir()
    target = tmp_path / "shared" / "run"
    script = """import sys,json
from openternary.config.loader import load_config
import openternary.experiment.run as runs
runs.collect_environment=lambda config,run_id: {'run_id':run_id}
path,id=runs.create_run(load_config(cli_overrides={'output':sys.argv[1]}))
print(json.dumps({'path':str(path),'id':id}))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(target)],
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for directory in directories
    ]
    outputs = [p.communicate(timeout=30) for p in processes]
    assert all(p.returncode == 0 for p in processes), outputs
    results = [json.loads(out) for out, _ in outputs]
    assert len({row["path"] for row in results}) == 2
    for row in results:
        assert json.loads((pathlib.Path(row["path"]) / "environment.json").read_text())["run_id"] == row["id"]


def test_concurrent_resume_is_rejected_before_writes(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    script = """import sys
from openternary.experiment.run import run_lock
try:
 with run_lock(sys.argv[1]): raise AssertionError('lock was not exclusive')
except ValueError as e:
 print(str(e))
"""
    with run_lock(folder):
        process = subprocess.run(
            [sys.executable, "-c", script, str(folder)], text=True, capture_output=True, timeout=30
        )
    assert process.returncode == 0, process.stderr
    assert "already in use" in process.stdout
    assert not (folder / "metrics.json").exists()


def test_error_json_and_unsupported_export_have_no_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for args, expected in [
        (["doctor", "--invalid", "--json"], 2),
        (["export", "missing", "--format", "gguf", "--json"], 3),
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == expected, result.output
        assert json.loads(result.stdout)["error"]
    assert not list(tmp_path.iterdir())


def test_events_have_monotonic_sequence(tmp_path):
    events = tmp_path / "events.jsonl"
    result = runner.invoke(app, ["doctor", "--json", "--events-jsonl", str(events)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[-1]["stage"] == "completed"
    assert all(row["total"] is None for row in rows)


def test_dry_run_rejects_events_file(tmp_path):
    events = tmp_path / "events.jsonl"
    result = runner.invoke(app, ["calibrate", "--dry-run", "--events-jsonl", str(events), "--json"])
    assert result.exit_code == 2
    assert not events.exists()


@pytest.mark.parametrize("exception,expected", [(RuntimeError("metadata failed"), 1), (KeyboardInterrupt(), 130)])
def test_failure_before_initial_metrics_is_recorded(tmp_path, monkeypatch, exception, expected):
    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr("openternary.experiment.run.collect_environment", fail)
    output = tmp_path / "run"
    result = runner.invoke(app, ["inspect", "missing", "--output", str(output), "--json"])
    assert result.exit_code == expected, result.output
    payload = json.loads(result.stdout)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["exit_code"] == expected
    assert metrics["status"] == payload["execution"]
    assert metrics["error_message"]
