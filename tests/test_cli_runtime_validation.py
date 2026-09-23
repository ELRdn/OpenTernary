"""Failure, runtime, and tensor contracts; no pretrained weights."""

from __future__ import annotations

import errno
import json
import subprocess
import sys

import psutil
import pytest

from openternary.services.process import run_process


@pytest.mark.parametrize("interrupt", [False, True])
def test_process_deadline_and_interrupt_reap_owned_descendants(tmp_path, interrupt):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)'])\n"
        "Path(sys.argv[1]).write_text(str(p.pid))\ntime.sleep(90)\n"
    )

    def check():
        if interrupt and pid_file.exists():
            raise KeyboardInterrupt

    with (tmp_path / "worker.log").open("w") as log, pytest.raises(KeyboardInterrupt if interrupt else TimeoutError):
        run_process([sys.executable, str(script), str(pid_file)], timeout=20, stdout=log, poll_check=check)
    assert pid_file.exists()
    child = int(pid_file.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_run_lock_blocks_a_second_process(tmp_path):
    from openternary.experiment.run import run_lock

    folder = tmp_path / "run"
    folder.mkdir()
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; from openternary.experiment.run import run_lock; "
        "c=run_lock(Path(__import__('sys').argv[1])); c.__enter__()",
        str(folder),
    ]
    with run_lock(folder):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "lock" in result.stderr.lower()
    with run_lock(folder):
        pass


def test_explicit_cpu_runtime_probe_and_metadata_only_doctor(monkeypatch):
    from typer.testing import CliRunner

    import openternary.services.runtime_probe as runtime
    from openternary.cli.main import app

    monkeypatch.setattr(runtime, "probe_runtime", lambda *a: pytest.fail("metadata doctor ran a kernel"))
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output


def test_tensor_validation_rejects_shape_dtype_and_target_forgery(tmp_path):
    pytest.importorskip("torch")
    from openternary.config.loader import load_config
    from openternary.services.artifacts import write_manifest
    from openternary.services.execution import convert_artifact
    from openternary.services.tensor_validation import validate_tensors
    from tests.test_artifact_product import tiny_source

    source = tmp_path / "source"
    tiny_source(source)
    artifact = tmp_path / "artifact"
    cfg = load_config(cli_overrides={"model.id": str(source), "device": "cpu"})
    convert_artifact(source, artifact, cfg)
    assert validate_tensors(artifact)["tensor_count"] == 3
    report_file = artifact / "quantization.json"
    original = json.loads(report_file.read_text())
    for key, value, message in (("shape", [999], "shape"), ("dtype", "torch.float16", "dtype")):
        data = json.loads(json.dumps(original))
        data["per_tensor"][0][key] = value
        report_file.write_text(json.dumps(data))
        write_manifest(artifact, format="safetensors", provenance={})
        with pytest.raises(ValueError, match=message):
            validate_tensors(artifact)
    report_file.write_text(json.dumps(original))
    write_manifest(artifact, format="safetensors", provenance={"targets": []})
    with pytest.raises(ValueError, match="targets"):
        validate_tensors(artifact)


def test_capacity_failure_preserves_source_and_releases_artifact_lock(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from openternary.config.loader import load_config
    from openternary.services.artifacts import file_hash
    from openternary.services.execution import convert_artifact
    from tests.test_artifact_product import tiny_source

    source = tmp_path / "source"
    tiny_source(source)
    before = {p.name: file_hash(p) for p in source.iterdir()}
    destination = tmp_path / "out"
    cfg = load_config(cli_overrides={"model.id": str(source), "device": "cpu"})

    class FullBackend:
        def convert(self, source, destination, config):
            destination.mkdir()
            (destination / "partial").write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "injected disk full")

    with monkeypatch.context() as patch:
        patch.setattr("openternary.services.execution.get_backend", lambda cfg: FullBackend())
        with pytest.raises(OSError, match="disk full"):
            convert_artifact(source, destination, cfg)
    assert not destination.exists()
    assert {p.name: file_hash(p) for p in source.iterdir()} == before
    convert_artifact(source, destination, cfg)


def test_identity_survives_relocation_and_detects_tokenizer_change(tmp_path):
    import shutil

    from openternary.services.artifacts import write_manifest
    from openternary.services.identity import snapshot_identity

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    (a / "model.safetensors").write_bytes(b"fixture")
    (a / "tokenizer.json").write_text('{"fixture":1}')
    identity = snapshot_identity(a)
    shutil.copytree(a, b)
    assert snapshot_identity(b) == identity
    write_manifest(b, format="safetensors", provenance={"identity": identity})
    (b / "tokenizer.json").write_text('{"fixture":2}')
    write_manifest(b, format="safetensors", provenance={"identity": identity})
    changed = snapshot_identity(b)
    assert changed["source_fingerprint"] == identity["source_fingerprint"]
    assert changed["interface_fingerprint"] != identity["interface_fingerprint"]


@pytest.mark.parametrize("scope,accepted", [("model", True), ("synthetic", False), ("unknown", False)])
def test_quality_schema3_requires_matched_non_synthetic_evidence(tmp_path, scope, accepted):
    from openternary.experiment.compare import compare_runs
    from openternary.services.artifacts import canonical_hash
    from tests.test_cli_compare import _quality_v2_payload

    payload = _quality_v2_payload(
        {"general_ppl": 10.0, "japanese_ppl": 12.0, "instruction_score": 100.0, "collapse_count": 0}
    )
    payload.update(
        report_schema_version=3,
        identity={
            "status": "resolved",
            "scope": scope,
            "source_fingerprint": "a" * 64,
            "interface_files": {"tokenizer.json": "b" * 64},
            "interface_fingerprint": canonical_hash({"tokenizer.json": "b" * 64}),
        },
        measurement_environment={"actual_device": "cuda:0", "packages": {"torch": "test-double"}},
    )
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for path in (baseline, candidate):
        path.mkdir()
        content = json.loads(json.dumps(payload))
        if path == baseline:
            control = _quality_v2_payload(
                {"general_ppl": 12.0, "japanese_ppl": 14.0, "instruction_score": 100.0, "collapse_count": 0}
            )
            content.update(summary=control["summary"], perplexity=control["perplexity"])
        (path / "quality.json").write_text(json.dumps(content))
    assert compare_runs(baseline, candidate)["quality_gate"]["accepted"] is accepted
    payload["identity"]["source_fingerprint"] = "c" * 64
    (candidate / "quality.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source_fingerprint mismatch"):
        compare_runs(baseline, candidate)


def test_schema2_remains_observation_only(tmp_path):
    from openternary.experiment.compare import compare_runs
    from tests.test_cli_compare import _quality_v2_payload

    payload = _quality_v2_payload(
        {"general_ppl": 10.0, "japanese_ppl": 12.0, "instruction_score": 100.0, "collapse_count": 0}
    )
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "quality.json").write_text(json.dumps(payload))
    assert compare_runs(tmp_path / "a", tmp_path / "b")["quality_gate"]["accepted"] is False


@pytest.mark.parametrize("accepted", [True, False])
def test_search_control_fixture_in_real_process_selects_and_resumes(tmp_path, accepted):
    from openternary.services.optimize import SearchBudget, run_search
    from tests.test_search_product import inputs

    cfg, profile, candidates = inputs(tmp_path)
    script = tmp_path / "controlled_worker.py"
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "from openternary.services.artifacts import write_manifest\n"
        "folder=Path(sys.argv[1]); artifact=folder/'artifact'; artifact.mkdir()\n"
        "(artifact/'control.txt').write_text('CONTROL FIXTURE, NOT MODEL QUALITY')\n"
        "m=write_manifest(artifact,format='fixture',provenance={'scope':'controlled_worker'})\n"
        "r={'schema_version':1,'artifact_reloaded':True,'quality_accepted':sys.argv[2]=='True',"
        "'latency_ms':float(sys.argv[3]),'inference_vram_bytes':0,'gpu_seconds':0,"
        "'artifact_fingerprint':m['manifest_fingerprint'],'protocol_fingerprint':'a'*64}\n"
        "(folder/'measurement.json').write_text(json.dumps(r))\n"
    )
    calls = []

    def execute(config, profile, folder, timeout):
        calls.append(folder)
        with (folder / "worker.log").open("w") as log:
            run_process(
                [sys.executable, str(script), str(folder), str(accepted), str(config.quantization.group_size)],
                timeout=timeout,
                stdout=log,
            )
        return json.loads((folder / "measurement.json").read_text())

    output = tmp_path / "search"
    initial = run_search(
        cfg, profile, candidates, output, SearchBudget(max_candidates=1), target_vram=100, executor=execute
    )
    assert initial["status"] == "budget_exhausted"
    result = run_search(
        cfg, profile, candidates, output, SearchBudget(max_candidates=3), target_vram=100, executor=execute, resume=True
    )
    assert len(calls) == 2
    assert result["status"] == ("completed" if accepted else "no_feasible_candidate")
    assert bool(result["best"]) is accepted
    measured = output / result["candidates"][0]["folder"] / "artifact" / "control.txt"
    measured.write_text("tampered")
    with pytest.raises(ValueError, match="hash|inventory|integrity"):
        run_search(
            cfg,
            profile,
            candidates,
            output,
            SearchBudget(max_candidates=3),
            target_vram=100,
            executor=execute,
            resume=True,
        )
