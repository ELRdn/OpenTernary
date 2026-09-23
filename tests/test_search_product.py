"""Search control tests use recorded measurements and text artifacts, never models."""

from __future__ import annotations

import json

import pytest

from openternary.config.loader import load_config
from openternary.config.schema import QuantizationConfig
from openternary.services.artifacts import write_manifest
from openternary.services.optimize import QualityProfile, SearchBudget, run_search


def test_worker_keeps_evaluation_device_separate_from_conversion(tmp_path, monkeypatch):
    import sys
    import types

    from openternary.services import search_worker

    config = load_config(cli_overrides={"device": "cuda", "model.id": str(tmp_path / "source")})
    profile = QualityProfile(dataset=str(tmp_path / "data"), baseline=str(tmp_path / "baseline"))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"config": config.model_dump(), "profile": profile.model_dump()}))
    devices = []
    monkeypatch.setattr(search_worker, "convert_artifact", lambda source, artifact, cfg: devices.append(cfg.device))
    monkeypatch.setattr(search_worker, "validate_artifact", lambda path: {"manifest_fingerprint": "a" * 64})

    def quality(cfg, *args, **kwargs):
        devices.append(cfg.device)
        return {"protocol_fingerprint": "b" * 64, "model": {"actual_device": "cuda"}}

    monkeypatch.setattr(search_worker, "quality", quality)
    monkeypatch.setattr(
        search_worker,
        "benchmark",
        lambda *a: {"summary": {"avg_generation_latency_ms": 1}, "suite": {"fingerprint": "c" * 64}},
    )
    monkeypatch.setattr(search_worker, "compare_runs", lambda *a: {"quality_gate": {"accepted": False}})
    cuda = types.SimpleNamespace(
        is_available=lambda: True, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 4096
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    search_worker.execute(request)
    assert devices == ["cpu", "cuda"]
    measurement = json.loads((tmp_path / "measurement.json").read_text())
    assert measurement["inference_vram_bytes"] == 4096
    assert measurement["quality_accepted"] is False


def inputs(tmp_path):
    from tests.test_cli_product import header_fixture

    source = header_fixture(tmp_path / "source")
    data = tmp_path / "data.json"
    data.write_text("{}")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "quality.json").write_text("{}")
    cfg = load_config(cli_overrides={"model.id": str(source), "device": "cpu"})
    profile = QualityProfile(dataset=str(data), baseline=str(baseline))
    candidates = [QuantizationConfig(group_size=group, scale_granularity="per_group") for group in (2, 4)]
    return cfg, profile, candidates


def fake_executor(calls, *, accepted=True, fail_first=False):
    def execute(config, profile, folder, timeout):
        calls.append(config.quantization.group_size)
        if fail_first and len(calls) == 1:
            raise RuntimeError("injected failure")
        artifact = folder / "artifact"
        artifact.mkdir()
        (artifact / "fixture.txt").write_text("fixture data")
        manifest = write_manifest(artifact, format="fixture", provenance={"scope": "unit_test"})
        return {
            "schema_version": 1,
            "artifact_reloaded": True,
            "quality_accepted": accepted,
            "latency_ms": float(config.quantization.group_size),
            "inference_vram_bytes": 100,
            "gpu_seconds": 0,
            "artifact_fingerprint": manifest["manifest_fingerprint"],
            "protocol_fingerprint": "a" * 64,
        }

    return execute


def test_budget_stop_and_resume_avoid_completed_candidates(tmp_path):
    cfg, profile, candidates = inputs(tmp_path)
    calls = []
    output = tmp_path / "search"
    result = run_search(
        cfg, profile, candidates, output, SearchBudget(max_candidates=1), target_vram=200, executor=fake_executor(calls)
    )
    assert result["status"] == "budget_exhausted" and result["best"] is None
    resumed = run_search(
        cfg,
        profile,
        candidates,
        output,
        SearchBudget(max_candidates=3),
        target_vram=200,
        resume=True,
        executor=fake_executor(calls),
    )
    assert calls == [2, 4]
    assert resumed["status"] == "completed"
    assert resumed["best"] == resumed["candidates"][0]["folder"]


def test_no_feasible_candidate_never_publishes_best(tmp_path):
    cfg, profile, candidates = inputs(tmp_path)
    result = run_search(
        cfg,
        profile,
        candidates,
        tmp_path / "search",
        SearchBudget(),
        target_vram=200,
        executor=fake_executor([], accepted=False),
    )
    assert result["status"] == "no_feasible_candidate"
    assert result["best"] is None


def test_partial_failure_is_not_cache_hit_and_changes_invalidate_resume(tmp_path):
    cfg, profile, candidates = inputs(tmp_path)
    calls = []
    output = tmp_path / "search"
    first = run_search(
        cfg,
        profile,
        candidates,
        output,
        SearchBudget(),
        target_vram=200,
        executor=fake_executor(calls, fail_first=True),
    )
    assert first["candidates"][0]["status"] == "failed"
    run_search(
        cfg, profile, candidates, output, SearchBudget(), target_vram=200, resume=True, executor=fake_executor(calls)
    )
    assert calls == [2, 4, 2]
    from pathlib import Path

    Path(profile.dataset).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="contract changed"):
        run_search(
            cfg,
            profile,
            candidates,
            output,
            SearchBudget(),
            target_vram=200,
            resume=True,
            executor=fake_executor(calls),
        )


def test_interruption_is_recorded_and_final_test_split_rejected(tmp_path):
    cfg, profile, candidates = inputs(tmp_path)
    with pytest.raises(ValueError):
        QualityProfile(dataset=profile.dataset, baseline=profile.baseline, split="test")

    def interrupted(*args):
        raise KeyboardInterrupt

    output = tmp_path / "search"
    with pytest.raises(KeyboardInterrupt):
        run_search(cfg, profile, candidates, output, SearchBudget(), target_vram=200, executor=interrupted)
    journal = json.loads((output / "search.json").read_text())
    assert journal["status"] == "interrupted"
    assert journal["best"] is None


def test_tampered_cached_artifact_is_rejected(tmp_path):
    cfg, profile, candidates = inputs(tmp_path)
    output = tmp_path / "search"
    result = run_search(cfg, profile, candidates, output, SearchBudget(), target_vram=200, executor=fake_executor([]))
    (output / result["best"] / "artifact/fixture.txt").write_text("corruption")
    with pytest.raises(ValueError, match="inventory"):
        run_search(
            cfg, profile, candidates, output, SearchBudget(), target_vram=200, resume=True, executor=fake_executor([])
        )
