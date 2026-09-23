"""Shared evaluation entry points and evidence-aware multi-run comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openternary.config.schema import AppConfig
from openternary.experiment.compare import compare_runs


def benchmark(config: AppConfig, snapshot_path: Path, suite: str) -> dict[str, Any]:
    from openternary.benchmark.runner import run_benchmark

    return run_benchmark(config, snapshot_path=snapshot_path, suite=suite)


def quality(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from openternary.benchmark.quality_runner import run_quality_benchmark

    return run_quality_benchmark(*args, **kwargs)


def resource_observations(path: Path) -> dict[str, Any]:
    source = path if path.is_dir() else path.parent
    metrics = source / "metrics.json"
    measured: dict[str, Any] = {}
    if metrics.is_file():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("resources"), dict):
            measured = data["resources"]
    return {
        "disk_bytes": sum(p.stat().st_size for p in source.rglob("*") if p.is_file() and not p.is_symlink()),
        **{
            key: measured.get(key)
            for key in (
                "load_ram_bytes",
                "load_vram_bytes",
                "inference_ram_bytes",
                "inference_vram_bytes",
                "latency_ms",
                "tokens_per_second",
            )
        },
    }


def compare_many(baseline: Path, candidates: list[Path]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("comparison requires at least one candidate")
    paths = [baseline.resolve(), *(p.resolve() for p in candidates)]
    if len(set(paths)) != len(paths):
        raise ValueError("comparison inputs must be distinct")
    pairs = []
    for candidate in candidates:
        if (baseline / "evaluation.json").exists() or (candidate / "evaluation.json").exists():
            pairs.append(compare_diffusion(baseline, candidate))
        else:
            pairs.append(compare_runs(baseline, candidate))
    accepted = all(pair.get("quality_gate", {}).get("accepted") is True for pair in pairs if pair.get("quality_gate"))
    if any(not pair.get("quality_gate") for pair in pairs):
        accepted = False
    return {
        "schema_version": 1,
        "baseline": str(baseline),
        "comparisons": pairs,
        "resources": {str(path): resource_observations(path) for path in [baseline, *candidates]},
        "quality_acceptance": {"accepted": accepted, "basis": "validated quality reports only"},
        "speed_ranking": None,
        "speed_ranking_reason": "timing protocol/hardware parity not established",
    }


def compare_diffusion(baseline: Path, candidate: Path) -> dict[str, Any]:
    import math

    from openternary.services.artifacts import canonical_hash, file_hash, member_path

    reports = []
    for root in (baseline, candidate):
        report = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        if (
            not isinstance(report, dict)
            or report.get("report_schema_version") != 1
            or report.get("family") != "diffusion"
        ):
            raise ValueError("comparison requires diffusion evaluation schema 1 on both sides")
        if report.get("protocol_fingerprint") != canonical_hash(report.get("protocol")):
            raise ValueError("diffusion protocol fingerprint mismatch")
        rows = report.get("results")
        if not isinstance(rows, list) or not rows:
            raise ValueError("diffusion comparison requires non-empty image results")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("file"), str):
                raise ValueError("invalid diffusion image record")
            if file_hash(member_path(root, row["file"])) != row.get("sha256"):
                raise ValueError("diffusion image hash mismatch")
            value = row.get("latency_ms")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("diffusion latency must be positive and finite")
        reports.append(report)
    if reports[0]["protocol_fingerprint"] != reports[1]["protocol_fingerprint"]:
        raise ValueError("diffusion prompt/seed/scheduler/steps/resolution/runtime protocol mismatch")
    cases = [[(row.get("id"), row.get("prompt"), row.get("seed")) for row in report["results"]] for report in reports]
    if cases[0] != cases[1] or len(set(cases[0])) != len(cases[0]):
        raise ValueError("diffusion case inventory mismatch")
    return {
        "family": "diffusion",
        "protocol_match": True,
        "baseline_dir": str(baseline),
        "quantized_dir": str(candidate),
        "result_observation": "image_proxy_only",
        "result_is_pass_fail": False,
        "quality_gate": {"accepted": False, "status": "human_review_required"},
        "latency_ms": [
            sum(row["latency_ms"] for row in report["results"]) / len(report["results"]) for report in reports
        ],
    }
