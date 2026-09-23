"""Fresh-process search candidate execution. Only optimize invokes this module."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from openternary.config.schema import AppConfig
from openternary.experiment.compare import compare_runs
from openternary.experiment.metadata import write_json
from openternary.services.artifacts import canonical_hash, validate_artifact
from openternary.services.evaluation import benchmark, quality
from openternary.services.execution import convert_artifact
from openternary.services.optimize import QualityProfile


def execute(request: Path) -> None:
    payload = json.loads(request.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(payload["config"])
    profile = QualityProfile.model_validate(payload["profile"])
    folder = request.parent
    artifact = folder / "artifact"
    started = time.monotonic()
    conversion_config = config.model_copy(deep=True)
    # Snapshot conversion is CPU work; the original device applies to evaluation.
    conversion_config.device = "cpu"
    convert_artifact(Path(config.model.id), artifact, conversion_config)
    manifest = validate_artifact(artifact)
    import torch

    uses_gpu = config.device != "cpu" and torch.cuda.is_available()
    if uses_gpu:
        torch.cuda.reset_peak_memory_stats()
    quality_report = quality(
        config,
        Path(profile.dataset),
        snapshot_path=artifact,
        split="validation",
        max_length=profile.max_length,
        stride=profile.stride,
    )
    benchmark_report = benchmark(config, artifact, config.benchmark.suite)
    write_json(folder / "quality.json", quality_report)
    write_json(folder / "benchmark.json", benchmark_report)
    comparison = compare_runs(Path(profile.baseline), folder)
    write_json(folder / "acceptance.json", comparison["quality_gate"])
    measured_peaks = [
        report.get("resources", {}).get("inference_vram_bytes") for report in (quality_report, benchmark_report)
    ]
    known_peaks = [int(value) for value in measured_peaks if value is not None]
    vram = max(known_peaks) if known_peaks else int(torch.cuda.max_memory_allocated()) if uses_gpu else 0
    latency = float(benchmark_report["summary"]["avg_generation_latency_ms"])
    write_json(
        folder / "measurement.json",
        {
            "schema_version": 1,
            "artifact_reloaded": True,
            "quality_accepted": comparison["quality_gate"]["accepted"],
            "latency_ms": latency,
            "inference_vram_bytes": vram,
            "gpu_seconds": time.monotonic() - started if uses_gpu else 0,
            "artifact_fingerprint": manifest["manifest_fingerprint"],
            "protocol_fingerprint": canonical_hash(
                {
                    "quality": quality_report["protocol_fingerprint"],
                    "benchmark": benchmark_report["suite"]["fingerprint"],
                    "device": quality_report["model"]["actual_device"],
                }
            ),
        },
    )


if __name__ == "__main__":
    execute(Path(sys.argv[1]))
