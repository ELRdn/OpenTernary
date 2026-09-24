"""Reload and evaluate a saved layer-0 plus learned q1 hard ternary pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.acceptance import compare_quality_metrics
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan
from openternary.services.identity import snapshot_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--upstream-block-dir", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detail_path = args.output.with_suffix(".quality.json")
    if args.output.exists() or detail_path.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    manifest_sha = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    layer0_names = {item for item in rotations if item.startswith("model.language_model.layers.0.")}
    layer0_report_path = args.upstream_block_dir / "layer0.json"
    layer0_artifact = layer0_report_path.with_suffix(".safetensors")
    layer0_report = json.loads(layer0_report_path.read_text(encoding="utf-8"))
    layer0_sha = hashlib.sha256(layer0_artifact.read_bytes()).hexdigest()
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    candidate_artifact = args.candidate_report.with_suffix(".safetensors")
    candidate_sha = hashlib.sha256(candidate_artifact.read_bytes()).hexdigest()
    name = "model.language_model.layers.1.self_attn.q_proj"
    if (
        layer0_report.get("artifact_sha256") != layer0_sha
        or layer0_report.get("rotation_manifest_sha256") != manifest_sha
        or Path(layer0_report.get("source", "")).resolve() != source
        or set(layer0_report.get("selected_modules", [])) != layer0_names
        or candidate_report.get("artifact_sha256") != candidate_sha
        or candidate_report.get("upstream_artifact_sha256") != layer0_sha
        or candidate_report.get("rotation_manifest_sha256") != manifest_sha
        or Path(candidate_report.get("source", "")).resolve() != source
        or candidate_report.get("module") != name
    ):
        raise ValueError("candidate provenance mismatch")
    upstream = load_file(str(layer0_artifact), device="cpu")
    candidate = load_file(str(candidate_artifact), device="cpu")
    if set(candidate) != {"rotation", "weight", "codes", "scales"}:
        raise ValueError("unexpected candidate tensors")
    identity = torch.eye(128)
    if (
        candidate["rotation"].shape != (128, 128)
        or not torch.isfinite(candidate["rotation"]).all()
        or (candidate["rotation"].T @ candidate["rotation"] - identity).abs().max() > 1e-4
        or candidate["codes"].dtype != torch.int8
        or not ((candidate["codes"] >= -1) & (candidate["codes"] <= 1)).all()
    ):
        raise ValueError("invalid candidate rotation or codes")

    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    handles = []
    with torch.no_grad():
        for module_name in sorted(layer0_names | {name}):
            if module_name == name:
                tensors = candidate
                prefix = ""
                matrix = candidate["rotation"]
            else:
                tensors = upstream
                prefix = module_name + "."
                matrix = rotations[module_name]
            weight = tensors[prefix + "weight"]
            codes = tensors[prefix + "codes"]
            scales = tensors[prefix + "scales"]
            if (
                weight.dtype != torch.bfloat16
                or weight.shape != codes.shape
                or weight.shape[-1] % 128
                or scales.numel() != weight.numel() // 128
                or not ((codes >= -1) & (codes <= 1)).all()
                or not torch.isfinite(scales).all()
                or (scales < 0).any()
                or not torch.equal(
                    (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight).to(torch.bfloat16),
                    weight,
                )
            ):
                raise ValueError(f"invalid hard ternary weight: {module_name}")
            parameters[module_name + ".weight"].copy_(weight.to("cuda:0"))
            matrix = matrix.to("cuda:0")

            def rotate_input(
                _module: torch.nn.Module,
                values: tuple[torch.Tensor, ...],
                *,
                rotation: torch.Tensor = matrix,
            ) -> tuple[torch.Tensor, ...]:
                transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), rotation)
                return (transformed.to(values[0].dtype), *values[1:])

            handles.append(model.get_submodule(module_name).register_forward_pre_hook(rotate_input))
    quality = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split=args.split, max_length=128, stride=64
    )
    baseline = json.loads(args.baseline_quality.read_text(encoding="utf-8"))
    if (
        quality["protocol_fingerprint"] != baseline["protocol_fingerprint"]
        or quality["dataset_fingerprint"] != baseline["dataset_fingerprint"]
        or snapshot_identity(source) != baseline["identity"]
        or baseline["protocol"]["split"] != args.split
    ):
        raise ValueError("quality protocol or data mismatch")
    gate = compare_quality_metrics(baseline["summary"], quality["summary"])
    result = {
        "status": "saved_eight_target_quality_only",
        "source": str(source),
        "split": args.split,
        "candidate_artifact_sha256": candidate_sha,
        "upstream_artifact_sha256": layer0_sha,
        "source_identity": baseline["identity"],
        "baseline_summary": baseline["summary"],
        "candidate_summary": quality["summary"],
        "quality_gate": gate,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for handle in handles:
        handle.remove()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
