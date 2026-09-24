"""Validation-only model screen from separately saved hard Gemma 4 blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
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
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--through-layer", type=int, default=34)
    parser.add_argument(
        "--layer1-role", choices=("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj")
    )
    parser.add_argument("--residual-report", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if not 0 <= args.through_layer < 35:
        raise ValueError("through-layer must be in [0, 35)")
    if args.layer1_role is not None and args.through_layer != 1:
        raise ValueError("layer1-role requires through-layer 1")
    detail_path = args.output.with_suffix(".quality.json")
    if args.output.exists() or detail_path.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    manifest_sha256 = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    weights: dict[str, torch.Tensor] = {}
    block_reports = []
    joint_layers = []
    for layer in range(args.through_layer + 1):
        report_path = args.block_dir / f"layer{layer}.json"
        artifact_path = report_path.with_suffix(".safetensors")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status")
            not in {
                "single_block_reconstruction_pilot_only",
                "upstream_conditioned_block_reconstruction_only",
                "joint_three_block_reconstruction_only",
                "global_logit_kd_scale_only_block",
                "global_logit_kd_scale_only_upstream_block",
                "global_logit_kd_hard_codes_block",
                "global_logit_kd_hard_codes_upstream_block",
                "dense_latent_block_reconstruction_only",
                "dense_latent_upstream_block_reconstruction_only",
            }
            or report.get("layer") != layer
            or Path(report.get("source", "")).resolve() != source
            or report.get("rotation_manifest_sha256") != manifest_sha256
            or report.get("bf16_replay_exact") is not True
        ):
            raise ValueError(f"block provenance mismatch: layer {layer}")
        if report["status"] in {
            "upstream_conditioned_block_reconstruction_only",
            "dense_latent_upstream_block_reconstruction_only",
            "global_logit_kd_scale_only_upstream_block",
            "global_logit_kd_hard_codes_upstream_block",
        } and report.get("upstream_artifact_sha256") != {
            str(row["layer"]): row["artifact_sha256"] for row in block_reports
        }:
            raise ValueError(f"upstream block chain mismatch: layer {layer}")
        if report["status"] == "joint_three_block_reconstruction_only" and (
            report.get("joint_window") != [0, 1, 2]
            or args.through_layer < 2
            or layer > 2
            or not (args.block_dir / "joint.json").is_file()
        ):
            raise ValueError(f"incomplete joint three-block candidate: layer {layer}")
        if report["status"] == "joint_three_block_reconstruction_only":
            joint_layers.append(layer)
        artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if report.get("artifact_sha256") != artifact_digest:
            raise ValueError(f"block artifact hash mismatch: layer {layer}")
        names = set(report["selected_modules"])
        expected = {name for name in rotations if name.startswith(f"model.language_model.layers.{layer}.")}
        if names != expected or len(names) != (7 if layer < 15 else 5):
            raise ValueError(f"incomplete block target set: layer {layer}")
        tensors = load_file(str(artifact_path), device="cpu")
        if set(tensors) != {
            name + suffix for name in names for suffix in (".weight", ".codes", ".scales", ".thresholds")
        }:
            raise ValueError(f"unexpected block artifact tensors: layer {layer}")
        for name in sorted(names):
            if name in weights:
                raise ValueError(f"duplicate block target: {name}")
            weight = tensors[name + ".weight"]
            codes = tensors[name + ".codes"]
            scales = tensors[name + ".scales"]
            thresholds = tensors[name + ".thresholds"]
            if (
                weight.dtype != torch.bfloat16
                or codes.dtype != torch.int8
                or weight.shape != codes.shape
                or weight.shape[-1] % 128
                or scales.numel() != weight.numel() // 128
                or thresholds.numel() != scales.numel()
                or not torch.isfinite(weight).all()
                or not torch.isfinite(scales).all()
                or not torch.isfinite(thresholds).all()
                or (scales < 0).any()
                or (thresholds <= 0).any()
                or (thresholds >= 1).any()
                or not ((codes >= -1) & (codes <= 1)).all()
            ):
                raise ValueError(f"invalid hard ternary block tensor: {name}")
            reconstructed = (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight)
            if not torch.equal(reconstructed.to(torch.bfloat16), weight):
                raise ValueError(f"hard codes and scales do not reproduce saved weight: {name}")
            if layer == 1 and args.layer1_role is not None and not name.endswith(f".{args.layer1_role}"):
                continue
            weights[name] = weight
        block_reports.append({"layer": layer, "artifact_sha256": artifact_digest, "best": report["best"]})

    if joint_layers:
        joint = json.loads((args.block_dir / "joint.json").read_text(encoding="utf-8"))
        if (
            joint_layers != [0, 1, 2]
            or joint.get("status") != "joint_three_block_reconstruction_only"
            or joint.get("rotation_manifest_sha256") != manifest_sha256
            or Path(joint.get("source", "")).resolve() != source
            or any(
                joint.get("artifact_sha256", {}).get(str(layer)) != block_reports[layer]["artifact_sha256"]
                for layer in joint_layers
            )
        ):
            raise ValueError("joint three-block artifact set mismatch")

    residual_metadata = None
    residual_runtime_factors = {}
    if args.residual_report is not None:
        residual_report = json.loads(args.residual_report.read_text(encoding="utf-8"))
        residual_artifact = args.residual_report.with_suffix(".safetensors")
        residual_digest = hashlib.sha256(residual_artifact.read_bytes()).hexdigest()
        residual_layer = residual_report.get("layer")
        residual_names = {name for name in weights if name.startswith(f"model.language_model.layers.{residual_layer}.")}
        if (
            residual_report.get("source") != str(source)
            or residual_report.get("artifact_sha256") != residual_digest
            or not isinstance(residual_layer, int)
            or not 0 <= residual_layer < len(block_reports)
            or not residual_names
            or residual_report.get("block_artifact_sha256") != block_reports[residual_layer]["artifact_sha256"]
            or set(residual_report.get("modules", {})) != residual_names
        ):
            raise ValueError("low-rank residual provenance mismatch")
        residual_tensors = load_file(str(residual_artifact), device="cpu")
        if set(residual_tensors) != {name + suffix for name in residual_names for suffix in (".left", ".right")}:
            raise ValueError("low-rank residual tensor set mismatch")
        residual_parameter_count = 0
        factorized_runtime = residual_report.get("runtime_mode") == "factorized"
        for name in sorted(residual_names):
            left = residual_tensors[name + ".left"]
            right = residual_tensors[name + ".right"]
            if (
                left.dtype != torch.bfloat16
                or right.dtype != torch.bfloat16
                or left.ndim != 2
                or right.ndim != 2
                or left.shape[1] != residual_report["modules"][name]["saved_rank"]
                or right.shape[0] != residual_report["modules"][name]["saved_rank"]
                or not 1 <= left.shape[1] <= residual_report["save_rank"]
                or left.shape[0] != weights[name].shape[0]
                or right.shape[1] != weights[name].shape[1]
                or not torch.isfinite(left).all()
                or not torch.isfinite(right).all()
            ):
                raise ValueError(f"invalid low-rank residual: {name}")
            if factorized_runtime:
                residual_runtime_factors[name] = (left.to("cuda:0"), right.to("cuda:0"))
            else:
                weights[name] = (weights[name].float() + left.float() @ right.float()).to(torch.bfloat16)
            residual_parameter_count += left.numel() + right.numel()
        residual_metadata = {
            "layer": residual_layer,
            "rank": residual_report["save_rank"],
            "artifact_sha256": residual_digest,
            "bf16_parameter_count": residual_parameter_count,
            "runtime_mode": "factorized" if factorized_runtime else "dense_sum_screen",
        }

    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    hooks = []
    hook_calls = dict.fromkeys(weights, 0)
    with torch.no_grad():
        for name, weight in sorted(weights.items()):
            weight_name = name + ".weight"
            if weight_name not in parameters or parameters[weight_name].shape != weight.shape:
                raise ValueError(f"model/block weight mismatch: {weight_name}")
            parameters[weight_name].copy_(weight.to("cuda:0"))
            rotation = rotations[name].to("cuda:0")

            def rotate_input(
                _module: torch.nn.Module,
                values: tuple[torch.Tensor, ...],
                *,
                module_name: str = name,
                matrix: torch.Tensor = rotation,
            ) -> tuple[torch.Tensor, ...]:
                hook_calls[module_name] += 1
                transformed = hadamard_last_dim(values[0].float(), 128)
                transformed = apply_block_rotation(transformed, matrix).to(values[0].dtype)
                return (transformed, *values[1:])

            hooks.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))
        for name, (left, right) in residual_runtime_factors.items():

            def add_residual(
                _module: torch.nn.Module,
                values: tuple[torch.Tensor, ...],
                original: torch.Tensor,
                *,
                matrix_left: torch.Tensor = left,
                matrix_right: torch.Tensor = right,
            ) -> torch.Tensor:
                correction = functional.linear(
                    functional.linear(values[0].float(), matrix_right.float()), matrix_left.float()
                )
                return (original.float() + correction).to(original.dtype)

            hooks.append(model.get_submodule(name).register_forward_hook(add_residual))

    candidate = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split="validation", max_length=128, stride=64
    )
    candidate["identity"] = snapshot_identity(source)
    elapsed_s = time.monotonic() - started
    peak_vram_allocated_bytes = torch.cuda.max_memory_allocated("cuda:0")
    for hook in hooks:
        hook.remove()
    if any(count == 0 for count in hook_calls.values()):
        raise ValueError("a saved ternary module was never called")
    candidate["research_candidate"] = {
        "status": "in_memory_validation_only",
        "method": "rotated_g128_hard_codes_and_scales",
        "rotation_manifest_sha256": manifest_sha256,
        "block_artifact_sha256": {str(row["layer"]): row["artifact_sha256"] for row in block_reports},
        "ternary_module_count": len(weights),
        "layer1_role": args.layer1_role,
        "saved_snapshot": False,
        "lowrank_residual": residual_metadata,
    }
    baseline = json.loads(args.baseline_quality.read_text(encoding="utf-8"))
    if baseline.get("report_schema_version") != 3 or candidate.get("report_schema_version") != 3:
        raise ValueError("quality comparison requires schema 3 reports")
    baseline_identity = baseline.get("identity", {})
    candidate_identity = candidate.get("identity", {})
    if (
        baseline_identity.get("status") != "resolved"
        or candidate_identity.get("status") != "resolved"
        or baseline_identity.get("scope") != "model"
        or candidate_identity.get("scope") != "model"
        or any(
            baseline_identity.get(key) != candidate_identity.get(key)
            for key in ("source_fingerprint", "interface_fingerprint")
        )
        or not baseline_identity.get("source_fingerprint")
        or not baseline_identity.get("interface_fingerprint")
    ):
        raise ValueError("baseline/candidate source or interface identity mismatch")
    if baseline.get("model", {}).get("revision") != candidate.get("model", {}).get("revision"):
        raise ValueError("baseline/candidate revision mismatch")
    if candidate["protocol_fingerprint"] != baseline["protocol_fingerprint"]:
        raise ValueError("baseline/candidate protocol mismatch")
    if candidate["dataset_fingerprint"] != baseline["dataset_fingerprint"]:
        raise ValueError("baseline/candidate data mismatch")
    gate = compare_quality_metrics(baseline["summary"], candidate["summary"])
    result = {
        "status": (
            "ternary_plus_bf16_lowrank_residual_in_memory_validation_only"
            if residual_metadata
            else "reloaded_block_artifacts_in_memory_validation_only"
        ),
        "source": str(source),
        "through_layer": args.through_layer,
        "ternary_module_count": len(weights),
        "layer1_role": args.layer1_role,
        "ternary_parameter_count": sum(weight.numel() for weight in weights.values()),
        "rotation_manifest_sha256": manifest_sha256,
        "block_reports": block_reports,
        "hook_calls": hook_calls,
        "baseline_summary": baseline["summary"],
        "candidate_summary": candidate["summary"],
        "quality_gate": gate,
        "lowrank_residual": residual_metadata,
        "elapsed_s": elapsed_s,
        "peak_vram_allocated_bytes": peak_vram_allocated_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps({"through_layer": args.through_layer, "candidate": candidate["summary"], "gate": gate}), flush=True
    )


if __name__ == "__main__":
    main()
