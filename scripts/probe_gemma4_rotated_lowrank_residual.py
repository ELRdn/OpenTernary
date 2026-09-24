"""Measure how much hard rotated ternary weight error a low-rank residual can capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--save-rank", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_output = args.output.with_suffix(".safetensors")
    if args.output.exists() or (args.save_rank and artifact_output.exists()):
        raise FileExistsError(args.output)
    if args.save_rank is not None and args.save_rank not in (16, 64, 128, 256):
        raise ValueError("save-rank must be 16, 64, 128, or 256")
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    names = sorted(name for name in rotations if name.startswith(f"model.language_model.layers.{args.layer}."))
    if len(names) != (7 if args.layer < 15 else 5):
        raise ValueError("incomplete layer target set")
    report_path = args.block_dir / f"layer{args.layer}.json"
    artifact_path = report_path.with_suffix(".safetensors")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if report.get("artifact_sha256") != digest or set(report.get("selected_modules", [])) != set(names):
        raise ValueError("block provenance mismatch")
    results = {}
    payload = {}
    with (
        safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_file,
        safe_open(artifact_path, framework="pt", device="cpu") as block_file,
        torch.no_grad(),
    ):
        for name in names:
            weight = source_file.get_tensor(name + ".weight").to("cuda:0", dtype=torch.float32)
            rotation = rotations[name].to("cuda:0")
            rotated = apply_block_rotation(hadamard_last_dim(weight, 128), rotation)
            hard = block_file.get_tensor(name + ".weight").to("cuda:0", dtype=torch.float32)
            error = rotated - hard
            total = float(error.square().sum().item())
            torch.manual_seed(42)
            u, singular, v = torch.svd_lowrank(error, q=min(288, min(error.shape) - 1), niter=2)
            fractions = {
                str(rank): float(singular[:rank].square().sum().item()) / total
                for rank in (16, 64, 128, 256)
                if rank <= singular.numel()
            }
            if args.save_rank is not None:
                rank = min(args.save_rank, singular.numel())
                left = (u[:, :rank] * singular[:rank]).to(torch.bfloat16).contiguous()
                right = v[:, :rank].T.to(torch.bfloat16).contiguous()
                payload[name + ".left"] = left.cpu()
                payload[name + ".right"] = right.cpu()
                corrected_error = error - left.float() @ right.float()
                fractions["saved_bf16"] = 1 - float(corrected_error.square().sum().item()) / total
            results[name] = {
                "shape": list(weight.shape),
                "weight_error_energy": total,
                "captured_energy_fraction": fractions,
                "residual_params": {str(rank): rank * sum(weight.shape) for rank in (16, 64, 128, 256)},
                "saved_rank": rank if args.save_rank is not None else None,
            }
            del weight, rotated, hard, error, u, singular, v
    artifact_sha256 = None
    if args.save_rank is not None:
        save_file(payload, str(artifact_output))
        reloaded = load_file(str(artifact_output), device="cpu")
        if reloaded.keys() != payload.keys() or any(
            not torch.equal(reloaded[key], value) for key, value in payload.items()
        ):
            raise ValueError("saved low-rank factors did not reload exactly")
        artifact_sha256 = hashlib.sha256(artifact_output.read_bytes()).hexdigest()
    result = {
        "source": str(source),
        "layer": args.layer,
        "block_artifact_sha256": digest,
        "svd_method": "torch.svd_lowrank q=min(288,min(shape)-1) niter=2 seed=42",
        "save_rank": args.save_rank,
        "artifact_sha256": artifact_sha256,
        "modules": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: value["captured_energy_fraction"] for name, value in results.items()}), flush=True)


if __name__ == "__main__":
    main()
