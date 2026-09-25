"""Save fixed signed Hadamard hard-G128 tensors for selected Gemma 4 projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import fixed_signs_for_module, hadamard_segments, signed_hadamard_last_dim
from openternary.services.identity import snapshot_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--targets-manifest", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(35), required=True)
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument("--block", type=int, choices=(128, 256, 512, 1024), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.output.with_suffix(".safetensors")
    if args.output.exists() or artifact.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    manifest_raw = args.targets_manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("schema_version") != 1
        or Path(manifest.get("source", "")).resolve() != source
        or len(manifest.get("rotations", {})) != 205
        or len(args.roles) != len(set(args.roles))
    ):
        raise ValueError("canonical target manifest or role list mismatch")
    names = sorted(
        name
        for name in manifest["rotations"]
        if name.startswith(f"model.language_model.layers.{args.layer}.") and name.rsplit(".", 1)[-1] in args.roles
    )
    if len(names) != len(args.roles):
        raise ValueError("requested roles do not match canonical targets")
    tensors: dict[str, torch.Tensor] = {}
    modules = []
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
        for name in names:
            weight = checkpoint.get_tensor(name + ".weight").to(args.device)
            if weight.ndim != 2 or weight.shape[-1] % 128:
                raise ValueError(f"unsupported G128 Linear: {name}")
            signs = fixed_signs_for_module(name, weight.shape[-1], args.seed, weight.device)
            rotated = signed_hadamard_last_dim(weight.float(), args.block, signs)
            result = quantize_groupwise(rotated, 128)
            hard = dequantize_groupwise(result).to(torch.bfloat16)
            if not torch.equal(
                (result.codes.float().reshape(-1, 128) * result.scales.reshape(-1, 1))
                .reshape_as(hard)
                .to(torch.bfloat16),
                hard,
            ):
                raise ValueError(f"hard ternary reconstruction mismatch: {name}")
            tensors[name + ".weight"] = hard.cpu().contiguous()
            tensors[name + ".codes"] = result.codes.cpu().contiguous()
            tensors[name + ".scales"] = result.scales.cpu().contiguous()
            modules.append(
                {
                    "name": name,
                    "shape": list(weight.shape),
                    "segments": list(hadamard_segments(weight.shape[-1], args.block)),
                    "weight_mse": float((rotated - hard.float()).square().mean().item()),
                    "zero_fraction": float((result.codes == 0).float().mean().item()),
                }
            )
            print(f"[materialize] {len(modules)}/{len(names)} {name}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(artifact))
    report = {
        "schema_version": 1,
        "status": "fixed_signed_hadamard_hard_g128_roles",
        "source": str(source),
        "source_identity": snapshot_identity(source),
        "targets_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "layer": args.layer,
        "roles": sorted(args.roles),
        "modules": modules,
        "block": args.block,
        "seed": args.seed,
        "group_size": 128,
        "quantization_device": args.device,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "modules"}), flush=True)


if __name__ == "__main__":
    main()
