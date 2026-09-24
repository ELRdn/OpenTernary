"""Compare AbsMean and alternating least-squares G128 scales after rotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file

from openternary.calibration.capture import DiskActivationCache
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim


def lloyd_reconstruct(weight: torch.Tensor, steps: int) -> torch.Tensor:
    grouped = weight.reshape(-1, 128)
    scale = grouped.abs().mean(dim=-1, keepdim=True)
    for _ in range(steps):
        active = grouped.abs() > scale / 2
        count = active.sum(dim=-1, keepdim=True)
        scale = torch.where(
            count > 0,
            (grouped.abs() * active).sum(dim=-1, keepdim=True) / count.clamp_min(1),
            0,
        )
    codes = torch.sign(grouped) * (grouped.abs() > scale / 2)
    return (codes * scale).reshape_as(weight)


def held_pairs(root: Path, module: str) -> tuple[torch.Tensor, torch.Tensor]:
    cache = DiskActivationCache(root)
    batches = [cache.load_valid(module, index) for index in range(cache.count_batches(module))]
    if not batches:
        raise ValueError(f"empty held cache: {module}")
    inputs = torch.cat([batch["input"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    teacher = torch.cat([batch["teacher_output"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    return inputs, teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--held-cache", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    rows = []
    with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as source_handle:
        for module in args.module:
            tensor_path = args.rotation_dir / f"{module}.safetensors"
            report = json.loads(tensor_path.with_suffix(".json").read_text(encoding="utf-8"))
            if report.get("module") != module or Path(report.get("source", "")).resolve() != args.source.resolve():
                raise ValueError(f"rotation provenance mismatch: {module}")
            rotation = load_file(str(tensor_path), device="cpu")["cayley_rotation"].to("cuda:0")
            weight = source_handle.get_tensor(module + ".weight").to("cuda:0", dtype=torch.float32)
            inputs, teacher = held_pairs(args.held_cache, module)
            transformed_weight = apply_block_rotation(hadamard_last_dim(weight, 128), rotation)
            transformed_inputs = apply_block_rotation(hadamard_last_dim(inputs, 128), rotation)
            candidates = {
                "absmean": dequantize_groupwise(quantize_groupwise(transformed_weight, 128)),
                "lloyd": lloyd_reconstruct(transformed_weight, args.steps),
            }
            for method, reconstructed in candidates.items():
                output = functional.linear(transformed_inputs, reconstructed)
                rows.append(
                    {
                        "module": module,
                        "method": method,
                        "steps": args.steps if method == "lloyd" else 0,
                        "held_output_rel_mse": float(
                            ((output - teacher).square().mean() / teacher.square().mean()).item()
                        ),
                        "weight_rel_mse": float(
                            (
                                (reconstructed - transformed_weight).square().mean()
                                / transformed_weight.square().mean()
                            ).item()
                        ),
                    }
                )
    result = {"status": "local_held_probe_only", "source": str(args.source.resolve()), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
