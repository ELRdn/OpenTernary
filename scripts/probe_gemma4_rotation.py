"""Bounded Gemma 4 rotation probe using held-out activation inputs.

Both the weights and input are transformed; the probe does not alter the
inference loader or claim a loadable rotated model artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open

from openternary.calibration.capture import DiskActivationCache
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import hadamard_last_dim


def measure(weight: torch.Tensor, inputs: torch.Tensor, teacher: torch.Tensor, block_size: int) -> dict:
    weight = weight.float()
    inputs = inputs.float()
    teacher = teacher.float()
    reference = functional.linear(inputs, weight)
    ref_mse = float((reference - teacher).square().mean().item())
    teacher_energy = float(teacher.square().mean().item())
    if teacher_energy <= 0:
        raise ValueError("teacher activation energy must be positive")
    rows = []
    rng = torch.Generator(device="cpu").manual_seed(42)
    for label in ("identity", "hadamard", "signed-hadamard"):
        start = time.monotonic()
        if label == "identity":
            rotated_weight, rotated_inputs = weight, inputs
        else:
            signs = torch.ones(weight.shape[-1], device=weight.device)
            if label == "signed-hadamard":
                signs = torch.randint(0, 2, (weight.shape[-1],), generator=rng).to(weight.device).float() * 2 - 1
            rotated_weight = hadamard_last_dim(weight * signs, block_size)
            rotated_inputs = hadamard_last_dim(inputs * signs, block_size)
        preserved = functional.linear(rotated_inputs, rotated_weight)
        preserving_rel_mse = float(((preserved - reference).square().mean() / reference.square().mean()).item())
        groupwise = quantize_groupwise(rotated_weight, 128)
        reconstructed_weight = dequantize_groupwise(groupwise)
        student = functional.linear(rotated_inputs, reconstructed_weight)
        rows.append(
            {
                "method": label,
                "block_size": block_size,
                "unquantized_function_rel_mse": preserving_rel_mse,
                "weight_rel_mse": float(
                    ((reconstructed_weight - rotated_weight).square().mean() / rotated_weight.square().mean()).item()
                ),
                "held_output_mse": float((student - teacher).square().mean().item()),
                "held_output_rel_mse": float(((student - teacher).square().mean() / teacher_energy).item()),
                "zero_ratio": float((groupwise.codes == 0).float().mean().item()),
                "elapsed_s": time.monotonic() - start,
            }
        )
    return {"teacher_vs_weight_mse": ref_mse, "teacher_energy": teacher_energy, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--module", default="model.language_model.layers.0.self_attn.q_proj")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{args.module}.weight").to("cuda:0")
    cache = DiskActivationCache(args.cache)
    batches = [cache.load_valid(args.module, index) for index in range(cache.count_batches(args.module))]
    if not batches:
        raise ValueError("no held-out activation batches")
    inputs = torch.cat([batch["input"] for batch in batches]).to("cuda:0")
    teacher = torch.cat([batch["teacher_output"] for batch in batches]).to("cuda:0")
    result = {
        "module": args.module,
        "source": str(args.source.resolve()),
        "cache": str(args.cache.resolve()),
        "samples": len(batches),
        "valid_tokens": inputs.shape[0],
        "weight_shape": list(weight.shape),
        **measure(weight, inputs, teacher, args.block_size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
