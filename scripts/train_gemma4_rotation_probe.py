"""Optimize one shared orthogonal block rotation before hard G128 ternarization.

Uses only the existing calibration train and held activation caches. The saved
rotation is a research artifact. Package it with a rotation manifest and the
CLI quantize command before evaluating a saved model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import save_file

from openternary.calibration.capture import DiskActivationCache
from openternary.quant.rotation import apply_block_rotation, cayley_orthogonal, hadamard_last_dim


def load_pairs(root: Path, module: str) -> tuple[torch.Tensor, torch.Tensor]:
    cache = DiskActivationCache(root)
    batches = [cache.load_valid(module, index) for index in range(cache.count_batches(module))]
    if not batches:
        raise ValueError(f"empty cache for {module}: {root}")
    inputs = torch.cat([batch["input"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    teacher = torch.cat([batch["teacher_output"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    return inputs, teacher


def quantize_rotated(weight: torch.Tensor, *, ste: bool) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = weight.reshape(-1, 128)
    scale = grouped.abs().mean(dim=-1, keepdim=True)
    safe_scale = scale.clamp_min(1e-9)
    normalized = grouped / safe_scale
    hard = normalized.round().clamp(-1, 1)
    if ste:
        soft = normalized.clamp(-1, 1)
        codes = soft + (hard - soft).detach()
    else:
        codes = hard
    reconstructed = (codes * scale).reshape_as(weight)
    return reconstructed, hard.to(torch.int8)


def hard_metrics(
    rotated_weight: torch.Tensor,
    rotated_inputs: torch.Tensor,
    teacher: torch.Tensor,
    original_weight: torch.Tensor,
) -> dict[str, float]:
    reconstructed, codes = quantize_rotated(rotated_weight, ste=False)
    output = functional.linear(rotated_inputs, reconstructed)
    source_output = functional.linear(rotated_inputs, rotated_weight)
    teacher_energy = teacher.square().mean()
    return {
        "held_output_mse": float((output - teacher).square().mean().item()),
        "held_output_rel_mse": float(((output - teacher).square().mean() / teacher_energy).item()),
        "weight_rel_mse": float(
            ((reconstructed - rotated_weight).square().mean() / rotated_weight.square().mean()).item()
        ),
        "zero_ratio": float((codes == 0).float().mean().item()),
        "source_weight_energy": float(original_weight.square().mean().item()),
        "unquantized_teacher_mse": float((source_output - teacher).square().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--held-cache", type=Path, required=True)
    parser.add_argument("--module", default="model.language_model.layers.0.self_attn.q_proj")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{args.module}.weight").to("cuda:0", dtype=torch.float32)
    train_inputs, train_teacher = load_pairs(args.train_cache, args.module)
    held_inputs, held_teacher = load_pairs(args.held_cache, args.module)
    block_size = args.block_size
    if weight.shape[-1] % block_size or block_size % 128:
        raise ValueError("block_size must divide input dimension and be a multiple of G128")
    hadamard_weight = hadamard_last_dim(weight, block_size).detach()
    hadamard_train = hadamard_last_dim(train_inputs, block_size).detach()
    hadamard_held = hadamard_last_dim(held_inputs, block_size).detach()
    identity = torch.eye(block_size, dtype=torch.float32, device="cuda:0")
    raw = torch.nn.Parameter(torch.zeros_like(identity))
    optimizer = torch.optim.Adam([raw], lr=args.lr)
    history = []
    best = None
    best_raw = None
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            rotation = cayley_orthogonal(raw - raw.T)
            rotated_weight = apply_block_rotation(hadamard_weight, rotation)
            rotated_inputs = apply_block_rotation(hadamard_train, rotation)
            reconstructed, _ = quantize_rotated(rotated_weight, ste=True)
            output = functional.linear(rotated_inputs, reconstructed)
            train_loss = (output - train_teacher).square().mean() / train_teacher.square().mean()
            if not torch.isfinite(train_loss):
                raise ValueError(f"nonfinite train loss at step {step}")
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 1.0)
            optimizer.step()
        if step == 0 or step % 5 == 0 or step == args.steps:
            with torch.no_grad():
                rotation = cayley_orthogonal(raw - raw.T)
                rotated_weight = apply_block_rotation(hadamard_weight, rotation)
                rotated_inputs = apply_block_rotation(hadamard_held, rotation)
                metrics = hard_metrics(rotated_weight, rotated_inputs, held_teacher, weight)
                row = {"step": step, **metrics}
                history.append(row)
                if best is None or metrics["held_output_mse"] < best["held_output_mse"]:
                    best = row
                    best_raw = raw.detach().clone()
                print(json.dumps(row, sort_keys=True), flush=True)
    assert best is not None and best_raw is not None
    with torch.no_grad():
        best_rotation = cayley_orthogonal(best_raw - best_raw.T)
        orthogonality_error = float((best_rotation.T @ best_rotation - identity).abs().max().item())
        restored_weight = apply_block_rotation(hadamard_weight, best_rotation)
        restored_inputs = apply_block_rotation(hadamard_held, best_rotation)
        source_output = functional.linear(held_inputs, weight)
        unquantized_rel_mse = float(
            (
                (functional.linear(restored_inputs, restored_weight) - source_output).square().mean()
                / source_output.square().mean()
            ).item()
        )
    result = {
        "module": args.module,
        "source": str(args.source.resolve()),
        "train_cache": str(args.train_cache.resolve()),
        "held_cache": str(args.held_cache.resolve()),
        "train_tokens": int(train_inputs.shape[0]),
        "held_tokens": int(held_inputs.shape[0]),
        "block_size": block_size,
        "group_size": 128,
        "steps": args.steps,
        "lr": args.lr,
        "initial": history[0],
        "best": best,
        "orthogonality_max_abs": orthogonality_error,
        "unquantized_function_rel_mse": unquantized_rel_mse,
        "elapsed_s": time.monotonic() - started,
        "history": history,
        "artifact_note": "matrix-only artifact; package with a rotation manifest and CLI quantize",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"cayley_rotation": best_rotation.cpu().contiguous()}, str(args.output.with_suffix(".safetensors")))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps({key: result[key] for key in ("initial", "best", "orthogonality_max_abs", "elapsed_s")}), flush=True
    )


if __name__ == "__main__":
    main()
