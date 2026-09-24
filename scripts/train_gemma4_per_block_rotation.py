"""Pilot one orthogonal 128x128 rotation per input block of one Linear."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from openternary.calibration.capture import DiskActivationCache
from openternary.quant.rotation import hadamard_last_dim


def load_pairs(root: Path, module: str) -> tuple[torch.Tensor, torch.Tensor]:
    cache = DiskActivationCache(root)
    batches = [cache.load_valid(module, index) for index in range(cache.count_batches(module))]
    if not batches:
        raise ValueError(f"empty activation cache: {module}")
    inputs = torch.cat([batch["input"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    teacher = torch.cat([batch["teacher_output"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    return inputs, teacher


def apply_per_block(x: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    blocks = matrices.shape[0]
    if x.shape[-1] != blocks * 128:
        raise ValueError("rotation block count does not match input dimension")
    return torch.einsum("nbi,bij->nbj", x.reshape(-1, blocks, 128), matrices).reshape_as(x)


def rotations(raw: torch.Tensor) -> torch.Tensor:
    skew = raw - raw.transpose(-1, -2)
    identity = torch.eye(128, device=raw.device, dtype=raw.dtype).expand_as(skew)
    return torch.linalg.solve(identity + skew, identity - skew)


def reconstruct(weight: torch.Tensor, *, ste: bool) -> torch.Tensor:
    grouped = weight.reshape(-1, 128)
    scale = grouped.abs().mean(dim=-1, keepdim=True)
    safe_scale = scale.clamp_min(1e-9)
    normalized = grouped / safe_scale
    hard = normalized.round().clamp(-1, 1)
    codes = normalized.clamp(-1, 1) + (hard - normalized.clamp(-1, 1)).detach() if ste else hard
    return (codes * scale).reshape_as(weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--held-cache", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--init-rotation", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        raise ValueError("invalid optimization parameters")
    if args.output.exists() or args.output.with_suffix(".safetensors").exists():
        raise FileExistsError(args.output)
    with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.module + ".weight").to("cuda:0", dtype=torch.float32)
    train_inputs, train_teacher = load_pairs(args.train_cache, args.module)
    held_inputs, held_teacher = load_pairs(args.held_cache, args.module)
    if weight.shape[-1] % 128:
        raise ValueError("input dimension must be divisible by 128")
    blocks = weight.shape[-1] // 128
    weight_h = hadamard_last_dim(weight, 128).detach()
    train_h = hadamard_last_dim(train_inputs, 128).detach()
    held_h = hadamard_last_dim(held_inputs, 128).detach()
    raw_initial = torch.zeros(blocks, 128, 128, device="cuda:0")
    if args.init_rotation is not None:
        shared = load_file(str(args.init_rotation), device="cpu")["cayley_rotation"].to("cuda:0")
        if shared.shape != (128, 128):
            raise ValueError("initial shared rotation must be 128x128")
        identity_single = torch.eye(128, device="cuda:0")
        skew = torch.linalg.solve(identity_single + shared, identity_single - shared)
        skew = (skew - skew.T) / 2
        raw_initial = (skew / 2).expand(blocks, -1, -1).contiguous()
        if (rotations(raw_initial)[0] - shared).abs().max().item() > 1e-4:
            raise ValueError("could not reconstruct initial shared rotation")
    raw = torch.nn.Parameter(raw_initial)
    optimizer = torch.optim.Adam([raw], lr=args.lr)
    history = []
    best = None
    best_raw = None
    teacher_energy = held_teacher.square().mean()
    for step in range(args.steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            matrix = rotations(raw)
            student_weight = reconstruct(apply_per_block(weight_h, matrix), ste=True)
            output = functional.linear(apply_per_block(train_h, matrix), student_weight)
            loss = (output - train_teacher).square().mean() / train_teacher.square().mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite train loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 1.0)
            optimizer.step()
        if step == 0 or step % 10 == 0 or step == args.steps:
            with torch.no_grad():
                matrix = rotations(raw)
                student_weight = reconstruct(apply_per_block(weight_h, matrix), ste=False)
                output = functional.linear(apply_per_block(held_h, matrix), student_weight)
                error = float(((output - held_teacher).square().mean() / teacher_energy).item())
                row = {"step": step, "held_output_rel_mse": error}
                history.append(row)
                if best is None or error < best["held_output_rel_mse"]:
                    best = row
                    best_raw = raw.detach().clone()
                print(json.dumps(row), flush=True)
    assert best is not None and best_raw is not None
    with torch.no_grad():
        matrix = rotations(best_raw)
        identity = torch.eye(128, device="cuda:0").expand_as(matrix)
        orthogonality = float((matrix.transpose(-1, -2) @ matrix - identity).abs().max().item())
        transformed_weight = apply_per_block(weight_h, matrix)
        transformed_input = apply_per_block(held_h, matrix)
        unquantized = functional.linear(transformed_input, transformed_weight)
        source_output = functional.linear(held_inputs, weight)
        function_error = float(((unquantized - source_output).square().mean() / source_output.square().mean()).item())
    result = {
        "status": "per_block_rotation_pilot_only",
        "module": args.module,
        "source": str(args.source.resolve()),
        "train_cache": str(args.train_cache.resolve()),
        "held_cache": str(args.held_cache.resolve()),
        "blocks": blocks,
        "train_tokens": int(train_inputs.shape[0]),
        "held_tokens": int(held_inputs.shape[0]),
        "group_size": 128,
        "steps": args.steps,
        "lr": args.lr,
        "init_rotation": str(args.init_rotation.resolve()) if args.init_rotation else None,
        "initial": history[0],
        "best": best,
        "orthogonality_max_abs": orthogonality,
        "unquantized_function_rel_mse": function_error,
        "history": history,
        "artifact_note": "per-block rotation is a research-only format without CLI loader support",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"per_block_rotation": matrix.cpu().contiguous()}, str(args.output.with_suffix(".safetensors")))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"initial": result["initial"], "best": best, "orthogonality": orthogonality}), flush=True)


if __name__ == "__main__":
    main()
