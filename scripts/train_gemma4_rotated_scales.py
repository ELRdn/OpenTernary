"""Pilot per-group reconstruction scales for fixed rotated ternary codes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from openternary.calibration.capture import DiskActivationCache
from openternary.quant.grouping import quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim


def load_pairs(root: Path, module: str) -> tuple[torch.Tensor, torch.Tensor]:
    cache = DiskActivationCache(root)
    batches = [cache.load_valid(module, index) for index in range(cache.count_batches(module))]
    if not batches:
        raise ValueError(f"empty activation cache: {module} at {root}")
    inputs = torch.cat([batch["input"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    outputs = torch.cat([batch["teacher_output"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    return inputs, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--held-cache", type=Path, required=True)
    parser.add_argument("--rotation-file", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=0.0001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0 or args.regularization < 0:
        raise ValueError("invalid optimization parameters")
    if args.output.exists() or args.output.with_suffix(".safetensors").exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    rotation_path = args.rotation_file.resolve()
    rotation_report = json.loads(rotation_path.with_suffix(".json").read_text(encoding="utf-8"))
    if rotation_report.get("module") != args.module or Path(rotation_report.get("source", "")).resolve() != source:
        raise ValueError("rotation provenance mismatch")
    rotation = load_file(str(rotation_path), device="cpu")["cayley_rotation"].to("cuda:0")
    if rotation.shape != (128, 128) or not torch.isfinite(rotation).all():
        raise ValueError("invalid rotation matrix")
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.module + ".weight").to("cuda:0", dtype=torch.float32)
    transformed = apply_block_rotation(hadamard_last_dim(weight, 128), rotation)
    quantized = quantize_groupwise(transformed, 128)
    codes = quantized.codes.float().reshape(-1, 128)
    initial_scales = quantized.scales.reshape(-1, 1)
    train_inputs, train_teacher = load_pairs(args.train_cache, args.module)
    held_inputs, held_teacher = load_pairs(args.held_cache, args.module)
    train_inputs = apply_block_rotation(hadamard_last_dim(train_inputs, 128), rotation)
    held_inputs = apply_block_rotation(hadamard_last_dim(held_inputs, 128), rotation)
    train_energy = train_teacher.square().mean()
    held_energy = held_teacher.square().mean()
    if train_energy <= 0 or held_energy <= 0:
        raise ValueError("zero teacher energy")
    delta = torch.nn.Parameter(torch.zeros_like(initial_scales))
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    history = []
    best = None
    best_scales = None

    def reconstruct() -> torch.Tensor:
        return (codes * (initial_scales * delta.exp())).reshape_as(weight)

    for step in range(args.steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            train_output = functional.linear(train_inputs, reconstruct())
            train_loss = (train_output - train_teacher).square().mean() / train_energy
            loss = train_loss + args.regularization * delta.square().mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss at step {step}")
            loss.backward()
            optimizer.step()
        if step == 0 or step % 10 == 0 or step == args.steps:
            with torch.no_grad():
                reconstructed = reconstruct()
                train_rel = float(
                    (
                        (functional.linear(train_inputs, reconstructed) - train_teacher).square().mean() / train_energy
                    ).item()
                )
                held_rel = float(
                    (
                        (functional.linear(held_inputs, reconstructed) - held_teacher).square().mean() / held_energy
                    ).item()
                )
                row = {"step": step, "train_output_rel_mse": train_rel, "held_output_rel_mse": held_rel}
                history.append(row)
                if best is None or held_rel < best["held_output_rel_mse"]:
                    best = row
                    best_scales = (initial_scales * delta.exp()).detach().clone()
                print(json.dumps(row), flush=True)
    assert best is not None and best_scales is not None
    if not torch.isfinite(best_scales).all() or (best_scales < 0).any():
        raise ValueError("invalid learned scales")
    result = {
        "status": "local_scale_pilot_only",
        "module": args.module,
        "source": str(source),
        "rotation_file": str(rotation_path),
        "rotation_sha256": hashlib.sha256(rotation_path.read_bytes()).hexdigest(),
        "train_cache": str(args.train_cache.resolve()),
        "held_cache": str(args.held_cache.resolve()),
        "train_tokens": int(train_inputs.shape[0]),
        "held_tokens": int(held_inputs.shape[0]),
        "group_size": 128,
        "steps": args.steps,
        "lr": args.lr,
        "regularization": args.regularization,
        "initial": history[0],
        "best": best,
        "history": history,
        "artifact_note": "fixed codes with learned positive scales; no CLI materialization contract",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"reconstruction_scales": best_scales.reshape(-1).cpu().contiguous()},
        str(args.output.with_suffix(".safetensors")),
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"initial": result["initial"], "best": best}), flush=True)


if __name__ == "__main__":
    main()
