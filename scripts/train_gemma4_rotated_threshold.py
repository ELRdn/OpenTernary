"""Pilot trainable ternary thresholds and positive scales on a rotated Linear."""

from __future__ import annotations

import argparse
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
        raise ValueError(f"empty activation cache: {module}")
    inputs = torch.cat([batch["input"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    teacher = torch.cat([batch["teacher_output"] for batch in batches]).to("cuda:0", dtype=torch.float32)
    return inputs, teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--held-cache", type=Path, required=True)
    parser.add_argument("--rotation-file", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--scale-lr", type=float, default=0.05)
    parser.add_argument("--threshold-lr", type=float, default=0.01)
    parser.add_argument("--ste-width", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or min(args.scale_lr, args.threshold_lr, args.ste_width) <= 0:
        raise ValueError("invalid optimization parameters")
    if args.output.exists() or args.output.with_suffix(".safetensors").exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    rotation_path = args.rotation_file.resolve()
    rotation_report = json.loads(rotation_path.with_suffix(".json").read_text(encoding="utf-8"))
    if rotation_report.get("module") != args.module or Path(rotation_report.get("source", "")).resolve() != source:
        raise ValueError("rotation provenance mismatch")
    rotation = load_file(str(rotation_path), device="cpu")["cayley_rotation"].to("cuda:0")
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.module + ".weight").to("cuda:0", dtype=torch.float32)
    transformed = apply_block_rotation(hadamard_last_dim(weight, 128), rotation)
    quantized = quantize_groupwise(transformed, 128)
    grouped = transformed.reshape(-1, 128)
    reference_scale = quantized.scales.reshape(-1, 1)
    normal = grouped.abs() / reference_scale.clamp_min(1e-9)
    signs = grouped.sign()
    original_codes = quantized.codes.reshape(-1, 128)
    initial_codes = (signs * (normal > 0.5)).to(torch.int8)
    if not torch.equal(initial_codes, original_codes):
        raise ValueError("initial hard codes differ from canonical AbsMean")
    train_inputs, train_teacher = load_pairs(args.train_cache, args.module)
    held_inputs, held_teacher = load_pairs(args.held_cache, args.module)
    train_inputs = apply_block_rotation(hadamard_last_dim(train_inputs, 128), rotation)
    held_inputs = apply_block_rotation(hadamard_last_dim(held_inputs, 128), rotation)
    scale_delta = torch.nn.Parameter(torch.zeros_like(reference_scale))
    threshold_raw = torch.nn.Parameter(torch.zeros_like(reference_scale))
    optimizer = torch.optim.Adam(
        [{"params": [scale_delta], "lr": args.scale_lr}, {"params": [threshold_raw], "lr": args.threshold_lr}]
    )
    history = []
    best = None
    best_scales = None
    best_threshold = None
    best_codes = None

    def effective(hard_only: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        threshold = 0.05 + 0.9 * threshold_raw.sigmoid()
        hard_gate = (normal > threshold).float()
        if hard_only:
            gate = hard_gate
        else:
            surrogate = (0.5 + (normal - threshold) / (2 * args.ste_width)).clamp(0, 1)
            gate = hard_gate.detach() - surrogate.detach() + surrogate
        codes = signs * gate
        scales = reference_scale * scale_delta.exp()
        return (codes * scales).reshape_as(weight), threshold, codes

    for step in range(args.steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            student_weight, _, _ = effective(hard_only=False)
            output = functional.linear(train_inputs, student_weight)
            loss = (output - train_teacher).square().mean() / train_teacher.square().mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite train loss at step {step}")
            loss.backward()
            optimizer.step()
        if step == 0 or step % 10 == 0 or step == args.steps:
            with torch.no_grad():
                student_weight, threshold, codes = effective(hard_only=True)
                train_error = float(
                    (
                        (functional.linear(train_inputs, student_weight) - train_teacher).square().mean()
                        / train_teacher.square().mean()
                    ).item()
                )
                held_error = float(
                    (
                        (functional.linear(held_inputs, student_weight) - held_teacher).square().mean()
                        / held_teacher.square().mean()
                    ).item()
                )
                code_change = float((codes.to(torch.int8) != original_codes).float().mean().item())
                row = {
                    "step": step,
                    "train_output_rel_mse": train_error,
                    "held_output_rel_mse": held_error,
                    "hard_code_change_ratio": code_change,
                }
                history.append(row)
                if best is None or held_error < best["held_output_rel_mse"]:
                    best = row
                    best_scales = (reference_scale * scale_delta.exp()).detach().clone()
                    best_threshold = threshold.detach().clone()
                    best_codes = codes.detach().clone()
                print(json.dumps(row), flush=True)
    assert best is not None and best_scales is not None and best_threshold is not None and best_codes is not None
    if not torch.isfinite(best_scales).all() or not torch.isfinite(best_threshold).all():
        raise ValueError("invalid learned ternary state")
    result = {
        "status": "local_threshold_pilot_only",
        "module": args.module,
        "source": str(source),
        "rotation_file": str(rotation_path),
        "train_cache": str(args.train_cache.resolve()),
        "held_cache": str(args.held_cache.resolve()),
        "train_tokens": int(train_inputs.shape[0]),
        "held_tokens": int(held_inputs.shape[0]),
        "group_size": 128,
        "steps": args.steps,
        "scale_lr": args.scale_lr,
        "threshold_lr": args.threshold_lr,
        "ste_width": args.ste_width,
        "initial": history[0],
        "best": best,
        "history": history,
        "artifact_note": "research-only hard codes and learned scales; no CLI materialization contract",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "hard_codes": best_codes.to(torch.int8).reshape_as(weight).cpu().contiguous(),
            "reconstruction_scales": best_scales.reshape(-1).cpu().contiguous(),
            "threshold_ratios": best_threshold.reshape(-1).cpu().contiguous(),
        },
        str(args.output.with_suffix(".safetensors")),
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"initial": result["initial"], "best": best}), flush=True)


if __name__ == "__main__":
    main()
