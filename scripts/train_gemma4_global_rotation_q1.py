"""Pilot end-to-end KD of one learned orthogonal rotation before hard G128 ternary."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from train_gemma4_rotated_block import tokenize_calibration_rows

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.acceptance import compare_quality_metrics
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.quant.grouping import quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan
from openternary.services.identity import snapshot_identity


def cayley_from_raw(raw: torch.Tensor) -> torch.Tensor:
    skew = raw - raw.T
    identity = torch.eye(raw.shape[0], device=raw.device, dtype=raw.dtype)
    return torch.linalg.solve(identity + skew, identity - skew)


def hard_g128(weight: torch.Tensor, *, ste: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = weight.reshape(-1, 128)
    scales = grouped.abs().mean(dim=-1, keepdim=True)
    normalized = grouped / scales.clamp_min(1e-9)
    hard = normalized.round().clamp(-1, 1)
    codes = hard.detach() - normalized.clamp(-1, 1).detach() + normalized.clamp(-1, 1) if ste else hard
    return (codes * scales).reshape_as(weight), hard.to(torch.int8).reshape_as(weight), scales.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--upstream-block-dir", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--held-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top1-weight", type=float, default=0.1)
    parser.add_argument("--rotation-reg", type=float, default=0.001)
    args = parser.parse_args()
    artifact = args.output.with_suffix(".safetensors")
    if (
        args.output.exists()
        or artifact.exists()
        or args.output.with_suffix(".quality.json").exists()
        or min(args.train_samples, args.held_samples, args.steps, args.eval_every) < 1
        or args.seq_len < 8
        or args.lr <= 0
        or args.top_k < 2
        or min(args.top1_weight, args.rotation_reg) < 0
    ):
        raise ValueError("invalid experiment arguments or output exists")
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    name = "model.language_model.layers.1.self_attn.q_proj"
    if name not in rotations:
        raise ValueError("missing learned q1 rotation")
    manifest_sha = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    layer0_report_path = args.upstream_block_dir / "layer0.json"
    layer0_artifact = layer0_report_path.with_suffix(".safetensors")
    layer0_report = json.loads(layer0_report_path.read_text(encoding="utf-8"))
    layer0_sha = hashlib.sha256(layer0_artifact.read_bytes()).hexdigest()
    layer0_names = {item for item in rotations if item.startswith("model.language_model.layers.0.")}
    if (
        layer0_report.get("artifact_sha256") != layer0_sha
        or layer0_report.get("rotation_manifest_sha256") != manifest_sha
        or Path(layer0_report.get("source", "")).resolve() != source
        or set(layer0_report.get("selected_modules", [])) != layer0_names
    ):
        raise ValueError("upstream block provenance mismatch")
    upstream = load_file(str(layer0_artifact), device="cpu")
    calibration_raw = args.calibration_file.read_bytes()
    calibration = json.loads(calibration_raw)
    if calibration.get("schema_version") != 1 or calibration.get("status") != "calibration_only":
        raise ValueError("invalid calibration file")
    rng = random.Random(42)
    sampled_rows = {}
    for split, count in (("train", args.train_samples), ("held", args.held_samples)):
        rows = list(calibration[split])
        rng.shuffle(rows)
        if len(rows) < count:
            raise ValueError("insufficient calibration rows")
        sampled_rows[split] = rows[:count]
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    q_weight = parameters[name + ".weight"].detach().float()
    source_h = hadamard_last_dim(q_weight, 128).detach()
    initial_rotation = rotations[name].to("cuda:0")
    identity = torch.eye(128, device="cuda:0")
    initial_skew = torch.linalg.solve(identity + initial_rotation, identity - initial_rotation)
    raw = torch.nn.Parameter(((initial_skew - initial_skew.T) / 4).contiguous())
    if (cayley_from_raw(raw) - initial_rotation).abs().max().item() > 1e-4:
        raise ValueError("could not reconstruct initial rotation")
    initial_codes = quantize_groupwise(apply_block_rotation(source_h, initial_rotation), 128).codes
    batches = {
        split: tokenize_calibration_rows(sampled_rows[split], processor, args.seq_len) for split in ("train", "held")
    }
    teacher: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    with torch.no_grad():
        for split in ("train", "held"):
            for batch in batches[split]:
                inputs = {
                    "input_ids": batch["input_ids"].unsqueeze(0).to("cuda:0"),
                    "attention_mask": batch["attention_mask"].unsqueeze(0).to("cuda:0"),
                    "use_cache": False,
                }
                log_probs = functional.log_softmax(model(**inputs).logits.float(), dim=-1)
                top_log_probs, indices = log_probs.topk(args.top_k, dim=-1)
                top_probs = top_log_probs.exp()
                teacher[split].append(
                    {
                        "inputs": inputs,
                        "indices": indices,
                        "top_probs": top_probs,
                        "top_log_probs": top_log_probs,
                        "rest_prob": (1 - top_probs.sum(dim=-1)).clamp_min(1e-8),
                        "loss_mask": batch.get("loss_mask", batch["attention_mask"]).unsqueeze(0).to("cuda:0"),
                    }
                )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hooks = []
    with torch.no_grad():
        for upstream_name in sorted(layer0_names):
            weight = upstream[upstream_name + ".weight"]
            codes = upstream[upstream_name + ".codes"]
            scales = upstream[upstream_name + ".scales"]
            if not torch.equal(
                (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight).to(torch.bfloat16), weight
            ):
                raise ValueError(f"invalid upstream hard weight: {upstream_name}")
            parameters[upstream_name + ".weight"].copy_(weight.to("cuda:0"))
            upstream_rotation = rotations[upstream_name].to("cuda:0")

            def rotate_upstream(
                _module: torch.nn.Module,
                values: tuple[torch.Tensor, ...],
                *,
                matrix: torch.Tensor = upstream_rotation,
            ) -> tuple[torch.Tensor, ...]:
                transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), matrix)
                return (transformed.to(values[0].dtype), *values[1:])

            hooks.append(model.get_submodule(upstream_name).register_forward_pre_hook(rotate_upstream))

    def replace_q(current: torch.nn.Module, values: tuple[torch.Tensor, ...], original: torch.Tensor) -> torch.Tensor:
        matrix = cayley_from_raw(raw)
        rotated_weight = apply_block_rotation(source_h, matrix)
        reconstructed, _, _ = hard_g128(rotated_weight, ste=torch.is_grad_enabled())
        rotated_input = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), matrix)
        bias = current.bias.float() if current.bias is not None else None
        return functional.linear(rotated_input, reconstructed, bias).to(original.dtype)

    hooks.append(model.get_submodule(name).register_forward_hook(replace_q))

    def objective(example: dict[str, Any]) -> torch.Tensor:
        student_logits = model(**example["inputs"]).logits.float()
        student_log_probs = functional.log_softmax(student_logits, dim=-1)
        student_top_log_probs = student_log_probs.gather(-1, example["indices"])
        student_rest_log_prob = (1 - student_top_log_probs.exp().sum(dim=-1)).clamp_min(1e-8).log()
        top_terms = example["top_probs"] * (example["top_log_probs"] - student_top_log_probs)
        rest_terms = example["rest_prob"] * (example["rest_prob"].log() - student_rest_log_prob)
        mask = example["loss_mask"]
        kl = ((top_terms.sum(dim=-1) + rest_terms) * mask).sum() / mask.sum()
        top1 = -(student_top_log_probs[..., 0] * mask).sum() / mask.sum()
        return kl + args.top1_weight * top1

    def evaluate(split: str) -> float:
        return float(torch.stack([objective(example).detach() for example in teacher[split]]).mean().item())

    optimizer = torch.optim.Adam([raw], lr=args.lr)
    history = []
    best = None
    best_raw = None
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(teacher["train"][(step - 1) % len(teacher["train"])])
            if args.rotation_reg:
                loss = loss + args.rotation_reg * (cayley_from_raw(raw) - initial_rotation).square().mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite global rotation loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 1.0)
            optimizer.step()
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                matrix = cayley_from_raw(raw)
                current_codes = hard_g128(apply_block_rotation(source_h, matrix), ste=False)[1]
                row = {
                    "step": step,
                    "train_objective": evaluate("train"),
                    "held_objective": evaluate("held"),
                    "hard_code_change_ratio": float((current_codes != initial_codes).float().mean().item()),
                    "rotation_max_abs_change": float((matrix - initial_rotation).abs().max().item()),
                }
            history.append(row)
            if best is None or row["held_objective"] < best["held_objective"]:
                best = row
                best_raw = raw.detach().clone()
            print(json.dumps(row), flush=True)
    assert best is not None and best_raw is not None
    with torch.no_grad():
        raw.copy_(best_raw)
        if abs(evaluate("held") - best["held_objective"]) > 1e-6:
            raise ValueError("best rotation did not replay")
        matrix = cayley_from_raw(raw)
        orthogonality = float((matrix.T @ matrix - identity).abs().max().item())
        if orthogonality > 1e-4:
            raise ValueError("learned rotation is not orthogonal")
        reconstructed, codes, scales = hard_g128(apply_block_rotation(source_h, matrix), ste=False)
        saved = {
            "rotation": matrix.detach().cpu().contiguous(),
            "weight": reconstructed.to(torch.bfloat16).detach().cpu().contiguous(),
            "codes": codes.detach().cpu().contiguous(),
            "scales": scales.detach().cpu().contiguous(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(saved, str(artifact))
    reloaded = load_file(str(artifact), device="cpu")
    if set(reloaded) != set(saved) or any(not torch.equal(reloaded[key], value) for key, value in saved.items()):
        raise ValueError("saved rotation/hard codes did not reload exactly")
    with torch.no_grad():
        if not torch.equal(
            (reloaded["codes"].float().reshape(-1, 128) * reloaded["scales"].reshape(-1, 1))
            .reshape_as(reloaded["weight"])
            .to(torch.bfloat16),
            reloaded["weight"],
        ):
            raise ValueError("saved G128 hard weight mismatch")
        parameters[name + ".weight"].copy_(reloaded["weight"].to("cuda:0"))
    hooks[-1].remove()
    fixed_matrix = reloaded["rotation"].to("cuda:0")

    def rotate_fixed(_module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), fixed_matrix)
        return (transformed.to(values[0].dtype), *values[1:])

    hooks[-1] = model.get_submodule(name).register_forward_pre_hook(rotate_fixed)
    candidate = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split="validation", max_length=128, stride=64
    )
    baseline = json.loads(args.baseline_quality.read_text(encoding="utf-8"))
    if (
        candidate["protocol_fingerprint"] != baseline["protocol_fingerprint"]
        or candidate["dataset_fingerprint"] != baseline["dataset_fingerprint"]
        or snapshot_identity(source) != baseline["identity"]
    ):
        raise ValueError("quality source, protocol or data mismatch")
    gate = compare_quality_metrics(baseline["summary"], candidate["summary"])
    result = {
        "status": "single_module_global_rotation_kd_validation_only",
        "source": str(source),
        "module": name,
        "rotation_manifest_sha256": manifest_sha,
        "upstream_artifact_sha256": layer0_sha,
        "calibration_file_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "train_samples": len(teacher["train"]),
        "held_samples": len(teacher["held"]),
        "steps": args.steps,
        "best": best,
        "history": history,
        "orthogonality_max_abs": orthogonality,
        "baseline_summary": baseline["summary"],
        "candidate_summary": candidate["summary"],
        "quality_gate": gate,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    args.output.with_suffix(".quality.json").write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for hook in hooks:
        hook.remove()
    print(json.dumps({"best": best, "candidate": candidate["summary"], "gate": gate}), flush=True)


if __name__ == "__main__":
    main()
