"""Research pilot: train a BF16 low-rank residual on disjoint QA answers.

This is a hybrid diagnostic. The residual is not part of the ternary codebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from train_gemma4_rotated_block import tokenize_calibration_rows

from openternary.adapters.runtime import load_model, load_processor
from openternary.config.loader import load_config
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--initial-residual-report", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--held-samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_suffix(".safetensors").exists()
        or min(args.train_samples, args.held_samples, args.steps, args.eval_every) < 1
        or args.lr <= 0
        or args.top_k < 2
    ):
        raise ValueError("invalid experiment arguments or output exists")
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    manifest_sha = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    block_tensors = {}
    block_hashes = {}
    for layer in (0, 1):
        report_path = args.block_dir / f"layer{layer}.json"
        artifact_path = report_path.with_suffix(".safetensors")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        names = {name for name in rotations if name.startswith(f"model.language_model.layers.{layer}.")}
        if (
            report.get("artifact_sha256") != digest
            or report.get("rotation_manifest_sha256") != manifest_sha
            or Path(report.get("source", "")).resolve() != source
            or set(report.get("selected_modules", [])) != names
            or len(names) != 7
        ):
            raise ValueError(f"block provenance mismatch: {layer}")
        block_tensors.update(load_file(str(artifact_path), device="cpu"))
        block_hashes[str(layer)] = digest
    initial_report = json.loads(args.initial_residual_report.read_text(encoding="utf-8"))
    initial_artifact = args.initial_residual_report.with_suffix(".safetensors")
    initial_sha = hashlib.sha256(initial_artifact.read_bytes()).hexdigest()
    layer1_names = {name for name in rotations if name.startswith("model.language_model.layers.1.")}
    if (
        initial_report.get("artifact_sha256") != initial_sha
        or initial_report.get("block_artifact_sha256") != block_hashes["1"]
        or initial_report.get("layer") != 1
        or set(initial_report.get("modules", {})) != layer1_names
        or Path(initial_report.get("source", "")).resolve() != source
    ):
        raise ValueError("initial residual provenance mismatch")
    initial = load_file(str(initial_artifact), device="cpu")
    calibration_raw = args.calibration_file.read_bytes()
    calibration = json.loads(calibration_raw)
    if calibration.get("schema_version") != 1 or calibration.get("status") != "calibration_only":
        raise ValueError("invalid calibration file")
    rows = {split: [row for row in calibration[split] if row["kind"] == "qa"] for split in ("train", "held")}
    if len(rows["train"]) < args.train_samples or len(rows["held"]) < args.held_samples:
        raise ValueError("insufficient disjoint QA rows")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    batches = {
        split: tokenize_calibration_rows(rows[split][:count], processor, args.seq_len)
        for split, count in (("train", args.train_samples), ("held", args.held_samples))
    }
    teacher = {"train": [], "held": []}
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
                        "loss_mask": batch["loss_mask"].unsqueeze(0).to("cuda:0"),
                    }
                )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = dict(model.named_parameters())
    hooks = []
    states = {}
    with torch.no_grad():
        for layer in (0, 1):
            for name in sorted(name for name in rotations if name.startswith(f"model.language_model.layers.{layer}.")):
                weight = block_tensors[name + ".weight"]
                codes = block_tensors[name + ".codes"]
                scales = block_tensors[name + ".scales"]
                reconstructed = (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight)
                if not torch.equal(reconstructed.to(torch.bfloat16), weight):
                    raise ValueError(f"hard ternary weight mismatch: {name}")
                parameters[name + ".weight"].copy_(weight.to("cuda:0"))
                matrix = rotations[name].to("cuda:0")

                def rotate_input(
                    _module: torch.nn.Module,
                    values: tuple[torch.Tensor, ...],
                    *,
                    rotation: torch.Tensor = matrix,
                ) -> tuple[torch.Tensor, ...]:
                    transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), rotation)
                    return (transformed.to(values[0].dtype), *values[1:])

                hooks.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))
    for name in sorted(layer1_names):
        left = initial[name + ".left"]
        right = initial[name + ".right"]
        if (
            left.dtype != torch.bfloat16
            or right.dtype != torch.bfloat16
            or left.shape[0] != block_tensors[name + ".weight"].shape[0]
            or right.shape[1] != block_tensors[name + ".weight"].shape[1]
            or left.shape[1] != right.shape[0]
            or left.shape[1] != initial_report["modules"][name]["saved_rank"]
        ):
            raise ValueError(f"invalid initial residual factors: {name}")
        states[name] = {
            "left": torch.nn.Parameter(left.to("cuda:0", dtype=torch.float32)),
            "right": torch.nn.Parameter(right.to("cuda:0", dtype=torch.float32)),
        }

        def add_residual(
            _module: torch.nn.Module,
            values: tuple[torch.Tensor, ...],
            original: torch.Tensor,
            *,
            module_name: str = name,
        ) -> torch.Tensor:
            state = states[module_name]
            correction = functional.linear(functional.linear(values[0].float(), state["right"]), state["left"])
            return (original.float() + correction).to(original.dtype)

        hooks.append(model.get_submodule(name).register_forward_hook(add_residual))

    def objective(example: dict) -> torch.Tensor:
        logits = model(**example["inputs"]).logits.float()
        log_probs = functional.log_softmax(logits, dim=-1)
        top_log_probs = log_probs.gather(-1, example["indices"])
        rest_log_prob = (1 - top_log_probs.exp().sum(dim=-1)).clamp_min(1e-8).log()
        terms = example["top_probs"] * (example["top_log_probs"] - top_log_probs)
        terms = terms.sum(dim=-1) + example["rest_prob"] * (example["rest_prob"].log() - rest_log_prob)
        return (terms * example["loss_mask"]).sum() / example["loss_mask"].sum()

    def evaluate(split: str) -> float:
        return float(torch.stack([objective(example).detach() for example in teacher[split]]).mean().item())

    trainable = [factor for state in states.values() for factor in state.values()]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    history = []
    best = None
    best_state = None
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(teacher["train"][(step - 1) % len(teacher["train"])])
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite QA KL at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                row = {"step": step, "held_kl": evaluate("held")}
            history.append(row)
            if best is None or row["held_kl"] < best["held_kl"]:
                best = row
                best_state = {
                    name: {key: value.detach().clone() for key, value in state.items()}
                    for name, state in states.items()
                }
            print(json.dumps(row), flush=True)
    assert best is not None and best_state is not None
    with torch.no_grad():
        for name, state in best_state.items():
            for key, value in state.items():
                states[name][key].copy_(value)
        if abs(evaluate("held") - best["held_kl"]) > 1e-6:
            raise ValueError("best held KL did not replay")
    payload = {
        name + "." + key: value.detach().to(torch.bfloat16).cpu().contiguous()
        for name, state in states.items()
        for key, value in state.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = args.output.with_suffix(".safetensors")
    save_file(payload, str(artifact))
    reloaded = load_file(str(artifact), device="cpu")
    if reloaded.keys() != payload.keys() or any(
        not torch.equal(reloaded[key], value) for key, value in payload.items()
    ):
        raise ValueError("saved residual did not reload exactly")
    report = {
        **initial_report,
        "status": "qa_logit_distilled_lowrank_residual_pilot_only",
        "runtime_mode": "factorized",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "initial_artifact_sha256": initial_sha,
        "rotation_manifest_sha256": manifest_sha,
        "calibration_file_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "upstream_block_artifact_sha256": block_hashes["0"],
        "qa_train_samples": len(teacher["train"]),
        "qa_held_samples": len(teacher["held"]),
        "steps": args.steps,
        "best": best,
        "history": history,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for hook in hooks:
        hook.remove()
    print(json.dumps({"best": best, "artifact_sha256": report["artifact_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
