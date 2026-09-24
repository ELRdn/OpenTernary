"""Train hard G128 block scales against BF16 full-model next-token logits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

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
    parser.add_argument("--initial-block-dir", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--upstream-block-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or not 0 <= args.layer < 35
        or args.steps < 1
        or args.eval_every < 1
        or args.lr <= 0
        or args.top_k < 2
        or args.seq_len < 8
    ):
        raise ValueError("invalid training arguments or output exists")
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    prefix = f"model.language_model.layers.{args.layer}."
    selected = {name: matrix.to("cuda:0") for name, matrix in rotations.items() if name.startswith(prefix)}
    if len(selected) != (7 if args.layer < 15 else 5):
        raise ValueError("incomplete canonical target set")
    report_path = args.initial_block_dir / f"layer{args.layer}.json"
    artifact_path = report_path.with_suffix(".safetensors")
    initial_report = json.loads(report_path.read_text(encoding="utf-8"))
    initial_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    if (
        initial_report.get("artifact_sha256") != initial_sha256
        or initial_report.get("rotation_manifest_sha256") != manifest_sha256
        or Path(initial_report.get("source", "")).resolve() != source
        or set(initial_report.get("selected_modules", [])) != set(selected)
    ):
        raise ValueError("initial block provenance mismatch")
    initial = load_file(str(artifact_path), device="cpu")
    calibration_raw = args.calibration_file.read_bytes()
    calibration = json.loads(calibration_raw)
    if calibration.get("schema_version") != 1 or calibration.get("status") != "calibration_only":
        raise ValueError("invalid calibration file")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    batches = {
        split: tokenize_calibration_rows(calibration[split], processor, args.seq_len) for split in ("train", "held")
    }
    layer = model.get_submodule(f"model.language_model.layers.{args.layer}")
    captured: dict[str, Any] = {}

    def capture_input(_module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
        captured["positional"] = copy.deepcopy(positional)
        captured["keyword"] = copy.deepcopy(keyword)

    def capture_output(_module: torch.nn.Module, _values: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["output"] = output.detach().clone()

    pre = layer.register_forward_pre_hook(capture_input, with_kwargs=True)
    post = layer.register_forward_hook(capture_output)
    teacher_data: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    with torch.no_grad():
        for split in ("train", "held"):
            for batch in batches[split]:
                captured.clear()
                inputs = {
                    "input_ids": batch["input_ids"].unsqueeze(0).to("cuda:0"),
                    "attention_mask": batch["attention_mask"].unsqueeze(0).to("cuda:0"),
                    "use_cache": False,
                }
                logits = model(**inputs).logits.float()
                if set(captured) != {"positional", "keyword", "output"}:
                    raise ValueError("incomplete BF16 block capture")
                if not torch.equal(layer(*captured["positional"], **captured["keyword"]), captured["output"]):
                    raise ValueError("BF16 block replay mismatch")
                log_probs = functional.log_softmax(logits, dim=-1)
                top_log_probs, indices = log_probs.topk(args.top_k, dim=-1)
                top_probs = top_log_probs.exp()
                rest_prob = (1 - top_probs.sum(dim=-1)).clamp_min(1e-8)
                teacher_data[split].append(
                    {
                        "inputs": inputs,
                        "indices": indices.detach(),
                        "top_probs": top_probs.detach(),
                        "top_log_probs": top_log_probs.detach(),
                        "rest_prob": rest_prob.detach(),
                        "loss_mask": batch.get("loss_mask", batch["attention_mask"]).unsqueeze(0).to("cuda:0"),
                    }
                )
    pre.remove()
    post.remove()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    upstream_artifacts: dict[str, str] = {}
    upstream_hooks = []
    if args.upstream_block_dir is not None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for previous_layer in range(args.layer):
                previous_path = args.upstream_block_dir / f"layer{previous_layer}.json"
                previous_artifact = previous_path.with_suffix(".safetensors")
                previous_report = json.loads(previous_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(previous_artifact.read_bytes()).hexdigest()
                names = {
                    name for name in rotations if name.startswith(f"model.language_model.layers.{previous_layer}.")
                }
                if (
                    previous_report.get("artifact_sha256") != digest
                    or previous_report.get("rotation_manifest_sha256") != manifest_sha256
                    or Path(previous_report.get("source", "")).resolve() != source
                    or set(previous_report.get("selected_modules", [])) != names
                ):
                    raise ValueError(f"upstream artifact provenance mismatch: {previous_layer}")
                tensors = load_file(str(previous_artifact), device="cpu")
                for name in sorted(names):
                    weight = tensors[name + ".weight"]
                    codes = tensors[name + ".codes"]
                    scales = tensors[name + ".scales"]
                    reconstructed = (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight)
                    if not torch.equal(reconstructed.to(torch.bfloat16), weight):
                        raise ValueError(f"invalid upstream hard weight: {name}")
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

                    upstream_hooks.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))
                upstream_artifacts[str(previous_layer)] = digest

    states = {}
    hooks = []
    for name, rotation in selected.items():
        codes = initial[name + ".codes"].to("cuda:0").float()
        scales = initial[name + ".scales"].to("cuda:0").reshape(-1, 1)
        weight = initial[name + ".weight"]
        if not torch.equal((codes.reshape(-1, 128) * scales).reshape_as(weight).to(torch.bfloat16).cpu(), weight):
            raise ValueError(f"initial hard weight mismatch: {name}")
        states[name] = {
            "codes": codes,
            "scales": scales,
            "rotation": rotation,
            "delta": torch.nn.Parameter(torch.zeros_like(scales)),
        }

        def replace_output(
            current: torch.nn.Module,
            values: tuple[torch.Tensor, ...],
            original: torch.Tensor,
            *,
            module_name: str = name,
        ) -> torch.Tensor:
            state = states[module_name]
            dense_weight = (state["codes"].reshape(-1, 128) * state["scales"] * state["delta"].exp()).reshape_as(
                state["codes"]
            )
            transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), state["rotation"])
            bias = current.bias.float() if current.bias is not None else None
            return functional.linear(transformed, dense_weight, bias).to(original.dtype)

        hooks.append(model.get_submodule(name).register_forward_hook(replace_output))

    def objective(example: dict[str, Any]) -> torch.Tensor:
        student_logits = model(**example["inputs"]).logits.float()
        student_log_probs = functional.log_softmax(student_logits, dim=-1)
        student_top_log_probs = student_log_probs.gather(-1, example["indices"])
        student_rest_log_prob = (1 - student_top_log_probs.exp().sum(dim=-1)).clamp_min(1e-8).log()
        top_terms = example["top_probs"] * (example["top_log_probs"] - student_top_log_probs)
        rest_terms = example["rest_prob"] * (example["rest_prob"].log() - student_rest_log_prob)
        per_token = top_terms.sum(dim=-1) + rest_terms
        mask = example["loss_mask"]
        return (per_token * mask).sum() / mask.sum()

    def evaluate(split: str) -> float:
        losses = []
        for example in teacher_data[split]:
            losses.append(objective(example).detach())
        return float(torch.stack(losses).mean().item())

    optimizer = torch.optim.Adam([state["delta"] for state in states.values()], lr=args.lr)
    history = []
    best = None
    best_state = None
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            example = teacher_data["train"][(step - 1) % len(teacher_data["train"])]
            loss = objective(example)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite logit distillation loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([state["delta"] for state in states.values()], 1.0)
            optimizer.step()
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                row = {"step": step, "train_kl": evaluate("train"), "held_kl": evaluate("held")}
            history.append(row)
            if best is None or row["held_kl"] < best["held_kl"]:
                best = row
                best_state = {name: state["delta"].detach().clone() for name, state in states.items()}
            print(json.dumps(row), flush=True)
    assert best is not None and best_state is not None
    with torch.no_grad():
        for name, delta in best_state.items():
            states[name]["delta"].copy_(delta)
        if abs(evaluate("held") - best["held_kl"]) > 1e-7:
            raise ValueError("best logit KD state did not replay")
    args.output_dir.mkdir(parents=True)
    payload = {}
    for name in selected:
        state = states[name]
        scales = state["scales"] * state["delta"].exp()
        weight = (state["codes"].reshape(-1, 128) * scales).reshape_as(state["codes"])
        payload[name + ".weight"] = weight.to(torch.bfloat16).detach().cpu().contiguous()
        payload[name + ".codes"] = initial[name + ".codes"]
        payload[name + ".scales"] = scales.detach().reshape(-1).cpu().contiguous()
        payload[name + ".thresholds"] = initial[name + ".thresholds"]
    saved_path = args.output_dir / f"layer{args.layer}.safetensors"
    save_file(payload, str(saved_path))
    reloaded = load_file(str(saved_path), device="cpu")
    for name in selected:
        reconstructed = (
            reloaded[name + ".codes"].float().reshape(-1, 128) * reloaded[name + ".scales"].reshape(-1, 1)
        ).reshape_as(reloaded[name + ".weight"])
        if not torch.equal(reconstructed.to(torch.bfloat16), reloaded[name + ".weight"]):
            raise ValueError(f"saved logit KD hard weight mismatch: {name}")
    artifact_sha256 = hashlib.sha256(saved_path.read_bytes()).hexdigest()
    report = {
        "status": (
            "global_logit_kd_scale_only_upstream_block" if upstream_artifacts else "global_logit_kd_scale_only_block"
        ),
        "source": str(source),
        "rotation_manifest_sha256": manifest_sha256,
        "calibration_file_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "initial_artifact_sha256": initial_sha256,
        "upstream_artifact_sha256": upstream_artifacts,
        "layer": args.layer,
        "selected_modules": sorted(selected),
        "bf16_replay_exact": True,
        "train_samples": len(teacher_data["train"]),
        "held_samples": len(teacher_data["held"]),
        "steps": args.steps,
        "top_k": args.top_k,
        "seq_len": args.seq_len,
        "best": best,
        "history": history,
        "artifact_sha256": artifact_sha256,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    (args.output_dir / f"layer{args.layer}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for hook in hooks:
        hook.remove()
    for hook in upstream_hooks:
        hook.remove()
    print(json.dumps({"best": best, "artifact_sha256": artifact_sha256}), flush=True)


if __name__ == "__main__":
    main()
