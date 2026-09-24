"""Jointly reconstruct the first three Gemma 4 text blocks with hard G128 codes."""

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
from openternary.quant.grouping import quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--initial-block-dir", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--scale-lr", type=float, default=0.01)
    parser.add_argument("--threshold-lr", type=float, default=0.003)
    parser.add_argument("--ste-width", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.steps < 1 or args.eval_every < 1:
        raise ValueError("output exists or invalid training schedule")
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    manifest_sha256 = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    selected = {
        name: matrix.to("cuda:0")
        for name, matrix in rotations.items()
        if any(name.startswith(f"model.language_model.layers.{layer}.") for layer in range(3))
    }
    if len(selected) != 21:
        raise ValueError("expected 21 canonical targets in blocks 0-2")
    initial_artifacts = {}
    initial_tensors = {}
    for layer in range(3):
        report_path = args.initial_block_dir / f"layer{layer}.json"
        artifact_path = report_path.with_suffix(".safetensors")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        names = {name for name in selected if name.startswith(f"model.language_model.layers.{layer}.")}
        if (
            report.get("layer") != layer
            or Path(report.get("source", "")).resolve() != source
            or report.get("rotation_manifest_sha256") != manifest_sha256
            or report.get("artifact_sha256") != digest
            or set(report.get("selected_modules", [])) != names
        ):
            raise ValueError(f"initial block provenance mismatch: {layer}")
        initial_artifacts[str(layer)] = digest
        initial_tensors.update(load_file(str(artifact_path), device="cpu"))

    calibration_raw = args.calibration_file.read_bytes()
    calibration_sha256 = hashlib.sha256(calibration_raw).hexdigest()
    calibration = json.loads(calibration_raw)
    if calibration.get("schema_version") != 1 or calibration.get("status") != "calibration_only":
        raise ValueError("invalid calibration file")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    layers = [model.get_submodule(f"model.language_model.layers.{i}") for i in range(3)]
    batches = {split: tokenize_calibration_rows(calibration[split], processor, 128) for split in ("train", "held")}
    examples: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    captured: dict[str, Any] = {}
    capture_hooks = []
    for index, layer in enumerate(layers):

        def capture_input(
            _module: torch.nn.Module,
            positional: tuple[Any, ...],
            keyword: dict[str, Any],
            *,
            layer_index: int = index,
        ) -> None:
            captured[f"input{layer_index}"] = copy.deepcopy(positional)
            captured[f"keyword{layer_index}"] = copy.deepcopy(keyword)

        def capture_output(
            _module: torch.nn.Module,
            _positional: tuple[Any, ...],
            output: torch.Tensor,
            *,
            layer_index: int = index,
        ) -> None:
            captured[f"teacher{layer_index}"] = output.detach().clone()

        capture_hooks.extend(
            (
                layer.register_forward_pre_hook(capture_input, with_kwargs=True),
                layer.register_forward_hook(capture_output),
            )
        )
    with torch.no_grad():
        for split in ("train", "held"):
            for batch in batches[split]:
                captured.clear()
                model(
                    input_ids=batch["input_ids"].unsqueeze(0).to("cuda:0"),
                    attention_mask=batch["attention_mask"].unsqueeze(0).to("cuda:0"),
                    use_cache=False,
                )
                if len(captured) != 9:
                    raise ValueError("incomplete three-block teacher capture")
                examples[split].append(dict(captured))
    for hook in capture_hooks:
        hook.remove()

    def forward_window(example: dict[str, Any]) -> list[torch.Tensor]:
        outputs = []
        hidden = example["input0"][0]
        for index, layer in enumerate(layers):
            positional = (hidden, *example[f"input{index}"][1:])
            hidden = layer(*positional, **example[f"keyword{index}"])
            outputs.append(hidden)
        return outputs

    with torch.no_grad():
        for split in ("train", "held"):
            for example in examples[split]:
                for index, output in enumerate(forward_window(example)):
                    if not torch.equal(output, example[f"teacher{index}"]):
                        raise ValueError(f"BF16 three-block replay mismatch at block {index}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    states = {}
    hooks = []
    initialization_code_mismatches = {}
    for name, rotation in sorted(selected.items()):
        module = model.get_submodule(name)
        source_weight = module.weight.detach().float()
        rotated_weight = apply_block_rotation(hadamard_last_dim(source_weight, 128), rotation)
        quantized = quantize_groupwise(rotated_weight, 128)
        reference_scale = quantized.scales.reshape(-1, 1)
        grouped = rotated_weight.reshape(-1, 128)
        normal = grouped.abs() / reference_scale.clamp_min(1e-9)
        signs = grouped.sign()
        saved_codes = initial_tensors[name + ".codes"].to("cuda:0").reshape(-1, 128)
        saved_scales = initial_tensors[name + ".scales"].to("cuda:0").reshape(-1, 1)
        saved_thresholds = initial_tensors[name + ".thresholds"].to("cuda:0").reshape(-1, 1)
        if (
            saved_codes.shape != normal.shape
            or saved_scales.shape != reference_scale.shape
            or saved_thresholds.shape != reference_scale.shape
        ):
            raise ValueError(f"saved initial hard tensor shapes mismatch: {name}")
        code_mismatch = int(((signs * (normal > saved_thresholds)).to(torch.int8) != saved_codes).sum().item())
        if code_mismatch > max(2, saved_codes.numel() // 100000):
            raise ValueError(f"saved initial hard codes mismatch: {name}, {code_mismatch}/{saved_codes.numel()}")
        initialization_code_mismatches[name] = code_mismatch
        threshold_unit = ((saved_thresholds - 0.05) / 0.9).clamp(1e-5, 1 - 1e-5)
        states[name] = {
            "shape": tuple(source_weight.shape),
            "rotation": rotation,
            "reference_scale": reference_scale,
            "normal": normal,
            "signs": signs,
            "scale_delta": torch.nn.Parameter(
                torch.where(
                    reference_scale > 0,
                    saved_scales / reference_scale.clamp_min(1e-9),
                    torch.ones_like(saved_scales),
                ).log()
            ),
            "threshold_raw": torch.nn.Parameter(torch.logit(threshold_unit)),
        }

        def replace_output(
            current: torch.nn.Module,
            values: tuple[torch.Tensor, ...],
            original: torch.Tensor,
            *,
            module_name: str = name,
        ) -> torch.Tensor:
            state = states[module_name]
            threshold = 0.05 + 0.9 * state["threshold_raw"].sigmoid()
            hard_gate = (state["normal"] > threshold).float()
            surrogate = (0.5 + (state["normal"] - threshold) / (2 * args.ste_width)).clamp(0, 1)
            gate = hard_gate.detach() - surrogate.detach() + surrogate
            weight = (state["signs"] * gate * state["reference_scale"] * state["scale_delta"].exp()).reshape(
                state["shape"]
            )
            transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), state["rotation"])
            bias = current.bias.float() if current.bias is not None else None
            return functional.linear(transformed, weight, bias).to(original.dtype)

        hooks.append(module.register_forward_hook(replace_output))

    optimizer = torch.optim.Adam(
        [
            {"params": [state["scale_delta"] for state in states.values()], "lr": args.scale_lr},
            {"params": [state["threshold_raw"] for state in states.values()], "lr": args.threshold_lr},
        ]
    )

    def losses(example: dict[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = forward_window(example)
        individual = [
            (output.float() - example[f"teacher{index}"].float()).square().mean()
            / example[f"teacher{index}"].float().square().mean()
            for index, output in enumerate(outputs)
        ]
        return 0.2 * individual[0] + 0.2 * individual[1] + 0.6 * individual[2], individual

    def evaluate(split: str) -> dict[str, float]:
        totals = torch.zeros(4, dtype=torch.float64, device="cuda:0")
        for example in examples[split]:
            objective, individual = losses(example)
            totals += torch.stack((objective, *individual)).detach().double()
        totals /= len(examples[split])
        return {
            "objective": float(totals[0].item()),
            "block0_rel_mse": float(totals[1].item()),
            "block1_rel_mse": float(totals[2].item()),
            "block2_rel_mse": float(totals[3].item()),
        }

    history = []
    best = None
    best_state = None
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            example = examples["train"][(step - 1) % len(examples["train"])]
            objective, _ = losses(example)
            if not torch.isfinite(objective):
                raise ValueError(f"nonfinite joint loss at step {step}")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for state in states.values()
                    for parameter in (state["scale_delta"], state["threshold_raw"])
                ],
                1.0,
            )
            optimizer.step()
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                row = {"step": step, "train": evaluate("train"), "held": evaluate("held")}
            history.append(row)
            if best is None or row["held"]["objective"] < best["held"]["objective"]:
                best = row
                best_state = {
                    name: (state["scale_delta"].detach().clone(), state["threshold_raw"].detach().clone())
                    for name, state in states.items()
                }
            print(json.dumps(row), flush=True)
    assert best is not None and best_state is not None
    with torch.no_grad():
        for name, (scale_delta, threshold_raw) in best_state.items():
            states[name]["scale_delta"].copy_(scale_delta)
            states[name]["threshold_raw"].copy_(threshold_raw)
        replay = evaluate("held")
        if abs(replay["objective"] - best["held"]["objective"]) > 1e-7:
            raise ValueError("joint best state did not replay")

    args.output_dir.mkdir(parents=True)
    artifact_hashes = {}
    for layer in range(3):
        payload = {}
        names = sorted(name for name in selected if name.startswith(f"model.language_model.layers.{layer}."))
        with torch.no_grad():
            for name in names:
                state = states[name]
                threshold = 0.05 + 0.9 * state["threshold_raw"].sigmoid()
                codes = (state["signs"] * (state["normal"] > threshold)).to(torch.int8)
                scales = state["reference_scale"] * state["scale_delta"].exp()
                weight = (codes.float() * scales).reshape(state["shape"])
                payload[name + ".weight"] = weight.to(torch.bfloat16).cpu().contiguous()
                payload[name + ".codes"] = codes.reshape(state["shape"]).cpu().contiguous()
                payload[name + ".scales"] = scales.reshape(-1).cpu().contiguous()
                payload[name + ".thresholds"] = threshold.reshape(-1).cpu().contiguous()
        artifact_path = args.output_dir / f"layer{layer}.safetensors"
        save_file(payload, str(artifact_path))
        reloaded = load_file(str(artifact_path), device="cpu")
        if reloaded.keys() != payload.keys() or any(
            not torch.equal(reloaded[key], value) for key, value in payload.items()
        ):
            raise ValueError(f"joint block reload mismatch: {layer}")
        for name in names:
            reconstructed = (
                reloaded[name + ".codes"].float().reshape(-1, 128) * reloaded[name + ".scales"].reshape(-1, 1)
            ).reshape_as(reloaded[name + ".weight"])
            if not torch.equal(reconstructed.to(torch.bfloat16), reloaded[name + ".weight"]):
                raise ValueError(f"joint hard weight mismatch: {name}")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        artifact_hashes[str(layer)] = digest
        report = {
            "status": "joint_three_block_reconstruction_only",
            "source": str(source),
            "rotation_manifest_sha256": manifest_sha256,
            "calibration_file_sha256": calibration_sha256,
            "layer": layer,
            "joint_window": [0, 1, 2],
            "initial_artifact_sha256": initial_artifacts,
            "selected_modules": names,
            "bf16_replay_exact": True,
            "best": best,
            "artifact_sha256": digest,
        }
        (args.output_dir / f"layer{layer}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for hook in hooks:
        hook.remove()
    result = {
        "status": "joint_three_block_reconstruction_only",
        "source": str(source),
        "calibration_file_sha256": calibration_sha256,
        "rotation_manifest_sha256": manifest_sha256,
        "initial_artifact_sha256": initial_artifacts,
        "initialization_code_mismatches": initialization_code_mismatches,
        "artifact_sha256": artifact_hashes,
        "train_samples": len(examples["train"]),
        "held_samples": len(examples["held"]),
        "steps": args.steps,
        "eval_every": args.eval_every,
        "best": best,
        "history": history,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    (args.output_dir / "joint.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "artifact_sha256": artifact_hashes}), flush=True)


if __name__ == "__main__":
    main()
