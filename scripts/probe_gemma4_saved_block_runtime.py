"""Compare saved rotated hard block replay under float and BF16 runtime math."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file

from openternary.adapters.runtime import load_model, load_processor
from openternary.calibration.dataset import load_wikitext_texts, split_train_held, tokenize_texts
from openternary.config.loader import load_config
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def relative_mse(actual: torch.Tensor, teacher: torch.Tensor) -> float:
    return float(((actual.float() - teacher.float()).square().sum() / teacher.float().square().sum()).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer < 35 or args.output.exists():
        raise ValueError("invalid layer or output already exists")
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    prefix = f"model.language_model.layers.{args.layer}."
    selected = {name: matrix.to("cuda:0") for name, matrix in rotations.items() if name.startswith(prefix)}
    expected = 7 if args.layer < 15 else 5
    if len(selected) != expected:
        raise ValueError("incomplete block rotation set")
    report_path = args.block_dir / f"layer{args.layer}.json"
    artifact_path = report_path.with_suffix(".safetensors")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if (
        report.get("artifact_sha256") != digest
        or report.get("rotation_manifest_sha256") != hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
        or set(report.get("selected_modules", [])) != set(selected)
    ):
        raise ValueError("block artifact provenance mismatch")
    artifact = load_file(str(artifact_path), device="cpu")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    layer = model.get_submodule(f"model.language_model.layers.{args.layer}")
    texts = load_wikitext_texts(8, seed=42)
    _, held_texts = split_train_held(texts, 0.25)
    batches = tokenize_texts(held_texts[:2], processor.tokenizer, 128)
    examples: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    def capture_input(_module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
        captured["positional"] = copy.deepcopy(positional)
        captured["keyword"] = copy.deepcopy(keyword)

    def capture_output(_module: torch.nn.Module, _positional: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["teacher"] = output.detach().clone()

    hooks = [
        layer.register_forward_pre_hook(capture_input, with_kwargs=True),
        layer.register_forward_hook(capture_output),
    ]
    with torch.no_grad():
        for batch in batches:
            captured.clear()
            model(
                input_ids=batch["input_ids"].unsqueeze(0).to("cuda:0"),
                attention_mask=batch["attention_mask"].unsqueeze(0).to("cuda:0"),
                use_cache=False,
            )
            examples.append(dict(captured))
    for hook in hooks:
        hook.remove()

    with torch.no_grad():
        for example in examples:
            if not torch.equal(layer(*example["positional"], **example["keyword"]), example["teacher"]):
                raise ValueError("BF16 teacher block replay mismatch")

    def evaluate() -> float:
        with torch.no_grad():
            outputs = [layer(*example["positional"], **example["keyword"]) for example in examples]
        error = sum(
            float((output.float() - example["teacher"].float()).square().sum().item())
            for output, example in zip(outputs, examples, strict=True)
        )
        energy = sum(float(example["teacher"].float().square().sum().item()) for example in examples)
        return error / energy

    float_hooks = []
    for name, rotation in selected.items():
        weight = artifact[name + ".weight"].to("cuda:0")

        def float_output(
            current: torch.nn.Module,
            values: tuple[torch.Tensor, ...],
            original: torch.Tensor,
            *,
            matrix: torch.Tensor = rotation,
            saved_weight: torch.Tensor = weight,
        ) -> torch.Tensor:
            transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), matrix)
            bias = current.bias.float() if current.bias is not None else None
            return functional.linear(transformed, saved_weight.float(), bias).to(original.dtype)

        float_hooks.append(model.get_submodule(name).register_forward_hook(float_output))
    float_replay = evaluate()
    for hook in float_hooks:
        hook.remove()

    runtime_hooks = []
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, rotation in selected.items():
            parameters[name + ".weight"].copy_(artifact[name + ".weight"].to("cuda:0"))

            def rotate_input(
                _module: torch.nn.Module,
                values: tuple[torch.Tensor, ...],
                *,
                matrix: torch.Tensor = rotation,
            ) -> tuple[torch.Tensor, ...]:
                transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), matrix)
                return (transformed.to(values[0].dtype), *values[1:])

            runtime_hooks.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))
    bf16_runtime_replay = evaluate()
    for hook in runtime_hooks:
        hook.remove()
    result = {
        "source": str(source),
        "layer": args.layer,
        "artifact_sha256": digest,
        "training_hook_held_rel_mse": report["best_hooked_held_output_rel_mse"],
        "saved_bf16_weight_float_matmul_held_rel_mse": float_replay,
        "saved_bf16_weight_bf16_runtime_held_rel_mse": bf16_runtime_replay,
        "teacher_replay_exact": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
