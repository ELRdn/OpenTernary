"""Locate where a 205-target rotated ternary model diverges from BF16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from openternary.adapters.runtime import load_model, load_processor
from openternary.config.loader import load_config
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan

PROMPTS = (
    "Machine learning systems require careful evaluation.",
    "Write a one-sentence greeting in English.",
    "日本語で一文の挨拶を書いてください。",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    matrices, _ = load_rotation_plan(args.rotation_manifest, source)
    if len(matrices) != 205:
        raise ValueError(f"expected 205 rotations, found {len(matrices)}")
    rotations = {name: matrix.to("cuda:0") for name, matrix in matrices.items()}
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    input_ids = [processor.tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0") for prompt in PROMPTS]
    layer_outputs: dict[int, torch.Tensor] = {}
    handles = []

    for layer in range(35):

        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
            *,
            layer_index: int = layer,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"unexpected layer output type: {layer_index}")
            layer_outputs[layer_index] = hidden.detach().float().cpu()

        handles.append(model.get_submodule(f"model.language_model.layers.{layer}").register_forward_hook(capture))

    references: list[tuple[dict[int, torch.Tensor], torch.Tensor]] = []
    with torch.inference_mode():
        for ids in input_ids:
            logits = model(input_ids=ids).logits.float().cpu()
            if len(layer_outputs) != 35:
                raise ValueError("did not capture all 35 BF16 layers")
            references.append((dict(layer_outputs), logits))
            layer_outputs.clear()

    input_hooks = []
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_handle, torch.no_grad():
        for module_name in sorted(rotations):
            name = module_name + ".weight"
            weight = source_handle.get_tensor(name).to("cuda:0", dtype=torch.float32)
            transformed = apply_block_rotation(hadamard_last_dim(weight, 128), rotations[module_name])
            reconstructed = dequantize_groupwise(quantize_groupwise(transformed, 128))
            parameters[name].copy_(reconstructed.to(torch.bfloat16))

            def rotate_input(
                _module: torch.nn.Module,
                values: tuple[torch.Tensor, ...],
                *,
                name: str = module_name,
            ) -> tuple[torch.Tensor, ...]:
                data = hadamard_last_dim(values[0].float(), 128)
                data = apply_block_rotation(data, rotations[name]).to(values[0].dtype)
                return (data, *values[1:])

            input_hooks.append(model.get_submodule(module_name).register_forward_pre_hook(rotate_input))

    rows = []
    with torch.inference_mode():
        for prompt, ids, (teacher_layers, teacher_logits) in zip(PROMPTS, input_ids, references, strict=True):
            student_logits = model(input_ids=ids).logits.float().cpu()
            if len(layer_outputs) != 35:
                raise ValueError("did not capture all 35 ternary layers")
            for layer in range(35):
                teacher = teacher_layers[layer]
                student = layer_outputs[layer]
                if student.shape != teacher.shape or not torch.isfinite(student).all():
                    raise ValueError(f"invalid ternary layer output: {layer}")
                rows.append(
                    {
                        "prompt": prompt,
                        "layer": layer,
                        "relative_mse": float(((student - teacher).square().mean() / teacher.square().mean()).item()),
                    }
                )
            rows.append(
                {
                    "prompt": prompt,
                    "layer": "logits",
                    "relative_mse": float(
                        ((student_logits - teacher_logits).square().mean() / teacher_logits.square().mean()).item()
                    ),
                    "top1_match_count": int((student_logits.argmax(-1) == teacher_logits.argmax(-1)).sum().item()),
                    "token_count": int(ids.shape[-1]),
                }
            )
            layer_outputs.clear()
    for handle in (*handles, *input_hooks):
        handle.remove()
    result = {
        "status": "in_memory_layer_drift_probe",
        "source": str(source),
        "rotation_manifest": str(args.rotation_manifest.resolve()),
        "rotation_count": len(rotations),
        "prompts": list(PROMPTS),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "row_count": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
