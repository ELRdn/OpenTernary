"""Check full-model BF16 function preservation before ternary rounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from openternary.adapters.runtime import load_model, load_processor
from openternary.config.loader import load_config
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
    rotations_cpu, _ = load_rotation_plan(args.rotation_manifest, source)
    if len(rotations_cpu) != 205:
        raise ValueError(f"expected 205 rotations, found {len(rotations_cpu)}")
    rotations = {name: matrix.to("cuda:0") for name, matrix in rotations_cpu.items()}
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    inputs = [processor.tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0") for prompt in PROMPTS]
    with torch.inference_mode():
        references = [model(input_ids=ids).logits.float() for ids in inputs]

    handles = []
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_handle, torch.no_grad():
        for name in sorted(rotations):
            weight_name = name + ".weight"
            if weight_name not in parameters:
                raise ValueError(f"rotation target is not a model parameter: {weight_name}")
            weight = source_handle.get_tensor(weight_name).to("cuda:0", dtype=torch.float32)
            transformed = apply_block_rotation(hadamard_last_dim(weight, 128), rotations[name])
            parameters[weight_name].copy_(transformed.to(torch.bfloat16))

            def rotate_input(
                _module: torch.nn.Module,
                values: tuple[torch.Tensor, ...],
                *,
                module_name: str = name,
            ) -> tuple[torch.Tensor, ...]:
                data = hadamard_last_dim(values[0].float(), 128)
                data = apply_block_rotation(data, rotations[module_name]).to(values[0].dtype)
                return (data, *values[1:])

            handles.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))

    rows = []
    with torch.inference_mode():
        for prompt, ids, reference in zip(PROMPTS, inputs, references, strict=True):
            transformed = model(input_ids=ids).logits.float()
            if not torch.isfinite(transformed).all():
                raise ValueError(f"nonfinite logits after unquantized rotations: {prompt}")
            rows.append(
                {
                    "prompt": prompt,
                    "logits_relative_mse": float(
                        ((transformed - reference).square().mean() / reference.square().mean()).item()
                    ),
                    "top1_match": bool(torch.equal(transformed.argmax(-1), reference.argmax(-1))),
                    "top1_match_count": int((transformed.argmax(-1) == reference.argmax(-1)).sum().item()),
                    "token_count": int(ids.shape[-1]),
                }
            )
    for handle in handles:
        handle.remove()
    result = {
        "status": "unquantized_rotation_function_probe",
        "source": str(source),
        "rotation_manifest": str(args.rotation_manifest.resolve()),
        "rotation_count": len(rotations),
        "dtype": "bf16",
        "device": "cuda:0",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
