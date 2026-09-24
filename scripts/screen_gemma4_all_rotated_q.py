"""Validation-only in-memory screen of learned rotations for all 35 q projections.

Weights are recomputed from the BF16 source. Each transformed input and weight
uses the same orthogonal map. This probe does not create a reloadable snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.acceptance import compare_quality_metrics
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    quantized = args.quantized.resolve()
    report = json.loads((quantized / "quantization.json").read_text(encoding="utf-8"))
    q_names = sorted(row["name"] for row in report["per_tensor"] if row["quantizable"])
    expected = {f"model.language_model.layers.{layer}.self_attn.q_proj.weight" for layer in range(35)}
    if set(q_names) != expected or report["group_size"] != 128:
        raise ValueError("expected exactly 35 G128 q projections")
    if not (source / "model.safetensors").is_file():
        raise FileNotFoundError("BF16 source must have model.safetensors")

    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "auto", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    params = dict(model.named_parameters())
    input_ids = processor.tokenizer("Machine learning systems require careful evaluation.", return_tensors="pt")[
        "input_ids"
    ].to("cuda:0")
    with torch.inference_mode():
        reference_logits = model(input_ids=input_ids).logits.float()

    rotations: dict[str, torch.Tensor] = {}
    provenance: dict[str, dict[str, object]] = {}
    hooks = []
    hook_calls: dict[str, int] = {}
    for layer in range(35):
        module_name = f"model.language_model.layers.{layer}.self_attn.q_proj"
        tensor_path = args.rotation_dir / f"gemma4-q{layer}-learned-rotation-20260924.safetensors"
        metadata_path = tensor_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["module"] != module_name or Path(metadata["source"]).resolve() != source:
            raise ValueError(f"rotation provenance mismatch for {module_name}")
        if metadata["block_size"] != 128 or metadata["group_size"] != 128:
            raise ValueError(f"rotation dimensions mismatch for {module_name}")
        rotation = load_file(str(tensor_path), device="cpu")["cayley_rotation"].to("cuda:0")
        if rotation.shape != (128, 128) or not torch.isfinite(rotation).all():
            raise ValueError(f"invalid rotation for {module_name}")
        error = (rotation.T @ rotation - torch.eye(128, device="cuda:0")).abs().max().item()
        if error > 1e-4:
            raise ValueError(f"nonorthogonal rotation for {module_name}: {error}")
        rotations[module_name] = rotation
        provenance[module_name] = {
            "sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
            "held_output_rel_mse": metadata["best"]["held_output_rel_mse"],
            "orthogonality_max_abs": error,
        }
        hook_calls[module_name] = 0

        def rotate_input(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            module_name: str = module_name,
        ) -> tuple[torch.Tensor, ...]:
            hook_calls[module_name] += 1
            rotated = hadamard_last_dim(inputs[0].float(), 128)
            rotated = apply_block_rotation(rotated, rotations[module_name]).to(inputs[0].dtype)
            return (rotated, *inputs[1:])

        hooks.append(model.get_submodule(module_name).register_forward_pre_hook(rotate_input))

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_handle, torch.no_grad():
        for name in q_names:
            module_name = name.removesuffix(".weight")
            source_weight = source_handle.get_tensor(name).to("cuda:0", dtype=torch.float32)
            rotated = apply_block_rotation(hadamard_last_dim(source_weight, 128), rotations[module_name])
            params[name].copy_(rotated.to(torch.bfloat16))

    with torch.inference_mode():
        transformed_logits = model(input_ids=input_ids).logits.float()
    function_rel_mse = float(
        ((transformed_logits - reference_logits).square().mean() / reference_logits.square().mean()).item()
    )
    top1_match = bool(torch.equal(transformed_logits.argmax(-1), reference_logits.argmax(-1)))
    if min(hook_calls.values()) == 0 or function_rel_mse > 1e-3 or not top1_match:
        raise RuntimeError(f"BF16 function preservation failed: {function_rel_mse=} {top1_match=}")

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_handle, torch.no_grad():
        for name in q_names:
            module_name = name.removesuffix(".weight")
            source_weight = source_handle.get_tensor(name).to("cuda:0", dtype=torch.float32)
            rotated = apply_block_rotation(hadamard_last_dim(source_weight, 128), rotations[module_name])
            dequantized = dequantize_groupwise(quantize_groupwise(rotated, 128))
            params[name].copy_(dequantized.to(torch.bfloat16))
    candidate = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split="validation", max_length=128, stride=64
    )
    for hook in hooks:
        hook.remove()
    baseline = json.loads(args.baseline_quality.read_text(encoding="utf-8"))
    if candidate["protocol_fingerprint"] != baseline["protocol_fingerprint"]:
        raise ValueError("baseline/candidate protocol mismatch")
    if candidate["dataset_fingerprint"] != baseline["dataset_fingerprint"]:
        raise ValueError("baseline/candidate dataset mismatch")
    gate = compare_quality_metrics(baseline["summary"], candidate["summary"])
    result = {
        "status": "in_memory_validation_only",
        "source": str(source),
        "quantized_control": str(quantized),
        "module_count": len(q_names),
        "rotation_provenance": provenance,
        "function_preservation_bf16": {"logits_relative_mse": function_rel_mse, "top1_match": top1_match},
        "hook_calls": hook_calls,
        "baseline_summary": baseline["summary"],
        "candidate": candidate,
        "quality_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "function_preservation_bf16": result["function_preservation_bf16"],
                "candidate": candidate["summary"],
                "gate": gate,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
