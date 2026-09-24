"""Compare a single rotated q projection against the saved G128 q hybrid.

This is an in-memory research probe. Runtime hooks and rotated weights are not
currently represented by the standard snapshot/export format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--rotation-file", type=Path)
    args = parser.parse_args()
    source, quantized = args.source.resolve(), args.quantized.resolve()
    report = json.loads((quantized / "quantization.json").read_text(encoding="utf-8"))
    q_names = sorted(row["name"] for row in report["per_tensor"] if row["quantizable"] and ".q_proj." in row["name"])
    if len(q_names) != 35:
        raise ValueError("expected 35 quantized q projections")
    index = json.loads((quantized / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "auto", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    params = dict(model.named_parameters())
    module_name = "model.language_model.layers.0.self_attn.q_proj"
    weight_name = f"{module_name}.weight"
    module = model.get_submodule(module_name)
    cayley = None
    if args.rotation_file:
        cayley = load_file(str(args.rotation_file), device="cpu")["cayley_rotation"].to("cuda:0")
        if cayley.shape != (args.block_size, args.block_size):
            raise ValueError("saved rotation block size mismatch")
    if weight_name not in q_names:
        raise ValueError("rotated q0 is not in the selected ternary set")
    input_ids = processor.tokenizer("Machine learning systems require careful evaluation.", return_tensors="pt")[
        "input_ids"
    ].to("cuda:0")
    with torch.inference_mode():
        reference_logits = model(input_ids=input_ids).logits.float()

    hook_calls = [0]

    def rotate_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        hook_calls[0] += 1
        rotated = hadamard_last_dim(inputs[0].float(), args.block_size)
        if cayley is not None:
            rotated = apply_block_rotation(rotated, cayley)
        rotated = rotated.to(inputs[0].dtype)
        return (rotated, *inputs[1:])

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_handle:
        original_q0 = source_handle.get_tensor(weight_name).to("cuda:0")
    rotated_q0 = hadamard_last_dim(original_q0.float(), args.block_size)
    if cayley is not None:
        rotated_q0 = apply_block_rotation(rotated_q0, cayley)
    with torch.no_grad():
        params[weight_name].copy_(rotated_q0.to(torch.bfloat16))
    hook = module.register_forward_pre_hook(rotate_input)
    with torch.inference_mode():
        rotated_bf16_logits = model(input_ids=input_ids).logits.float()
    if not hook_calls[0]:
        raise RuntimeError("rotation input hook did not execute")
    function_rel_mse = float(
        ((reference_logits - rotated_bf16_logits).square().mean() / reference_logits.square().mean()).item()
    )
    top1_match = bool(torch.equal(reference_logits.argmax(-1), rotated_bf16_logits.argmax(-1)))
    hook.remove()
    if function_rel_mse > 1e-4 or not top1_match:
        raise RuntimeError(f"BF16 rotation function preservation failed: {function_rel_mse=} {top1_match=}")

    with torch.no_grad():
        for name in q_names:
            with safe_open(quantized / index[name], framework="pt", device="cpu") as quant_handle:
                params[name].copy_(quant_handle.get_tensor(name).to("cuda:0"))
    control = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split="validation", max_length=128, stride=64
    )
    rotated_q0_ternary = dequantize_groupwise(quantize_groupwise(rotated_q0, 128)).to(torch.bfloat16)
    with torch.no_grad():
        params[weight_name].copy_(rotated_q0_ternary)
    hook = module.register_forward_pre_hook(rotate_input)
    candidate = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split="validation", max_length=128, stride=64
    )
    hook.remove()
    result = {
        "source": str(source),
        "quantized_control": str(quantized),
        "rotation": {
            "module": module_name,
            "method": "normalized-hadamard" if cayley is None else "hadamard-learned-cayley",
            "block_size": args.block_size,
            "rotation_file": str(args.rotation_file.resolve()) if args.rotation_file else None,
        },
        "function_preservation_bf16": {"logits_relative_mse": function_rel_mse, "top1_match": top1_match},
        "input_hook_calls": hook_calls[0],
        "control": control,
        "candidate": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "function_preservation_bf16": result["function_preservation_bf16"],
                "control": control["summary"],
                "candidate": candidate["summary"],
                "protocol_match": control["protocol_fingerprint"] == candidate["protocol_fingerprint"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
