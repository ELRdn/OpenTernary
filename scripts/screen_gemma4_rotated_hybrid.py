"""Validation-only role ablations with learned G128 rotations.

This in-memory screen selects a candidate. Adoption requires a new saved,
reloaded artifact and an untouched test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.acceptance import compare_quality_metrics
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def role_sets(names: set[str]) -> dict[str, set[str]]:
    q = {name for name in names if name.endswith(".self_attn.q_proj.weight")}
    if len(q) != 35:
        raise ValueError("expected 35 q projections")
    result = {"q": q, "all": names}
    for cutoff in (2, 7, 14):
        result[f"late_after_{cutoff}"] = {
            name for name in names if int(name.split(".layers.", 1)[1].split(".", 1)[0]) > cutoff
        }
    for role in ("down_proj", "gate_proj", "up_proj", "k_proj", "v_proj", "o_proj"):
        extra = {name for name in names if name.endswith(f".{role}.weight")}
        if not extra:
            raise ValueError(f"missing role: {role}")
        result[f"q+{role}"] = q | extra
    result["attention"] = {name for name in names if ".self_attn." in name}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["q", "q+down_proj", "q+up_proj", "q+v_proj"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    matrices, _ = load_rotation_plan(args.rotation_manifest, source)
    names = {f"{module}.weight" for module in matrices}
    if len(names) != 205:
        raise ValueError("expected all 205 learned rotations")
    subsets = role_sets(names)
    if any(label not in subsets for label in args.candidates):
        raise ValueError(f"unknown candidate in {args.candidates}")
    baseline = json.loads(args.baseline_quality.read_text(encoding="utf-8"))
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    rotations = {module: matrix.to("cuda:0") for module, matrix in matrices.items()}
    handles = {}
    active: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_handle:
        for label in args.candidates:
            selected = subsets[label]
            started = time.monotonic()
            with torch.no_grad():
                for name in sorted(active - selected):
                    parameters[name].copy_(source_handle.get_tensor(name).to("cuda:0"))
                    handles.pop(name).remove()
                for name in sorted(selected - active):
                    module_name = name.removesuffix(".weight")
                    source_weight = source_handle.get_tensor(name).to("cuda:0", dtype=torch.float32)
                    transformed = apply_block_rotation(hadamard_last_dim(source_weight, 128), rotations[module_name])
                    reconstructed = dequantize_groupwise(quantize_groupwise(transformed, 128))
                    parameters[name].copy_(reconstructed.to(torch.bfloat16))

                    def rotate_input(
                        _module: torch.nn.Module,
                        inputs: tuple[torch.Tensor, ...],
                        *,
                        module_name: str = module_name,
                    ) -> tuple[torch.Tensor, ...]:
                        data = hadamard_last_dim(inputs[0].float(), 128)
                        data = apply_block_rotation(data, rotations[module_name]).to(inputs[0].dtype)
                        return (data, *inputs[1:])

                    handles[name] = model.get_submodule(module_name).register_forward_pre_hook(rotate_input)
            active = selected
            quality = run_quality_benchmark(
                cfg, args.data, model=model, processor=processor, split="validation", max_length=128, stride=64
            )
            if quality["protocol_fingerprint"] != baseline["protocol_fingerprint"]:
                raise ValueError("baseline/candidate protocol mismatch")
            if quality["dataset_fingerprint"] != baseline["dataset_fingerprint"]:
                raise ValueError("baseline/candidate dataset mismatch")
            gate = compare_quality_metrics(baseline["summary"], quality["summary"])
            row = {
                "candidate": label,
                "status": "in_memory_validation_only",
                "module_count": len(selected),
                "target_names_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest(),
                "ternary_parameter_count": sum(parameters[name].numel() for name in selected),
                "elapsed_s": time.monotonic() - started,
                "summary": quality["summary"],
                "quality_gate": gate,
                "dataset_fingerprint": quality["dataset_fingerprint"],
                "protocol_fingerprint": quality["protocol_fingerprint"],
            }
            detail = args.output.with_name(f"{args.output.stem}.{label}.quality.json")
            detail.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    for handle in handles.values():
        handle.remove()


if __name__ == "__main__":
    main()
