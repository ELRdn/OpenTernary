"""Screen frozen validation quality for selected Gemma 4 ternary module sets.

This is an in-memory ablation. It does not create a saved candidate or consume
the test split. Final claims require materialization, reload, and reevaluation.
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
from openternary.benchmark.quality import evaluate_quality
from openternary.benchmark.quality_runner import _load_dataset, run_quality_benchmark
from openternary.config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["mlp", "attention", "all-except-q"])
    parser.add_argument("--full-quality", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    quantized = args.quantized.resolve()
    report = json.loads((quantized / "quantization.json").read_text(encoding="utf-8"))
    names = {row["name"] for row in report["per_tensor"] if row["quantizable"]}
    if len(names) != 205 or report["group_size"] != 128:
        raise ValueError("expected canonical 205-target G128 ternary snapshot")
    index = json.loads((quantized / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    if not names <= index.keys():
        raise ValueError("quantized index misses canonical targets")
    source_file = source / "model.safetensors"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)

    cfg = load_config(
        cli_overrides={
            "model.id": str(source),
            "model.revision": "6befbaca7398925921802abd1f277b495b78b738",
            "device": "auto",
            "dtype": "bf16",
            "seed": 42,
        }
    )
    dataset, dataset_hash, _ = _load_dataset(args.data)
    processor = load_processor(cfg, source)
    tokenizer = processor.tokenizer
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    missing = names - parameters.keys()
    if missing:
        raise ValueError(f"model misses target parameters: {sorted(missing)[:3]}")
    if any(parameters[name].dtype != torch.bfloat16 for name in names):
        raise ValueError("target parameter dtype is not BF16")

    subsets = {
        "baseline": set(),
        "mlp": {name for name in names if ".mlp." in name},
        "attention": {name for name in names if ".self_attn." in name},
        "mlp-down": {name for name in names if ".mlp.down_proj." in name},
        "mlp-gate": {name for name in names if ".mlp.gate_proj." in name},
        "mlp-up": {name for name in names if ".mlp.up_proj." in name},
        "attn-q": {name for name in names if ".self_attn.q_proj." in name},
        "attn-k": {name for name in names if ".self_attn.k_proj." in name},
        "attn-v": {name for name in names if ".self_attn.v_proj." in name},
        "attn-o": {name for name in names if ".self_attn.o_proj." in name},
        "attn-qk": {name for name in names if any(f".self_attn.{kind}_proj." in name for kind in ("q", "k"))},
        "attn-qv": {name for name in names if any(f".self_attn.{kind}_proj." in name for kind in ("q", "v"))},
        "attn-kv": {name for name in names if any(f".self_attn.{kind}_proj." in name for kind in ("k", "v"))},
        "attn-qkv": {name for name in names if any(f".self_attn.{kind}_proj." in name for kind in ("q", "k", "v"))},
        "attn-qkv-mlp-up": {
            name
            for name in names
            if any(f".self_attn.{kind}_proj." in name for kind in ("q", "k", "v")) or ".mlp.up_proj." in name
        },
        "attn-no-q": {name for name in names if ".self_attn." in name and ".q_proj." not in name},
        "all-except-q": {name for name in names if ".self_attn.q_proj." not in name},
        "all": names,
    }
    for layer in (34, 24, 4, 12, 9):
        subsets[f"attn-q-minus-{layer}"] = {
            name for name in subsets["attn-q"] if f"model.language_model.layers.{layer}.self_attn.q_proj.weight" != name
        }
    for name in args.candidates:
        if name not in subsets:
            raise ValueError(f"unknown candidate: {name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = set()
    with safe_open(source_file, framework="pt", device="cpu") as source_handle:
        for label in (["baseline"] if not args.skip_baseline else []) + args.candidates:
            selected = subsets[label]
            start = time.monotonic()
            with torch.no_grad():
                for name in sorted(state - selected):
                    parameters[name].copy_(source_handle.get_tensor(name).to("cuda:0"))
                for name in sorted(selected - state):
                    with safe_open(quantized / index[name], framework="pt", device="cpu") as quant_handle:
                        parameters[name].copy_(quant_handle.get_tensor(name).to("cuda:0"))
            state = selected
            torch.cuda.synchronize()
            if args.full_quality:
                result = run_quality_benchmark(
                    cfg,
                    args.data,
                    model=model,
                    processor=processor,
                    split="validation",
                    max_length=128,
                    stride=64,
                )
                detail = args.output.with_name(f"{args.output.stem}.{label}.quality.json")
                detail.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                perplexity = result["perplexity"]
            else:
                result = evaluate_quality(
                    model,
                    tokenizer,
                    dataset["splits"],
                    split="validation",
                    max_length=128,
                    stride=64,
                    device="cuda:0",
                )
                perplexity = result
            row = {
                "candidate": label,
                "source": str(source),
                "quantized": str(quantized),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "dataset_fingerprint": dataset_hash,
                "target_names_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest(),
                "ternary_module_count": len(selected),
                "ternary_parameter_count": sum(parameters[name].numel() for name in selected),
                "elapsed_s": time.monotonic() - start,
                "perplexity": {key: value["perplexity"] for key, value in perplexity["by_language"].items()},
                "protocol_fingerprint": result["protocol_fingerprint"],
            }
            if args.full_quality:
                row["instruction_score"] = result["summary"]["instruction_score"]
                row["collapse_count"] = result["summary"]["collapse_count"]
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
