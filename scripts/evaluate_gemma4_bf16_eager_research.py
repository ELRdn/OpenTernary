"""Research-only BF16 quality control using Gemma 4 eager text attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.services.identity import snapshot_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    model.get_submodule("model.language_model").config._attn_implementation = "eager"
    active_attention = {
        model.get_submodule(f"model.language_model.layers.{index}.self_attn").config._attn_implementation
        for index in range(35)
    }
    if active_attention != {"eager"}:
        raise ValueError(f"eager attention did not reach all text layers: {active_attention}")
    quality = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split=args.split, max_length=128, stride=64
    )
    quality["research_execution"] = {
        "text_attention_implementation": "eager",
        "source_identity": snapshot_identity(source),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": quality["summary"], "protocol_fingerprint": quality["protocol_fingerprint"]}))


if __name__ == "__main__":
    main()
