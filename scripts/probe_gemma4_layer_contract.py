"""Inspect the runtime tensor contract for one Gemma 4 text decoder block."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from openternary.adapters.runtime import load_model, load_processor
from openternary.config.loader import load_config


def describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, Mapping):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
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
    ids = processor.tokenizer("Machine learning systems require careful evaluation.", return_tensors="pt")[
        "input_ids"
    ].to("cuda:0")
    layer = model.get_submodule(f"model.language_model.layers.{args.layer}")
    observations: dict[str, Any] = {}
    captured: dict[str, Any] = {}

    def before(_module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
        observations["positional"] = describe(positional)
        observations["keyword"] = describe(keyword)
        captured["positional"] = copy.deepcopy(positional)
        captured["keyword"] = copy.deepcopy(keyword)

    def after(_module: torch.nn.Module, _positional: tuple[Any, ...], output: Any) -> None:
        observations["output"] = describe(output)
        if not isinstance(output, torch.Tensor):
            raise TypeError("expected a tensor decoder-layer output")
        captured["output"] = output.detach().clone()

    pre = layer.register_forward_pre_hook(before, with_kwargs=True)
    post = layer.register_forward_hook(after)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    pre.remove()
    post.remove()
    if not observations or not torch.isfinite(logits).all():
        raise ValueError("layer contract probe did not produce finite output")
    with torch.no_grad():
        replay = layer(*captured["positional"], **captured["keyword"])
    teacher = captured["output"]
    if not isinstance(replay, torch.Tensor) or not torch.isfinite(replay).all():
        raise ValueError("layer replay did not produce a finite tensor")
    observations["replay_relative_mse"] = float(
        ((replay.float() - teacher.float()).square().mean() / teacher.float().square().mean()).item()
    )
    observations["replay_max_abs"] = float((replay.float() - teacher.float()).abs().max().item())
    result = {"source": str(source), "layer": args.layer, "logits_shape": list(logits.shape), **observations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
