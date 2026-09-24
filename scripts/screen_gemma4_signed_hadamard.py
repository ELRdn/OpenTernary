"""Screen fixed signed Hadamard G128 ternary Gemma 4 candidates on a frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.acceptance import compare_quality_metrics
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise, refine_groupwise_least_squares
from openternary.quant.rotation import hadamard_segments, signed_hadamard_last_dim
from openternary.services.identity import snapshot_identity


def signs_for_module(name: str, dimension: int, seed: int, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
    return (torch.randint(0, 2, (dimension,), generator=generator, dtype=torch.int8) * 2 - 1).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--targets-manifest", type=Path, required=True)
    parser.add_argument("--max-block", type=int, choices=(0, 128, 256, 512, 1024), required=True)
    parser.add_argument("--through-layer", type=int, choices=range(35), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("--scale-refit-steps", type=int, default=0)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detail_path = args.output.with_suffix(".quality.json")
    if args.output.exists() or detail_path.exists():
        raise FileExistsError(args.output)
    if args.scale_refit_steps < 0:
        raise ValueError("scale-refit-steps must be nonnegative")
    source = args.source.resolve()
    manifest_raw = args.targets_manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("schema_version") != 1
        or Path(manifest.get("source", "")).resolve() != source
        or len(manifest.get("rotations", {})) != 205
    ):
        raise ValueError("canonical target manifest mismatch")
    targets = sorted(
        name for name in manifest["rotations"] if int(name.split(".layers.")[1].split(".")[0]) <= args.through_layer
    )
    expected = sum(7 if layer < 15 else 5 for layer in range(args.through_layer + 1))
    if len(targets) != expected:
        raise ValueError(f"expected {expected} targets, found {len(targets)}")
    started = time.monotonic()
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    params = dict(model.named_parameters())
    handles = []
    stats = []
    with torch.no_grad():
        for index, name in enumerate(targets, 1):
            param = params[name + ".weight"]
            original = param.float()
            dim = original.shape[-1]
            if param.ndim != 2 or dim % 128:
                raise ValueError(f"unsupported G128 linear: {name}")
            if args.max_block:
                signs = (
                    torch.ones(dim, dtype=torch.int8, device=param.device)
                    if args.unsigned
                    else signs_for_module(name, dim, args.seed, param.device)
                )
                rotated = signed_hadamard_last_dim(original, args.max_block, signs)
                probe = torch.randn(2, dim, device=param.device)
                actual = torch.nn.functional.linear(signed_hadamard_last_dim(probe, args.max_block, signs), rotated)
                reference = torch.nn.functional.linear(probe, original)
                error = float((actual - reference).abs().max().item())
                bound = 1e-4 * float(reference.abs().max().item()) + 1e-4
                if error > bound:
                    raise ValueError(f"prequantization function mismatch: {name}: {error} > {bound}")

                def rotate_input(
                    _module: torch.nn.Module,
                    values: tuple[torch.Tensor, ...],
                    *,
                    fixed_signs: torch.Tensor = signs,
                    block: int = args.max_block,
                ) -> tuple[torch.Tensor, ...]:
                    mapped = signed_hadamard_last_dim(values[0].float(), block, fixed_signs)
                    return (mapped.to(values[0].dtype), *values[1:])

                handles.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))
            else:
                rotated = original
                error = 0.0
            result = quantize_groupwise(rotated, 128)
            if args.scale_refit_steps:
                result = refine_groupwise_least_squares(rotated, result, args.scale_refit_steps)
            hard = dequantize_groupwise(result).to(torch.bfloat16)
            if not torch.isfinite(hard).all():
                raise ValueError(f"nonfinite ternary tensor: {name}")
            mse = float((hard.float() - rotated).square().mean().item())
            stats.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "segments": list(hadamard_segments(dim, args.max_block)) if args.max_block else [],
                    "prequant_max_abs": error,
                    "weight_mse": mse,
                    "zero_fraction": float((result.codes == 0).float().mean().item()),
                }
            )
            param.copy_(hard)
            print(f"[quantize] {index}/{len(targets)} {name} mse={mse:.6g}", flush=True)
    quality = run_quality_benchmark(
        cfg, args.data, model=model, processor=processor, split=args.split, max_length=128, stride=64
    )
    baseline = json.loads(args.baseline_quality.read_text(encoding="utf-8"))
    if (
        quality["protocol_fingerprint"] != baseline["protocol_fingerprint"]
        or quality["dataset_fingerprint"] != baseline["dataset_fingerprint"]
        or snapshot_identity(source) != baseline["identity"]
        or baseline["protocol"]["split"] != args.split
    ):
        raise ValueError("quality protocol or data mismatch")
    gate = compare_quality_metrics(baseline["summary"], quality["summary"])
    if args.max_block == 0:
        status = "in_memory_g128_control_only"
    elif args.unsigned:
        status = "in_memory_unsigned_hadamard_g128_screen_only"
    else:
        status = "in_memory_signed_hadamard_g128_screen_only"
    report = {
        "status": status,
        "source": str(source),
        "source_identity": baseline["identity"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "geometry": "greedy_power_of_two_segments",
        "max_block": args.max_block,
        "unsigned": args.unsigned,
        "seed": args.seed,
        "group_size": 128,
        "scale_refit_steps": args.scale_refit_steps,
        "targets": len(targets),
        "split": args.split,
        "baseline_summary": baseline["summary"],
        "candidate_summary": quality["summary"],
        "quality_gate": gate,
        "modules": stats,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for handle in handles:
        handle.remove()
    print(json.dumps({key: value for key, value in report.items() if key != "modules"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
