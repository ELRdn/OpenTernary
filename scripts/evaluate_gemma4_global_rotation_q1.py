"""Reload and evaluate a saved layer-0 plus learned q1 hard ternary pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from openternary.adapters.runtime import load_model, load_processor
from openternary.benchmark.acceptance import compare_quality_metrics
from openternary.benchmark.quality_runner import run_quality_benchmark
from openternary.config.loader import load_config
from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
from openternary.quant.rotation import (
    apply_block_rotation,
    fixed_signs_for_module,
    hadamard_last_dim,
    load_rotation_plan,
    signed_hadamard_last_dim,
)
from openternary.services.identity import snapshot_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--upstream-block-dir", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--additional-block-report", type=Path)
    parser.add_argument("--additional-roles", nargs="+")
    parser.add_argument("--fixed-hadamard-additions", action="store_true")
    parser.add_argument("--fixed-hadamard-artifact-report", type=Path)
    parser.add_argument("--fixed-hadamard-block", type=int, default=1024)
    parser.add_argument("--fixed-hadamard-seed", type=int, default=42)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-case-id", type=str)
    parser.add_argument("--diagnose-repeats", type=int, default=3)
    parser.add_argument("--diagnose-quality-first", action="store_true")
    parser.add_argument("--diagnose-deterministic-algorithms", action="store_true")
    parser.add_argument("--diagnose-layer-hashes", action="store_true")
    parser.add_argument("--diagnose-sync-layers", action="store_true")
    args = parser.parse_args()
    if args.diagnose_case_id is None and (
        args.diagnose_quality_first or args.diagnose_layer_hashes or args.diagnose_sync_layers
    ):
        raise ValueError("diagnostic options require --diagnose-case-id")
    if args.diagnose_deterministic_algorithms:
        if args.diagnose_case_id is None:
            raise ValueError("deterministic algorithm override is diagnostic-only")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    detail_path = args.output.with_suffix(".quality.json")
    if args.output.exists() or detail_path.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    manifest_sha = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    layer0_names = {item for item in rotations if item.startswith("model.language_model.layers.0.")}
    layer0_report_path = args.upstream_block_dir / "layer0.json"
    layer0_artifact = layer0_report_path.with_suffix(".safetensors")
    layer0_report = json.loads(layer0_report_path.read_text(encoding="utf-8"))
    layer0_sha = hashlib.sha256(layer0_artifact.read_bytes()).hexdigest()
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    candidate_artifact = args.candidate_report.with_suffix(".safetensors")
    candidate_sha = hashlib.sha256(candidate_artifact.read_bytes()).hexdigest()
    name = "model.language_model.layers.1.self_attn.q_proj"
    if (
        layer0_report.get("artifact_sha256") != layer0_sha
        or layer0_report.get("rotation_manifest_sha256") != manifest_sha
        or Path(layer0_report.get("source", "")).resolve() != source
        or set(layer0_report.get("selected_modules", [])) != layer0_names
        or candidate_report.get("artifact_sha256") != candidate_sha
        or candidate_report.get("upstream_artifact_sha256") != layer0_sha
        or candidate_report.get("rotation_manifest_sha256") != manifest_sha
        or Path(candidate_report.get("source", "")).resolve() != source
        or candidate_report.get("module") != name
    ):
        raise ValueError("candidate provenance mismatch")
    upstream = load_file(str(layer0_artifact), device="cpu")
    candidate = load_file(str(candidate_artifact), device="cpu")
    additional = None
    fixed_artifact = None
    additional_sha = None
    additional_names: set[str] = set()
    if args.fixed_hadamard_additions and args.additional_block_report is not None:
        raise ValueError("fixed Hadamard additions and saved block additions are mutually exclusive")
    if args.fixed_hadamard_artifact_report is not None and not args.fixed_hadamard_additions:
        raise ValueError("saved fixed Hadamard additions require --fixed-hadamard-additions")
    if args.fixed_hadamard_block not in (128, 256, 512, 1024):
        raise ValueError("unsupported fixed Hadamard block size")
    if args.additional_roles and args.additional_block_report is None and not args.fixed_hadamard_additions:
        raise ValueError("additional roles require an additional block report")
    if args.fixed_hadamard_additions:
        available_names = {
            item for item in rotations if item.startswith("model.language_model.layers.1.") and item != name
        }
        available_roles = {item.rsplit(".", 1)[-1] for item in available_names}
        if args.additional_roles and not set(args.additional_roles) <= available_roles:
            raise ValueError("unknown additional module role")
        additional_names = {
            item
            for item in available_names
            if not args.additional_roles or item.rsplit(".", 1)[-1] in args.additional_roles
        }
        if args.fixed_hadamard_artifact_report is not None:
            fixed_report = json.loads(args.fixed_hadamard_artifact_report.read_text(encoding="utf-8"))
            fixed_path = args.fixed_hadamard_artifact_report.with_suffix(".safetensors")
            additional_sha = hashlib.sha256(fixed_path.read_bytes()).hexdigest()
            if (
                fixed_report.get("schema_version") != 1
                or fixed_report.get("status") != "fixed_signed_hadamard_hard_g128_roles"
                or fixed_report.get("source") != str(source)
                or fixed_report.get("source_identity") != snapshot_identity(source)
                or fixed_report.get("targets_manifest_sha256") != manifest_sha
                or fixed_report.get("layer") != 1
                or set(fixed_report.get("roles", [])) != {name.rsplit(".", 1)[-1] for name in additional_names}
                or {entry.get("name") for entry in fixed_report.get("modules", [])} != additional_names
                or fixed_report.get("block") != args.fixed_hadamard_block
                or fixed_report.get("seed") != args.fixed_hadamard_seed
                or fixed_report.get("group_size") != 128
                or fixed_report.get("artifact_sha256") != additional_sha
            ):
                raise ValueError("fixed Hadamard artifact provenance mismatch")
            fixed_artifact = load_file(str(fixed_path), device="cpu")
            if set(fixed_artifact) != {
                name + suffix for name in additional_names for suffix in (".weight", ".codes", ".scales")
            }:
                raise ValueError("fixed Hadamard artifact tensor set mismatch")
    if args.additional_block_report is not None:
        additional_artifact = args.additional_block_report.with_suffix(".safetensors")
        additional_report = json.loads(args.additional_block_report.read_text(encoding="utf-8"))
        additional_sha = hashlib.sha256(additional_artifact.read_bytes()).hexdigest()
        available_names = {
            item for item in rotations if item.startswith("model.language_model.layers.1.") and item != name
        }
        available_roles = {item.rsplit(".", 1)[-1] for item in available_names}
        if args.additional_roles and not set(args.additional_roles) <= available_roles:
            raise ValueError("unknown additional module role")
        additional_names = {
            item
            for item in available_names
            if not args.additional_roles or item.rsplit(".", 1)[-1] in args.additional_roles
        }
        reported_names = set(additional_report.get("selected_modules", []))
        if (
            additional_report.get("artifact_sha256") != additional_sha
            or additional_report.get("rotation_manifest_sha256") != manifest_sha
            or Path(additional_report.get("source", "")).resolve() != source
            or reported_names not in (available_names, available_names | {name})
            or (reported_names == available_names and additional_report.get("fixed_q_artifact_sha256") != candidate_sha)
            or additional_report.get("upstream_artifact_sha256") not in ({"0": layer0_sha}, layer0_sha)
        ):
            raise ValueError("additional block provenance mismatch")
        additional = load_file(str(additional_artifact), device="cpu")
    if set(candidate) != {"rotation", "weight", "codes", "scales"}:
        raise ValueError("unexpected candidate tensors")
    identity = torch.eye(128)
    if (
        candidate["rotation"].shape != (128, 128)
        or not torch.isfinite(candidate["rotation"]).all()
        or (candidate["rotation"].T @ candidate["rotation"] - identity).abs().max() > 1e-4
        or candidate["codes"].dtype != torch.int8
        or not ((candidate["codes"] >= -1) & (candidate["codes"] <= 1)).all()
    ):
        raise ValueError("invalid candidate rotation or codes")

    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    parameters = dict(model.named_parameters())
    handles = []
    with torch.no_grad():
        for module_name in sorted(layer0_names | {name} | additional_names):
            fixed_signs = None
            if module_name == name:
                tensors = candidate
                prefix = ""
                matrix = candidate["rotation"]
            elif args.fixed_hadamard_additions and module_name in additional_names:
                original = parameters[module_name + ".weight"].detach().float()
                fixed_signs = fixed_signs_for_module(
                    module_name, original.shape[-1], args.fixed_hadamard_seed, original.device
                )
                if fixed_artifact is not None:
                    tensors = fixed_artifact
                    prefix = module_name + "."
                else:
                    rotated = signed_hadamard_last_dim(original, args.fixed_hadamard_block, fixed_signs)
                    quantized = quantize_groupwise(rotated, 128)
                    tensors = {
                        "weight": dequantize_groupwise(quantized).to(torch.bfloat16),
                        "codes": quantized.codes,
                        "scales": quantized.scales,
                    }
                    prefix = ""
                matrix = None
            elif module_name in additional_names:
                assert additional is not None
                tensors = additional
                prefix = module_name + "."
                matrix = rotations[module_name]
            else:
                tensors = upstream
                prefix = module_name + "."
                matrix = rotations[module_name]
            weight = tensors[prefix + "weight"]
            codes = tensors[prefix + "codes"]
            scales = tensors[prefix + "scales"]
            if (
                weight.dtype != torch.bfloat16
                or weight.shape != codes.shape
                or weight.shape[-1] % 128
                or scales.numel() != weight.numel() // 128
                or not ((codes >= -1) & (codes <= 1)).all()
                or not torch.isfinite(scales).all()
                or (scales < 0).any()
                or not torch.equal(
                    (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight).to(torch.bfloat16),
                    weight,
                )
            ):
                raise ValueError(f"invalid hard ternary weight: {module_name}")
            parameters[module_name + ".weight"].copy_(weight.to("cuda:0"))
            if fixed_signs is not None:

                def rotate_input(
                    _module: torch.nn.Module,
                    values: tuple[torch.Tensor, ...],
                    *,
                    signs: torch.Tensor = fixed_signs,
                    block: int = args.fixed_hadamard_block,
                ) -> tuple[torch.Tensor, ...]:
                    transformed = signed_hadamard_last_dim(values[0].float(), block, signs)
                    return (transformed.to(values[0].dtype), *values[1:])

            else:
                assert matrix is not None
                matrix = matrix.to("cuda:0")

                def rotate_input(
                    _module: torch.nn.Module,
                    values: tuple[torch.Tensor, ...],
                    *,
                    rotation: torch.Tensor = matrix,
                ) -> tuple[torch.Tensor, ...]:
                    transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), rotation)
                    return (transformed.to(values[0].dtype), *values[1:])

            handles.append(model.get_submodule(module_name).register_forward_pre_hook(rotate_input))
    if args.diagnose_case_id is not None:
        if args.diagnose_repeats < 2:
            raise ValueError("diagnose-repeats must be at least two")
        dataset = json.loads(args.data.read_text(encoding="utf-8"))
        matches = [item for item in dataset["instruction_cases"][args.split] if item["id"] == args.diagnose_case_id]
        if len(matches) != 1:
            raise ValueError("diagnostic case ID must match exactly one case in the selected split")
        prompt = matches[0]["prompt"]
        generation = cfg.benchmark.generation
        preceding_quality = None
        if args.diagnose_quality_first:
            measured = run_quality_benchmark(
                cfg, args.data, model=model, processor=processor, split=args.split, max_length=128, stride=64
            )
            preceding_quality = {
                "summary": measured["summary"],
                "case_response": next(
                    item["response_text"]
                    for item in measured["instructions"]["results"]
                    if item["id"] == args.diagnose_case_id
                ),
            }
        trials = []
        for repeat in range(args.diagnose_repeats):
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=cfg.benchmark.thinking,
            )
            inputs = {
                key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
            }
            input_length = inputs["input_ids"].shape[-1]
            layer_hashes: dict[str, str] = {}
            trace_handles = []
            if args.diagnose_sync_layers:
                for layer_index in range(35):
                    layer_name = f"model.language_model.layers.{layer_index}"

                    def synchronize_layer(
                        _module: torch.nn.Module,
                        _values: tuple[torch.Tensor, ...],
                        _output: torch.Tensor | tuple[torch.Tensor, ...],
                    ) -> None:
                        torch.cuda.synchronize("cuda:0")

                    trace_handles.append(model.get_submodule(layer_name).register_forward_hook(synchronize_layer))
            if args.diagnose_layer_hashes:

                def hash_hidden(hidden: torch.Tensor) -> str:
                    raw = hidden.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
                    return hashlib.sha256(raw).hexdigest()

                layer_names = [f"model.language_model.layers.{index}" for index in range(35)]
                layer_names.extend(
                    [
                        "model.language_model.layers.0.self_attn",
                        "model.language_model.layers.0.mlp",
                        *sorted(name for name in layer0_names),
                    ]
                )

                def capture_layer_input(
                    _module: torch.nn.Module,
                    values: tuple[torch.Tensor, ...],
                    *,
                    hashes: dict[str, str] = layer_hashes,
                ) -> None:
                    if "layer0_input" not in hashes:
                        hashes["layer0_input"] = hash_hidden(values[0])

                trace_handles.append(
                    model.get_submodule("model.language_model.layers.0").register_forward_pre_hook(capture_layer_input)
                )
                for layer_name in layer_names:

                    def capture_layer(
                        _module: torch.nn.Module,
                        _values: tuple[torch.Tensor, ...],
                        output: torch.Tensor | tuple[torch.Tensor, ...],
                        *,
                        label: str = layer_name,
                        hashes: dict[str, str] = layer_hashes,
                    ) -> None:
                        if label not in hashes:
                            hidden = output if isinstance(output, torch.Tensor) else output[0]
                            hashes[label] = hash_hidden(hidden)

                    trace_handles.append(model.get_submodule(layer_name).register_forward_hook(capture_layer))
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=generation.do_sample,
                    num_beams=generation.num_beams,
                    max_new_tokens=generation.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            for trace_handle in trace_handles:
                trace_handle.remove()
            token_ids = generated.sequences[0, input_length:].tolist()
            scores = [score[0].float().cpu() for score in generated.scores]
            trials.append({"repeat": repeat, "token_ids": token_ids, "scores": scores, "layer_hashes": layer_hashes})
            print(f"[diagnose] repeat {repeat + 1}/{args.diagnose_repeats}: {len(token_ids)} tokens", flush=True)
        reference = trials[0]
        comparisons = []
        for trial in trials[1:]:
            first_difference = next(
                (
                    index
                    for index, pair in enumerate(zip(reference["token_ids"], trial["token_ids"], strict=False))
                    if pair[0] != pair[1]
                ),
                None,
            )
            if first_difference is None and len(reference["token_ids"]) != len(trial["token_ids"]):
                first_difference = min(len(reference["token_ids"]), len(trial["token_ids"]))
            first_score_difference = next(
                (
                    index
                    for index, pair in enumerate(zip(reference["scores"], trial["scores"], strict=False))
                    if not torch.equal(pair[0], pair[1])
                ),
                None,
            )
            score_details = None
            if first_difference is not None and first_difference < min(len(reference["scores"]), len(trial["scores"])):
                left = reference["scores"][first_difference]
                right = trial["scores"][first_difference]
                score_details = {
                    "max_abs_logit_difference": float((left - right).abs().max()),
                    "reference_top2": torch.topk(left, 2).indices.tolist(),
                    "repeat_top2": torch.topk(right, 2).indices.tolist(),
                    "reference_top2_logits": torch.topk(left, 2).values.tolist(),
                    "repeat_top2_logits": torch.topk(right, 2).values.tolist(),
                }
            comparisons.append(
                {
                    "repeat": trial["repeat"],
                    "first_token_difference": first_difference,
                    "first_score_difference": first_score_difference,
                    "divergence_logits": score_details,
                }
            )
        diagnostic = {
            "status": "saved_candidate_single_case_reproducibility_diagnostic",
            "case_id": args.diagnose_case_id,
            "artifact_sha256": [layer0_sha, candidate_sha, additional_sha],
            "deterministic_algorithms": args.diagnose_deterministic_algorithms,
            "layer_hashes_enabled": args.diagnose_layer_hashes,
            "synchronize_layers": args.diagnose_sync_layers,
            "preceding_quality": preceding_quality,
            "trials": [
                {
                    "repeat": trial["repeat"],
                    "token_ids": trial["token_ids"],
                    "text": processor.decode(trial["token_ids"], skip_special_tokens=True),
                    "layer_hashes": trial["layer_hashes"],
                    "top2_scores": [
                        {
                            "token_ids": torch.topk(score, 2).indices.tolist(),
                            "logits": torch.topk(score, 2).values.tolist(),
                        }
                        for score in trial["scores"]
                    ],
                }
                for trial in trials
            ],
            "comparisons": comparisons,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8")
        for handle in handles:
            handle.remove()
        print(
            json.dumps({key: value for key, value in diagnostic.items() if key != "trials"}, ensure_ascii=False),
            flush=True,
        )
        return
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
    result = {
        "status": (
            "saved_fixed_hadamard_hard_ternary_quality_only"
            if fixed_artifact is not None
            else "saved_eight_plus_in_memory_fixed_hadamard_quality_only"
            if args.fixed_hadamard_additions
            else "saved_rotated_hard_ternary_quality_only"
        ),
        "source": str(source),
        "split": args.split,
        "candidate_artifact_sha256": candidate_sha,
        "upstream_artifact_sha256": layer0_sha,
        "additional_artifact_sha256": additional_sha,
        "ternary_module_count": len(layer0_names) + 1 + len(additional_names),
        "additional_roles": sorted(item.rsplit(".", 1)[-1] for item in additional_names),
        "fixed_hadamard_additions": args.fixed_hadamard_additions,
        "fixed_hadamard_block": args.fixed_hadamard_block if args.fixed_hadamard_additions else None,
        "fixed_hadamard_seed": args.fixed_hadamard_seed if args.fixed_hadamard_additions else None,
        "source_identity": baseline["identity"],
        "baseline_summary": baseline["summary"],
        "candidate_summary": quality["summary"],
        "quality_gate": gate,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for handle in handles:
        handle.remove()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
