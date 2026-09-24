"""Pilot block-output reconstruction with hard ternary codes on one Gemma 4 layer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file

from openternary.adapters.runtime import load_model, load_processor
from openternary.calibration.dataset import WIKITEXT_REVISION, load_wikitext_texts, split_train_held, tokenize_texts
from openternary.config.loader import load_config
from openternary.quant.grouping import quantize_groupwise
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def tokenize_calibration_rows(rows: list[dict[str, str]], processor: Any, seq_len: int) -> list[dict[str, Any]]:
    batches = []
    for row in rows:
        if row["kind"] == "text":
            batches.append(tokenize_texts([row["text"]], processor.tokenizer, seq_len)[0])
            continue
        if row["kind"] == "qa":
            user_message = {"role": "user", "content": row["text"]}
            prompt = processor.apply_chat_template(
                [user_message],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )["input_ids"][0]
            complete = processor.apply_chat_template(
                [user_message, {"role": "assistant", "content": row["answer"]}],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=False,
            )["input_ids"][0]
            if complete.numel() <= prompt.numel() or not torch.equal(complete[: prompt.numel()], prompt):
                raise ValueError(f"QA chat template prefix mismatch: {row['id']}")
            ids = complete[:seq_len]
            padding = seq_len - ids.numel()
            if prompt.numel() >= ids.numel() or complete.numel() > seq_len:
                raise ValueError(f"QA answer truncated by seq_len: {row['id']}")
            mask = torch.ones(ids.numel(), dtype=torch.long)
            loss_mask = torch.zeros(seq_len, dtype=torch.long)
            loss_mask[prompt.numel() - 1 : ids.numel() - 1] = 1
            pad_id = processor.tokenizer.pad_token_id
            batches.append(
                {
                    "input_ids": functional.pad(ids, (0, padding), value=pad_id if pad_id is not None else 0),
                    "attention_mask": functional.pad(mask, (0, padding)),
                    "loss_mask": loss_mask,
                }
            )
            continue
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = inputs["input_ids"][0][:seq_len]
        mask = torch.ones(ids.numel(), dtype=torch.long)
        padding = seq_len - ids.numel()
        pad_id = processor.tokenizer.pad_token_id
        batches.append(
            {
                "input_ids": functional.pad(ids, (0, padding), value=pad_id if pad_id is not None else 0),
                "attention_mask": functional.pad(mask, (0, padding)),
            }
        )
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=6)
    parser.add_argument("--held-samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--scale-lr", type=float, default=0.02)
    parser.add_argument("--threshold-lr", type=float, default=0.005)
    parser.add_argument("--ste-width", type=float, default=0.1)
    parser.add_argument("--learn-dense-latent", action="store_true")
    parser.add_argument("--latent-lr", type=float, default=0.0001)
    parser.add_argument("--latent-reg", type=float, default=0.01)
    parser.add_argument("--upstream-block-dir", type=Path)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".safetensors").exists():
        raise FileExistsError(args.output)
    if not 0 <= args.layer < 35:
        raise ValueError("Gemma 4 E2B text layer must be in [0, 35)")
    if args.calibration_file is None and (not 1 <= args.train_samples <= 6 or not 1 <= args.held_samples <= 2):
        raise ValueError("train/held sample count exceeds the frozen 6/2 split")
    if (
        args.seq_len < 8
        or args.steps < 1
        or args.eval_every < 1
        or min(args.scale_lr, args.threshold_lr, args.ste_width, args.latent_lr) <= 0
        or args.latent_reg < 0
    ):
        raise ValueError("invalid experiment parameters")
    started = time.monotonic()
    source = args.source.resolve()
    matrices, _ = load_rotation_plan(args.rotation_manifest, source)
    prefix = f"model.language_model.layers.{args.layer}."
    selected = {name: matrix.to("cuda:0") for name, matrix in matrices.items() if name.startswith(prefix)}
    expected = 7 if args.layer < 15 else 5
    if len(selected) != expected:
        raise ValueError(f"expected {expected} canonical Linear targets in layer {args.layer}, found {len(selected)}")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    layer = model.get_submodule(f"model.language_model.layers.{args.layer}")
    calibration_file_sha256 = None
    dataset = "Salesforce/wikitext wikitext-2-raw-v1 train"
    dataset_revision = WIKITEXT_REVISION
    if args.calibration_file is None:
        texts = load_wikitext_texts(8, seed=42)
        texts_sha256 = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()
        train_texts, held_texts = split_train_held(texts, 0.25)
        train_batches = tokenize_texts(train_texts[: args.train_samples], processor.tokenizer, args.seq_len)
        held_batches = tokenize_texts(held_texts[: args.held_samples], processor.tokenizer, args.seq_len)
    else:
        raw = args.calibration_file.read_bytes()
        calibration_file_sha256 = hashlib.sha256(raw).hexdigest()
        calibration = json.loads(raw)
        if calibration.get("schema_version") != 1 or calibration.get("status") != "calibration_only":
            raise ValueError("invalid external calibration file")
        train_rows, held_rows = calibration["train"], calibration["held"]
        if (
            len(train_rows) < args.train_samples
            or len(held_rows) < args.held_samples
            or len({row["id"] for row in train_rows + held_rows}) != len(train_rows + held_rows)
            or any(row["kind"] not in {"text", "chat", "qa"} or not row["text"] for row in train_rows + held_rows)
        ):
            raise ValueError("invalid or insufficient external calibration rows")
        texts_sha256 = hashlib.sha256(
            json.dumps(train_rows + held_rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        train_batches = tokenize_calibration_rows(train_rows[: args.train_samples], processor, args.seq_len)
        held_batches = tokenize_calibration_rows(held_rows[: args.held_samples], processor, args.seq_len)
        dataset = "frozen disjoint English/Japanese text+chat calibration"
        dataset_revision = calibration_file_sha256
    examples: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    captured: dict[str, Any] = {}

    def capture_input(_module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
        captured["positional"] = copy.deepcopy(positional)
        captured["keyword"] = copy.deepcopy(keyword)

    def capture_output(_module: torch.nn.Module, _positional: tuple[Any, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("decoder block must return a tensor")
        captured["teacher"] = output.detach().clone()

    pre = layer.register_forward_pre_hook(capture_input, with_kwargs=True)
    post = layer.register_forward_hook(capture_output)
    with torch.no_grad():
        for split, batches in (("train", train_batches), ("held", held_batches)):
            for batch in batches:
                captured.clear()
                inputs = batch["input_ids"].unsqueeze(0).to("cuda:0")
                mask = batch["attention_mask"].unsqueeze(0).to("cuda:0")
                model(input_ids=inputs, attention_mask=mask, use_cache=False)
                if set(captured) != {"positional", "keyword", "teacher"}:
                    raise ValueError("incomplete decoder block capture")
                captured["valid_mask"] = mask.detach().clone()
                examples[split].append(dict(captured))
    pre.remove()
    post.remove()
    with torch.no_grad():
        for split in ("train", "held"):
            for example in examples[split]:
                replay = layer(*example["positional"], **example["keyword"])
                if not torch.equal(replay, example["teacher"]):
                    raise ValueError(f"BF16 block replay differs on {split}")
    upstream_artifacts: dict[str, str] = {}
    upstream_hooks = []
    if args.upstream_block_dir is not None and args.layer > 0:
        parameters = dict(model.named_parameters())
        manifest_sha256 = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
        with torch.no_grad():
            for previous_layer in range(args.layer):
                report_path = args.upstream_block_dir / f"layer{previous_layer}.json"
                artifact_path = report_path.with_suffix(".safetensors")
                previous_report = json.loads(report_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                previous_names = {
                    name for name in matrices if name.startswith(f"model.language_model.layers.{previous_layer}.")
                }
                if (
                    previous_report.get("status")
                    not in {
                        "single_block_reconstruction_pilot_only",
                        "upstream_conditioned_block_reconstruction_only",
                        "dense_latent_block_reconstruction_only",
                        "dense_latent_upstream_block_reconstruction_only",
                    }
                    or previous_report.get("layer") != previous_layer
                    or Path(previous_report.get("source", "")).resolve() != source
                    or previous_report.get("rotation_manifest_sha256") != manifest_sha256
                    or previous_report.get("artifact_sha256") != digest
                    or set(previous_report.get("selected_modules", [])) != previous_names
                ):
                    raise ValueError(f"upstream block provenance mismatch: layer {previous_layer}")
                tensors = load_file(str(artifact_path), device="cpu")
                for name in sorted(previous_names):
                    weight = tensors[name + ".weight"]
                    codes = tensors[name + ".codes"]
                    scales = tensors[name + ".scales"]
                    if (
                        weight.dtype != torch.bfloat16
                        or codes.dtype != torch.int8
                        or weight.shape != parameters[name + ".weight"].shape
                        or scales.numel() != weight.numel() // 128
                        or not torch.equal(
                            (codes.float().reshape(-1, 128) * scales.reshape(-1, 1))
                            .reshape_as(weight)
                            .to(torch.bfloat16),
                            weight,
                        )
                    ):
                        raise ValueError(f"invalid upstream hard ternary block: {name}")
                    parameters[name + ".weight"].copy_(weight.to("cuda:0"))
                    rotation = matrices[name].to("cuda:0")

                    def rotate_input(
                        _module: torch.nn.Module,
                        values: tuple[torch.Tensor, ...],
                        *,
                        matrix: torch.Tensor = rotation,
                    ) -> tuple[torch.Tensor, ...]:
                        transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), matrix)
                        return (transformed.to(values[0].dtype), *values[1:])

                    upstream_hooks.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))
                upstream_artifacts[str(previous_layer)] = digest

        student_capture: dict[str, Any] = {}

        def capture_student_input(
            _module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]
        ) -> None:
            student_capture["positional"] = copy.deepcopy(positional)
            student_capture["keyword"] = copy.deepcopy(keyword)

        student_hook = layer.register_forward_pre_hook(capture_student_input, with_kwargs=True)
        with torch.no_grad():
            for split, batches in (("train", train_batches), ("held", held_batches)):
                for example, batch in zip(examples[split], batches, strict=True):
                    student_capture.clear()
                    model(
                        input_ids=batch["input_ids"].unsqueeze(0).to("cuda:0"),
                        attention_mask=batch["attention_mask"].unsqueeze(0).to("cuda:0"),
                        use_cache=False,
                    )
                    if set(student_capture) != {"positional", "keyword"}:
                        raise ValueError("incomplete quantized-upstream input capture")
                    example["positional"] = student_capture["positional"]
                    example["keyword"] = student_capture["keyword"]
        student_hook.remove()
    for param in model.parameters():
        param.requires_grad_(False)

    states = {}
    hooks = []
    for module_name, rotation in sorted(selected.items()):
        module = model.get_submodule(module_name)
        source_weight = module.weight.detach().float()
        rotated_weight = apply_block_rotation(hadamard_last_dim(source_weight, 128), rotation)
        quantized = quantize_groupwise(rotated_weight, 128)
        grouped = rotated_weight.reshape(-1, 128)
        reference_scale = quantized.scales.reshape(-1, 1)
        normal = grouped.abs() / reference_scale.clamp_min(1e-9)
        signs = grouped.sign()
        original_codes = quantized.codes.reshape(-1, 128)
        if not torch.equal((signs * (normal > 0.5)).to(torch.int8), original_codes):
            raise ValueError(f"initial codes differ from AbsMean: {module_name}")
        states[module_name] = {
            "rotation": rotation,
            "shape": tuple(source_weight.shape),
            "reference_scale": reference_scale,
            "normal": normal,
            "signs": signs,
            "original_codes": original_codes,
            "scale_delta": torch.nn.Parameter(torch.zeros_like(reference_scale)),
            "threshold_raw": torch.nn.Parameter(torch.zeros_like(reference_scale)),
        }
        if args.learn_dense_latent:
            states[module_name]["latent"] = torch.nn.Parameter(grouped.detach().clone())
            states[module_name]["latent_source"] = grouped.detach().clone()

        def replace_output(
            current: torch.nn.Module,
            values: tuple[torch.Tensor, ...],
            original: torch.Tensor,
            *,
            name: str = module_name,
        ) -> torch.Tensor:
            state = states[name]
            threshold = 0.05 + 0.9 * state["threshold_raw"].sigmoid()
            if args.learn_dense_latent:
                normal = state["latent"].abs() / state["reference_scale"].clamp_min(1e-9)
                signs = state["latent"].sign()
            else:
                normal, signs = state["normal"], state["signs"]
            hard_gate = (normal > threshold).float()
            surrogate = (0.5 + (normal - threshold) / (2 * args.ste_width)).clamp(0, 1)
            gate = hard_gate.detach() - surrogate.detach() + surrogate
            codes = signs * gate
            scales = state["reference_scale"] * state["scale_delta"].exp()
            weight = (codes * scales).reshape(state["shape"])
            inputs = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), state["rotation"])
            bias = current.bias.float() if current.bias is not None else None
            return functional.linear(inputs, weight, bias).to(original.dtype)

        hooks.append(module.register_forward_hook(replace_output))

    optimizer_groups = [
        {"params": [state["scale_delta"] for state in states.values()], "lr": args.scale_lr},
        {"params": [state["threshold_raw"] for state in states.values()], "lr": args.threshold_lr},
    ]
    if args.learn_dense_latent:
        optimizer_groups.append({"params": [state["latent"] for state in states.values()], "lr": args.latent_lr})
    optimizer = torch.optim.Adam(optimizer_groups)

    def hard_codes(state: dict[str, Any]) -> torch.Tensor:
        threshold = 0.05 + 0.9 * state["threshold_raw"].sigmoid()
        if args.learn_dense_latent:
            normal = state["latent"].abs() / state["reference_scale"].clamp_min(1e-9)
            signs = state["latent"].sign()
        else:
            normal, signs = state["normal"], state["signs"]
        return (signs * (normal > threshold)).to(torch.int8)

    def evaluate(split: str) -> float:
        total_error = 0.0
        total_energy = 0.0
        for example in examples[split]:
            output = layer(*example["positional"], **example["keyword"]).float()
            teacher = example["teacher"].float()
            valid = example["valid_mask"].unsqueeze(-1)
            total_error += float(((output - teacher).square() * valid).sum().item())
            total_energy += float((teacher.square() * valid).sum().item())
        return total_error / total_energy

    def code_change() -> float:
        changed = 0
        total = 0
        for state in states.values():
            codes = hard_codes(state)
            changed += int((codes != state["original_codes"]).sum().item())
            total += codes.numel()
        return changed / total

    history = []
    best = None
    best_state = None
    for step in range(args.steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            example = examples["train"][(step - 1) % len(examples["train"])]
            output = layer(*example["positional"], **example["keyword"]).float()
            teacher = example["teacher"].float()
            valid = example["valid_mask"].unsqueeze(-1)
            loss = ((output - teacher).square() * valid).sum() / (teacher.square() * valid).sum()
            if args.learn_dense_latent and args.latent_reg:
                regularizer = torch.stack(
                    [
                        (state["latent"] - state["latent_source"]).square().mean()
                        / state["latent_source"].square().mean().clamp_min(1e-9)
                        for state in states.values()
                    ]
                ).mean()
                loss = loss + args.latent_reg * regularizer
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite block loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for group in optimizer.param_groups for param in group["params"]],
                1.0,
            )
            optimizer.step()
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                row = {
                    "step": step,
                    "train_output_rel_mse": evaluate("train"),
                    "held_output_rel_mse": evaluate("held"),
                    "hard_code_change_ratio": code_change(),
                }
            history.append(row)
            if best is None or row["held_output_rel_mse"] < best["held_output_rel_mse"]:
                best = row
                best_state = {
                    name: (
                        state["scale_delta"].detach().clone(),
                        state["threshold_raw"].detach().clone(),
                        state["latent"].detach().clone() if args.learn_dense_latent else None,
                    )
                    for name, state in states.items()
                }
            print(json.dumps(row), flush=True)
    assert best is not None and best_state is not None
    with torch.no_grad():
        for name, (scale_delta, threshold_raw, latent) in best_state.items():
            states[name]["scale_delta"].copy_(scale_delta)
            states[name]["threshold_raw"].copy_(threshold_raw)
            if latent is not None:
                states[name]["latent"].copy_(latent)
        best_hooked_held = evaluate("held")
        if abs(best_hooked_held - best["held_output_rel_mse"]) > 1e-7:
            raise ValueError("best block checkpoint did not replay")
        payload = {}
        for name, state in states.items():
            threshold = 0.05 + 0.9 * state["threshold_raw"].sigmoid()
            codes = hard_codes(state)
            scales = state["reference_scale"] * state["scale_delta"].exp()
            reconstructed = (codes.float() * scales).reshape(state["shape"])
            if not torch.isfinite(reconstructed).all() or not torch.isfinite(scales).all():
                raise ValueError(f"nonfinite hard block state: {name}")
            payload[name + ".weight"] = reconstructed.to(torch.bfloat16).cpu().contiguous()
            payload[name + ".codes"] = codes.reshape(state["shape"]).cpu().contiguous()
            payload[name + ".scales"] = scales.reshape(-1).cpu().contiguous()
            payload[name + ".thresholds"] = threshold.reshape(-1).cpu().contiguous()
    for hook in hooks:
        hook.remove()
    for hook in upstream_hooks:
        hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output.with_suffix(".safetensors")
    save_file(payload, str(artifact_path))
    loaded = load_file(str(artifact_path), device="cpu")
    if loaded.keys() != payload.keys() or any(not torch.equal(loaded[name], value) for name, value in payload.items()):
        raise ValueError("saved block artifact did not reload exactly")
    for name in selected:
        reconstructed = (
            loaded[name + ".codes"].float().reshape(-1, 128) * loaded[name + ".scales"].reshape(-1, 1)
        ).reshape_as(loaded[name + ".weight"])
        if not torch.equal(reconstructed.to(torch.bfloat16), loaded[name + ".weight"]):
            raise ValueError(f"saved hard codes and scales do not reproduce weight: {name}")
    result = {
        "status": (
            "dense_latent_upstream_block_reconstruction_only"
            if args.learn_dense_latent and upstream_artifacts
            else "dense_latent_block_reconstruction_only"
            if args.learn_dense_latent
            else "upstream_conditioned_block_reconstruction_only"
            if upstream_artifacts
            else "single_block_reconstruction_pilot_only"
        ),
        "source": str(source),
        "rotation_manifest": str(args.rotation_manifest.resolve()),
        "rotation_manifest_sha256": hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest(),
        "source_revision": cfg.model.revision,
        "layer": args.layer,
        "selected_modules": sorted(selected),
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "calibration_file_sha256": calibration_file_sha256,
        "calibration_texts_sha256": texts_sha256,
        "train_samples": args.train_samples,
        "held_samples": args.held_samples,
        "train_tokens": sum(int(batch["attention_mask"].sum().item()) for batch in train_batches),
        "held_tokens": sum(int(batch["attention_mask"].sum().item()) for batch in held_batches),
        "seq_len": args.seq_len,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "scale_lr": args.scale_lr,
        "threshold_lr": args.threshold_lr,
        "learn_dense_latent": args.learn_dense_latent,
        "latent_lr": args.latent_lr if args.learn_dense_latent else None,
        "latent_reg": args.latent_reg if args.learn_dense_latent else None,
        "ste_width": args.ste_width,
        "bf16_replay_exact": True,
        "masked_loss": True,
        "upstream_artifact_sha256": upstream_artifacts,
        "initial": history[0],
        "best": best,
        "best_hooked_held_output_rel_mse": best_hooked_held,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
        "history": history,
        "artifact_note": "research-only hard codes/scales/dequantized BF16 weights; no CLI materialization contract",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"initial": history[0], "best": best}), flush=True)


if __name__ == "__main__":
    main()
