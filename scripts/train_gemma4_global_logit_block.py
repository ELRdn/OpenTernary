"""Train hard G128 block scales against BF16 full-model next-token logits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from train_gemma4_rotated_block import tokenize_calibration_rows

from openternary.adapters.runtime import load_model, load_processor
from openternary.config.loader import load_config
from openternary.quant.logit_ste import hard_logit_codes, ste_logit_codes
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def ground_truth_next_token_ce(
    log_probs: torch.Tensor,
    example: dict[str, Any],
    *,
    qa_first_weight: float = 1.0,
    qa_end_weight: float = 1.0,
) -> torch.Tensor:
    """Score the next real token on text/chat rows or the answer span on QA rows."""
    ids = example["inputs"]["input_ids"]
    attention = example["inputs"]["attention_mask"]
    mask = (example["loss_mask"][..., :-1] * attention[..., 1:]).float()
    if log_probs.shape[:2] != ids.shape or mask.sum() == 0:
        raise ValueError("invalid or empty next-token calibration span")
    if example.get("kind") == "qa" and (qa_first_weight != 1.0 or qa_end_weight != 1.0):
        if mask.shape[0] != 1:
            raise ValueError("weighted QA calibration requires batch size one")
        positions = mask[0].nonzero().flatten()
        mask = mask.clone()
        mask[0, positions[0]] *= qa_first_weight
        # The final chat-template token is a newline; the preceding one ends the answer turn.
        mask[0, positions[-2] if positions.numel() >= 2 else positions[-1]] *= qa_end_weight
    selected = log_probs[..., :-1, :].gather(-1, ids[..., 1:].unsqueeze(-1)).squeeze(-1)
    return -(selected * mask).sum() / mask.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--initial-block-dir", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-hard-codes", action="store_true")
    parser.add_argument("--code-lr", type=float, default=0.005)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--teacher-top1-weight", type=float, default=0.0)
    parser.add_argument("--ground-truth-weight", type=float, default=0.0)
    parser.add_argument("--qa-first-weight", type=float, default=1.0)
    parser.add_argument("--qa-end-weight", type=float, default=1.0)
    parser.add_argument("--japanese-loss-weight", type=float, default=1.0)
    parser.add_argument("--shuffle-train", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--upstream-block-dir", type=Path)
    parser.add_argument("--fixed-q-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or not 0 <= args.layer < 35
        or args.steps < 1
        or args.eval_every < 1
        or args.lr <= 0
        or args.code_lr <= 0
        or args.top_k < 2
        or args.teacher_top1_weight < 0
        or args.ground_truth_weight < 0
        or min(args.qa_first_weight, args.qa_end_weight) <= 0
        or args.japanese_loss_weight <= 0
        or args.seq_len < 8
        or (args.fixed_q_report is not None and (args.layer != 1 or args.upstream_block_dir is None))
    ):
        raise ValueError("invalid training arguments or output exists")
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    prefix = f"model.language_model.layers.{args.layer}."
    all_selected = {name: matrix.to("cuda:0") for name, matrix in rotations.items() if name.startswith(prefix)}
    if len(all_selected) != (7 if args.layer < 15 else 5):
        raise ValueError("incomplete canonical target set")
    fixed_q_name = f"{prefix}self_attn.q_proj" if args.fixed_q_report is not None else None
    if fixed_q_name is not None and fixed_q_name not in all_selected:
        raise ValueError("fixed q projection is not canonical")
    selected = {name: matrix for name, matrix in all_selected.items() if name != fixed_q_name}
    report_path = args.initial_block_dir / f"layer{args.layer}.json"
    artifact_path = report_path.with_suffix(".safetensors")
    initial_report = json.loads(report_path.read_text(encoding="utf-8"))
    initial_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(args.rotation_manifest.read_bytes()).hexdigest()
    if (
        initial_report.get("artifact_sha256") != initial_sha256
        or initial_report.get("rotation_manifest_sha256") != manifest_sha256
        or Path(initial_report.get("source", "")).resolve() != source
        or set(initial_report.get("selected_modules", [])) != set(all_selected)
    ):
        raise ValueError("initial block provenance mismatch")
    initial = load_file(str(artifact_path), device="cpu")
    calibration_raw = args.calibration_file.read_bytes()
    calibration = json.loads(calibration_raw)
    if calibration.get("schema_version") != 1 or calibration.get("status") != "calibration_only":
        raise ValueError("invalid calibration file")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    batches = {
        split: tokenize_calibration_rows(calibration[split], processor, args.seq_len) for split in ("train", "held")
    }
    layer = model.get_submodule(f"model.language_model.layers.{args.layer}")
    captured: dict[str, Any] = {}

    def capture_input(_module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
        captured["positional"] = copy.deepcopy(positional)
        captured["keyword"] = copy.deepcopy(keyword)

    def capture_output(_module: torch.nn.Module, _values: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["output"] = output.detach().clone()

    pre = layer.register_forward_pre_hook(capture_input, with_kwargs=True)
    post = layer.register_forward_hook(capture_output)
    teacher_data: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    with torch.no_grad():
        for split in ("train", "held"):
            for batch, calibration_row in zip(batches[split], calibration[split], strict=True):
                if calibration_row.get("language") not in {"en", "ja"}:
                    raise ValueError("calibration language must be en or ja")
                captured.clear()
                inputs = {
                    "input_ids": batch["input_ids"].unsqueeze(0).to("cuda:0"),
                    "attention_mask": batch["attention_mask"].unsqueeze(0).to("cuda:0"),
                    "use_cache": False,
                }
                logits = model(**inputs).logits.float()
                if set(captured) != {"positional", "keyword", "output"}:
                    raise ValueError("incomplete BF16 block capture")
                if not torch.equal(layer(*captured["positional"], **captured["keyword"]), captured["output"]):
                    raise ValueError("BF16 block replay mismatch")
                log_probs = functional.log_softmax(logits, dim=-1)
                top_log_probs, indices = log_probs.topk(args.top_k, dim=-1)
                top_probs = top_log_probs.exp()
                rest_prob = (1 - top_probs.sum(dim=-1)).clamp_min(1e-8)
                teacher_data[split].append(
                    {
                        "inputs": inputs,
                        "indices": indices.detach(),
                        "top_probs": top_probs.detach(),
                        "top_log_probs": top_log_probs.detach(),
                        "rest_prob": rest_prob.detach(),
                        "loss_mask": batch.get("loss_mask", batch["attention_mask"]).unsqueeze(0).to("cuda:0"),
                        "language": calibration_row["language"],
                        "kind": calibration_row["kind"],
                    }
                )
    pre.remove()
    post.remove()
    if args.shuffle_train:
        random.Random(42).shuffle(teacher_data["train"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    upstream_artifacts: dict[str, str] = {}
    upstream_hooks = []
    if args.upstream_block_dir is not None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for previous_layer in range(args.layer):
                previous_path = args.upstream_block_dir / f"layer{previous_layer}.json"
                previous_artifact = previous_path.with_suffix(".safetensors")
                previous_report = json.loads(previous_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(previous_artifact.read_bytes()).hexdigest()
                names = {
                    name for name in rotations if name.startswith(f"model.language_model.layers.{previous_layer}.")
                }
                if (
                    previous_report.get("artifact_sha256") != digest
                    or previous_report.get("rotation_manifest_sha256") != manifest_sha256
                    or Path(previous_report.get("source", "")).resolve() != source
                    or set(previous_report.get("selected_modules", [])) != names
                ):
                    raise ValueError(f"upstream artifact provenance mismatch: {previous_layer}")
                tensors = load_file(str(previous_artifact), device="cpu")
                for name in sorted(names):
                    weight = tensors[name + ".weight"]
                    codes = tensors[name + ".codes"]
                    scales = tensors[name + ".scales"]
                    reconstructed = (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight)
                    if not torch.equal(reconstructed.to(torch.bfloat16), weight):
                        raise ValueError(f"invalid upstream hard weight: {name}")
                    parameters[name + ".weight"].copy_(weight.to("cuda:0"))
                    matrix = rotations[name].to("cuda:0")

                    def rotate_input(
                        _module: torch.nn.Module,
                        values: tuple[torch.Tensor, ...],
                        *,
                        rotation: torch.Tensor = matrix,
                    ) -> tuple[torch.Tensor, ...]:
                        transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), rotation)
                        return (transformed.to(values[0].dtype), *values[1:])

                    upstream_hooks.append(model.get_submodule(name).register_forward_pre_hook(rotate_input))
                upstream_artifacts[str(previous_layer)] = digest

    fixed_q_sha256 = None
    fixed_q_hook = None
    if args.fixed_q_report is not None:
        assert fixed_q_name is not None
        fixed_q_artifact = args.fixed_q_report.with_suffix(".safetensors")
        fixed_q_report = json.loads(args.fixed_q_report.read_text(encoding="utf-8"))
        fixed_q_sha256 = hashlib.sha256(fixed_q_artifact.read_bytes()).hexdigest()
        if (
            fixed_q_report.get("artifact_sha256") != fixed_q_sha256
            or fixed_q_report.get("rotation_manifest_sha256") != manifest_sha256
            or fixed_q_report.get("upstream_artifact_sha256") != upstream_artifacts["0"]
            or Path(fixed_q_report.get("source", "")).resolve() != source
            or fixed_q_report.get("module") != fixed_q_name
        ):
            raise ValueError("fixed q artifact provenance mismatch")
        fixed_q = load_file(str(fixed_q_artifact), device="cpu")
        weight = fixed_q["weight"]
        codes = fixed_q["codes"]
        scales = fixed_q["scales"]
        if set(fixed_q) != {"rotation", "weight", "codes", "scales"} or not torch.equal(
            (codes.float().reshape(-1, 128) * scales.reshape(-1, 1)).reshape_as(weight).to(torch.bfloat16),
            weight,
        ):
            raise ValueError("fixed q hard weight mismatch")
        with torch.no_grad():
            parameters[fixed_q_name + ".weight"].copy_(weight.to("cuda:0"))
        fixed_q_rotation = fixed_q["rotation"].to("cuda:0")

        def rotate_fixed_q(_module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), fixed_q_rotation)
            return (transformed.to(values[0].dtype), *values[1:])

        fixed_q_hook = model.get_submodule(fixed_q_name).register_forward_pre_hook(rotate_fixed_q)

    states = {}
    hooks = []
    for name, rotation in selected.items():
        codes = initial[name + ".codes"].to("cuda:0").float()
        scales = initial[name + ".scales"].to("cuda:0").reshape(-1, 1)
        weight = initial[name + ".weight"]
        if not torch.equal((codes.reshape(-1, 128) * scales).reshape_as(weight).to(torch.bfloat16).cpu(), weight):
            raise ValueError(f"initial hard weight mismatch: {name}")
        states[name] = {
            "codes": codes,
            "scales": scales,
            "rotation": rotation,
            "delta": torch.nn.Parameter(torch.zeros_like(scales)),
        }
        if args.train_hard_codes:
            states[name]["code_logits"] = torch.nn.Parameter(codes * 0.55)

        def current_codes(state: dict[str, Any]) -> torch.Tensor:
            if not args.train_hard_codes:
                return state["codes"]
            return ste_logit_codes(state["code_logits"])

        def replace_output(
            current: torch.nn.Module,
            values: tuple[torch.Tensor, ...],
            original: torch.Tensor,
            *,
            module_name: str = name,
        ) -> torch.Tensor:
            state = states[module_name]
            dense_weight = (current_codes(state).reshape(-1, 128) * state["scales"] * state["delta"].exp()).reshape_as(
                state["codes"]
            )
            transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), state["rotation"])
            bias = current.bias.float() if current.bias is not None else None
            return functional.linear(transformed, dense_weight, bias).to(original.dtype)

        hooks.append(model.get_submodule(name).register_forward_hook(replace_output))

    def objective(example: dict[str, Any]) -> torch.Tensor:
        student_logits = model(**example["inputs"]).logits.float()
        student_log_probs = functional.log_softmax(student_logits, dim=-1)
        student_top_log_probs = student_log_probs.gather(-1, example["indices"])
        student_rest_log_prob = (1 - student_top_log_probs.exp().sum(dim=-1)).clamp_min(1e-8).log()
        top_terms = example["top_probs"] * (example["top_log_probs"] - student_top_log_probs)
        rest_terms = example["rest_prob"] * (example["rest_prob"].log() - student_rest_log_prob)
        per_token = top_terms.sum(dim=-1) + rest_terms
        mask = example["loss_mask"]
        kl = (per_token * mask).sum() / mask.sum()
        if args.teacher_top1_weight:
            teacher_top1_log_prob = student_top_log_probs[..., 0]
            top1_ce = -(teacher_top1_log_prob * mask).sum() / mask.sum()
            objective_value = kl + args.teacher_top1_weight * top1_ce
        else:
            objective_value = kl
        if args.ground_truth_weight:
            objective_value = objective_value + args.ground_truth_weight * ground_truth_next_token_ce(
                student_log_probs,
                example,
                qa_first_weight=args.qa_first_weight,
                qa_end_weight=args.qa_end_weight,
            )
        return objective_value * (args.japanese_loss_weight if example["language"] == "ja" else 1.0)

    def evaluate(split: str) -> float:
        losses = []
        for example in teacher_data[split]:
            losses.append(objective(example).detach())
        return float(torch.stack(losses).mean().item())

    optimizer_groups = [{"params": [state["delta"] for state in states.values()], "lr": args.lr}]
    if args.train_hard_codes:
        optimizer_groups.append({"params": [state["code_logits"] for state in states.values()], "lr": args.code_lr})
    optimizer = torch.optim.Adam(optimizer_groups)
    trainable = [parameter for group in optimizer_groups for parameter in group["params"]]
    metric_label = "objective" if args.teacher_top1_weight or args.ground_truth_weight else "kl"
    held_key = f"held_{metric_label}"

    def code_change_ratio() -> float:
        if not args.train_hard_codes:
            return 0.0
        changed = 0
        total = 0
        for state in states.values():
            hard = hard_logit_codes(state["code_logits"])
            changed += int((hard != state["codes"]).sum().item())
            total += hard.numel()
        return changed / total

    history = []
    best = None
    best_state = None
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            example = teacher_data["train"][(step - 1) % len(teacher_data["train"])]
            loss = objective(example)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite logit distillation loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                row = {
                    "step": step,
                    f"train_{metric_label}": evaluate("train"),
                    held_key: evaluate("held"),
                    "hard_code_change_ratio": code_change_ratio(),
                }
            history.append(row)
            if best is None or row[held_key] < best[held_key]:
                best = row
                best_state = {
                    name: {
                        "delta": state["delta"].detach().clone(),
                        "code_logits": state["code_logits"].detach().clone() if args.train_hard_codes else None,
                    }
                    for name, state in states.items()
                }
            print(json.dumps(row), flush=True)
    assert best is not None and best_state is not None
    with torch.no_grad():
        for name, saved in best_state.items():
            states[name]["delta"].copy_(saved["delta"])
            if saved["code_logits"] is not None:
                states[name]["code_logits"].copy_(saved["code_logits"])
        if abs(evaluate("held") - best[held_key]) > 1e-7:
            raise ValueError("best logit KD state did not replay")
    args.output_dir.mkdir(parents=True)
    payload = {}
    for name in selected:
        state = states[name]
        scales = state["scales"] * state["delta"].exp()
        codes = (
            hard_logit_codes(state["code_logits"].detach()).to(torch.int8)
            if args.train_hard_codes
            else state["codes"].to(torch.int8)
        )
        weight = (codes.float().reshape(-1, 128) * scales).reshape_as(state["codes"])
        payload[name + ".weight"] = weight.to(torch.bfloat16).detach().cpu().contiguous()
        payload[name + ".codes"] = codes.detach().cpu().contiguous()
        payload[name + ".scales"] = scales.detach().reshape(-1).cpu().contiguous()
        payload[name + ".thresholds"] = initial[name + ".thresholds"]
    saved_path = args.output_dir / f"layer{args.layer}.safetensors"
    save_file(payload, str(saved_path))
    reloaded = load_file(str(saved_path), device="cpu")
    for name in selected:
        reconstructed = (
            reloaded[name + ".codes"].float().reshape(-1, 128) * reloaded[name + ".scales"].reshape(-1, 1)
        ).reshape_as(reloaded[name + ".weight"])
        if not torch.equal(reconstructed.to(torch.bfloat16), reloaded[name + ".weight"]):
            raise ValueError(f"saved logit KD hard weight mismatch: {name}")
    artifact_sha256 = hashlib.sha256(saved_path.read_bytes()).hexdigest()
    report = {
        "status": (
            "global_logit_kd_hard_codes_upstream_block"
            if args.train_hard_codes and upstream_artifacts
            else "global_logit_kd_hard_codes_block"
            if args.train_hard_codes
            else "global_logit_kd_scale_only_upstream_block"
            if upstream_artifacts
            else "global_logit_kd_scale_only_block"
        ),
        "source": str(source),
        "rotation_manifest_sha256": manifest_sha256,
        "calibration_file_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "initial_artifact_sha256": initial_sha256,
        "upstream_artifact_sha256": upstream_artifacts,
        "fixed_q_artifact_sha256": fixed_q_sha256,
        "layer": args.layer,
        "selected_modules": sorted(selected),
        "bf16_replay_exact": True,
        "train_samples": len(teacher_data["train"]),
        "held_samples": len(teacher_data["held"]),
        "steps": args.steps,
        "top_k": args.top_k,
        "teacher_top1_weight": args.teacher_top1_weight,
        "ground_truth_weight": args.ground_truth_weight,
        "qa_first_weight": args.qa_first_weight,
        "qa_end_weight": args.qa_end_weight,
        "japanese_loss_weight": args.japanese_loss_weight,
        "shuffle_train": args.shuffle_train,
        "train_hard_codes": args.train_hard_codes,
        "code_lr": args.code_lr if args.train_hard_codes else None,
        "seq_len": args.seq_len,
        "best": best,
        "history": history,
        "artifact_sha256": artifact_sha256,
        "elapsed_s": time.monotonic() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
    }
    (args.output_dir / f"layer{args.layer}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for hook in hooks:
        hook.remove()
    for hook in upstream_hooks:
        hook.remove()
    if fixed_q_hook is not None:
        fixed_q_hook.remove()
    print(json.dumps({"best": best, "artifact_sha256": artifact_sha256}), flush=True)


if __name__ == "__main__":
    main()
