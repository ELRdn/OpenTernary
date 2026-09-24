"""Test whether full-model teacher hidden loss can train a hard rotated block on RX 9070 XT."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from train_gemma4_rotated_block import tokenize_calibration_rows

from openternary.adapters.runtime import load_model, load_processor
from openternary.config.loader import load_config
from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim, load_rotation_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--objective", choices=("hidden", "logit_kl"), default="hidden")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    source = args.source.resolve()
    rotations, _ = load_rotation_plan(args.rotation_manifest, source)
    selected = {
        name: matrix.to("cuda:0")
        for name, matrix in rotations.items()
        if name.startswith("model.language_model.layers.0.")
    }
    if len(selected) != 7:
        raise ValueError("expected seven first-block targets")
    report_path = args.block_dir / "layer0.json"
    artifact_path = report_path.with_suffix(".safetensors")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if report.get("artifact_sha256") != artifact_sha256 or set(report.get("selected_modules", [])) != set(selected):
        raise ValueError("block provenance mismatch")
    artifact = load_file(str(artifact_path), device="cpu")
    calibration_raw = args.calibration_file.read_bytes()
    calibration = json.loads(calibration_raw)
    row = next(row for row in calibration["train"] if row["kind"] == "chat" and row["language"] == "ja")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={"model.id": str(source), "device": "cuda", "dtype": "bf16", "seed": 42},
    )
    processor = load_processor(cfg, source)
    model = load_model(cfg, source, torch.bfloat16, {"": "cuda:0"}).eval()
    batch = tokenize_calibration_rows([row], processor, 128)[0]
    inputs = {
        "input_ids": batch["input_ids"].unsqueeze(0).to("cuda:0"),
        "attention_mask": batch["attention_mask"].unsqueeze(0).to("cuda:0"),
        "use_cache": False,
    }
    last = model.get_submodule("model.language_model.layers.34")
    capture = {}

    def capture_output(_module: torch.nn.Module, _values: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        capture["hidden"] = output

    final_hook = last.register_forward_hook(capture_output)
    with torch.no_grad():
        teacher_output = model(**inputs)
        teacher = capture["hidden"].detach().clone()
        teacher_logits = teacher_output.logits.detach().float() if args.objective == "logit_kl" else None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    states = {}
    hooks = []
    for name, rotation in selected.items():
        codes = artifact[name + ".codes"].to("cuda:0").float()
        scales = artifact[name + ".scales"].to("cuda:0").reshape(-1, 1)
        state = {
            "codes": codes,
            "scales": scales,
            "rotation": rotation,
            "delta": torch.nn.Parameter(torch.zeros_like(scales)),
        }
        states[name] = state

        def replace_output(
            current: torch.nn.Module,
            values: tuple[torch.Tensor, ...],
            original: torch.Tensor,
            *,
            module_name: str = name,
        ) -> torch.Tensor:
            item = states[module_name]
            weight = (item["codes"].reshape(-1, 128) * item["scales"] * item["delta"].exp()).reshape_as(item["codes"])
            transformed = apply_block_rotation(hadamard_last_dim(values[0].float(), 128), item["rotation"])
            bias = current.bias.float() if current.bias is not None else None
            return functional.linear(transformed, weight, bias).to(original.dtype)

        hooks.append(model.get_submodule(name).register_forward_hook(replace_output))
    optimizer = torch.optim.Adam([state["delta"] for state in states.values()], lr=0.01)
    torch.cuda.reset_peak_memory_stats("cuda:0")
    optimizer.zero_grad(set_to_none=True)
    capture.clear()
    student_output = model(**inputs)
    student = capture["hidden"]
    if teacher_logits is None:
        loss = (student.float() - teacher.float()).square().mean() / teacher.float().square().mean()
    else:
        teacher_log_probs = functional.log_softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = functional.log_softmax(student_output.logits.float(), dim=-1)
        per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        mask = inputs["attention_mask"]
        loss = (per_token * mask).sum() / mask.sum()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite global hidden loss")
    initial_loss = float(loss.item())
    loss.backward()
    grad_norms = {name: float(state["delta"].grad.norm().item()) for name, state in states.items()}
    if not all(torch.isfinite(state["delta"].grad).all() for state in states.values()):
        raise ValueError("nonfinite global hidden gradient")
    optimizer.step()
    for hook in hooks:
        hook.remove()
    final_hook.remove()
    result = {
        "source": str(source),
        "artifact_sha256": artifact_sha256,
        "calibration_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "sample_id": row["id"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "objective": args.objective,
        "initial_global_hidden_rel_mse": initial_loss,
        "grad_norms": grad_norms,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
        "elapsed_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
