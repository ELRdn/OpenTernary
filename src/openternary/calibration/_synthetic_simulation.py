"""TEST ONLY — synthetic tiny calibration simulation.

Do not import from production runner (real-model path).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import typing

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from openternary.calibration.losses import mse_loss
from openternary.calibration.optimizer import build_scale_params, get_effective_scales


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_synthetic_tiny(
    app_config: typing.Any,
    output_dir: pathlib.Path | str,
    train_texts: list[str],
    held_texts: list[str],
    train_hashes: list[str],
    held_hashes: list[str],
    smoke_hashes: list[str],
    contamination: tuple[bool, bool, bool],
    requested_dataset: str,
    effective_dataset: str,
    dataset_fallback: bool,
    resume: bool = False,
    resume_from: pathlib.Path | str | None = None,
) -> dict[str, typing.Any]:
    """Synthetic dummy calibration for fast CI (TEST ONLY)."""
    if torch is None:
        raise ImportError("torch is required")
    import pathlib as _pl

    out = _pl.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import yaml

    cfg_path = out / "config.yaml"
    try:
        cfg_dict = app_config.model_dump() if hasattr(app_config, "model_dump") else dict(app_config)
    except Exception:
        cfg_dict = {}
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    calib_cfg = app_config.calibration
    num_samples = int(calib_cfg.num_samples)
    seq_len = int(calib_cfg.seq_len)

    from openternary.quant.grouping import quantize_groupwise
    from openternary.quant.ternary import quantize_absmean

    q_cfg = app_config.quantization
    scale_granularity = str(getattr(q_cfg, "scale_granularity", "per_tensor"))
    group_size = int(getattr(q_cfg, "group_size", 8))
    dummy_weight = torch.randn(8, 8, dtype=torch.float32)
    if scale_granularity == "per_tensor":
        tt = quantize_absmean(dummy_weight)
        orig_scales = torch.tensor([tt.scale], dtype=torch.float32)
        codes = tt.codes
        zero_mask = orig_scales == 0
    else:
        res = quantize_groupwise(dummy_weight, group_size)
        orig_scales = res.scales.clone()
        codes = res.codes
        zero_mask = orig_scales == 0

    raw_param, zero_mask_t = build_scale_params(orig_scales, zero_mask)
    effective_initial = get_effective_scales(raw_param, zero_mask_t)
    if not torch.allclose(effective_initial, orig_scales, atol=1e-6):
        raise ValueError(f"effective initial mismatch: {effective_initial} vs {orig_scales}")

    optimizer = torch.optim.Adam([raw_param], lr=float(calib_cfg.lr))  # type: ignore[union-attr]
    target_names = ["layer_0", "layer_1"]
    torch.manual_seed(int(calib_cfg.seed) if calib_cfg.seed is not None else int(app_config.seed))
    steps = int(calib_cfg.steps)
    ckpt_dir = out / "artifacts" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if resume or resume_from is not None:
        if resume_from is not None:
            ckpt_path = _pl.Path(resume_from)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"no resumable checkpoint found: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            raw_param.data = ckpt["raw_param"]
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt["step"]) + 1
        elif resume:
            ckpts = sorted(ckpt_dir.glob("step_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"no resumable checkpoint found in {ckpt_dir}")
            latest = ckpts[-1]
            ckpt = torch.load(str(latest), map_location="cpu")
            raw_param.data = ckpt["raw_param"]
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt["step"]) + 1

    def make_activations(n: int) -> list[torch.Tensor]:  # type: ignore[type-arg]
        acts: list[torch.Tensor] = []
        for _ in range(n):
            acts.append(torch.randn(seq_len, 8))
        return acts

    train_inputs = make_activations(len(train_texts))
    held_inputs = make_activations(len(held_texts))
    w_teacher = dummy_weight
    train_teacher_outs = [inp @ w_teacher.T for inp in train_inputs]
    held_teacher_outs = [inp @ w_teacher.T for inp in held_inputs]

    from openternary.quant.grouping import GroupwiseResult, dequantize_groupwise

    initial_loss: float | None = None
    best_loss = float("inf")
    loss_history: list[float] = []

    for step in range(start_step, steps):
        step_losses: list[float] = []
        optimizer.zero_grad()
        for _layer_name in target_names:  # noqa: B007
            for inp, tout in zip(train_inputs, train_teacher_outs, strict=False):  # noqa: B905
                eff = get_effective_scales(raw_param, zero_mask_t)
                if scale_granularity == "per_tensor":
                    w_hat = codes.to(torch.float32) * eff[0]
                else:
                    res_tmp = GroupwiseResult(
                        codes=codes,
                        scales=eff,
                        shape=tuple(dummy_weight.shape),
                        orig_dtype=str(dummy_weight.dtype),
                        group_size=group_size,
                        grouping_scheme="last-dim-rowwise-v1",
                    )
                    w_hat = dequantize_groupwise(res_tmp)
                y_hat = F.linear(inp, w_hat)  # type: ignore[union-attr]
                loss = mse_loss(tout, y_hat)
                loss.backward()
                step_losses.append(float(loss.item()))
        step_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
        if initial_loss is None:
            initial_loss = step_loss
        best_loss = min(best_loss, step_loss)
        loss_history.append(step_loss)
        optimizer.step()
        if (step + 1) % int(calib_cfg.checkpoint_interval) == 0 or (step + 1) == steps:
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            torch.save(
                {
                    "raw_param": raw_param.detach().cpu(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "loss": step_loss,
                },
                str(ckpt_path),
            )

    final_loss = loss_history[-1] if loss_history else 0.0
    held_losses: list[float] = []
    eff_final = get_effective_scales(raw_param, zero_mask_t)
    if scale_granularity == "per_tensor":
        w_hat_final = codes.to(torch.float32) * eff_final[0]
    else:
        res_tmp = GroupwiseResult(
            codes=codes,
            scales=eff_final,
            shape=tuple(dummy_weight.shape),
            orig_dtype=str(dummy_weight.dtype),
            group_size=group_size,
            grouping_scheme="last-dim-rowwise-v1",
        )
        w_hat_final = dequantize_groupwise(res_tmp)
    for inp, tout in zip(held_inputs, held_teacher_outs, strict=False):  # noqa: B905
        y_hat = F.linear(inp, w_hat_final)  # type: ignore[union-attr]
        held_losses.append(float(mse_loss(tout, y_hat).item()))
    if scale_granularity == "per_tensor":
        w_hat_init = codes.to(torch.float32) * orig_scales[0]
    else:
        res_init = GroupwiseResult(
            codes=codes,
            scales=orig_scales,
            shape=tuple(dummy_weight.shape),
            orig_dtype=str(dummy_weight.dtype),
            group_size=group_size,
            grouping_scheme="last-dim-rowwise-v1",
        )
        w_hat_init = dequantize_groupwise(res_init)
    held_before = [
        float(mse_loss(tout, F.linear(inp, w_hat_init)).item())  # type: ignore[union-attr]
        for inp, tout in zip(held_inputs, held_teacher_outs, strict=False)  # noqa: B905
    ]
    held_loss_before_val = sum(held_before) / len(held_before) if held_before else 0.0
    held_loss_after_val = sum(held_losses) / len(held_losses) if held_losses else 0.0

    calibrated_snapshot = out / "artifacts" / "calibrated_snapshot"
    calibrated_snapshot.mkdir(parents=True, exist_ok=True)
    (calibrated_snapshot / "README.txt").write_text(
        "calibrated snapshot placeholder (synthetic test only)", encoding="utf-8"
    )

    calib_json = {
        "method": str(calib_cfg.method),
        "dataset": str(effective_dataset),
        "requested_dataset": str(requested_dataset),
        "effective_dataset": str(effective_dataset),
        "dataset_fallback": bool(dataset_fallback),
        "num_samples": int(num_samples),
        "seq_len": int(seq_len),
        "steps": int(steps),
        "lr": float(calib_cfg.lr),
        "loss_history": loss_history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "relative_improvement": ((initial_loss - final_loss) / initial_loss) if initial_loss else 0.0,
        "heldout_loss_before": held_loss_before_val,
        "heldout_loss_after": held_loss_after_val,
        "heldout_improvement": ((held_loss_before_val - held_loss_after_val) / held_loss_before_val)
        if held_loss_before_val
        else 0.0,
        "train_fingerprint": _hash_text("".join(train_hashes)),
        "heldout_fingerprint": _hash_text("".join(held_hashes)),
        "train_sample_hashes": train_hashes,
        "heldout_sample_hashes": held_hashes,
        "smoke_sample_hashes": smoke_hashes,
        "contamination_train_smoke": contamination[0],
        "contamination_held_smoke": contamination[1],
        "contamination_train_held": contamination[2],
        "scale_fingerprint_before": hashlib.sha256(orig_scales.numpy().tobytes()).hexdigest()
        if orig_scales.numel()
        else "",
        "scale_fingerprint_after": hashlib.sha256(eff_final.detach().cpu().numpy().tobytes()).hexdigest()
        if eff_final.numel()
        else "",
        "mean_abs_scale_delta": float((eff_final - orig_scales).abs().mean().item()) if eff_final.numel() else 0.0,
        "zero_group_exact": bool((eff_final[zero_mask] == 0).all().item()) if zero_mask.any().item() else True,
        "trainable_param_count": int(raw_param.numel() - int(zero_mask.sum().item()))
        if zero_mask.numel()
        else int(raw_param.numel()),
        "total_param_count": int(raw_param.numel()),
        "used_dummy_simulation": True,
        "teacher_snapshot": None,
    }
    with open(out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_json, f, indent=2, ensure_ascii=False)

    metrics = {
        "status": "completed",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "heldout_before": held_loss_before_val,
        "heldout_after": held_loss_after_val,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "run_dir": str(out),
        "calibrated_snapshot": str(calibrated_snapshot),
        "calibration_json": str(out / "calibration.json"),
        "loss_history": loss_history,
    }
