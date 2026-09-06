"""TEST ONLY — synthetic tiny calibration simulation (Phase 4.2 threshold aware).

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
from openternary.calibration.optimizer import (
    build_scale_params,
    build_threshold_params,
    get_effective_scales,
    get_effective_threshold_ratio,
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_tensor(t: torch.Tensor) -> str:  # type: ignore[type-arg]
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


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
    """Synthetic dummy calibration for fast CI (TEST ONLY) — threshold aware."""
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
    # Phase 4.2 threshold config (backward compatible)
    threshold_enabled = bool(getattr(calib_cfg, "threshold_enabled", False))
    threshold_init_ratio = float(getattr(calib_cfg, "threshold_init_ratio", 0.5))
    threshold_eps = float(getattr(calib_cfg, "threshold_eps", 0.01))
    threshold_ste_width = float(getattr(calib_cfg, "threshold_ste_width", 0.1))
    threshold_lr = getattr(calib_cfg, "threshold_lr", None)
    threshold_lr = float(threshold_lr) if threshold_lr is not None else float(calib_cfg.lr)

    # Deterministic dummy weight: seed before generation for reproducibility (resume vs continuous)
    _seed = int(calib_cfg.seed) if calib_cfg.seed is not None else int(app_config.seed)
    torch.manual_seed(_seed)
    dummy_weight = torch.randn(8, 8, dtype=torch.float32)
    if scale_granularity == "per_tensor":
        tt = quantize_absmean(dummy_weight)
        orig_scales = torch.tensor([tt.scale], dtype=torch.float32)
        codes_fixed = tt.codes
        zero_mask = orig_scales == 0
        reference_scales = orig_scales.clone()
    else:
        res = quantize_groupwise(dummy_weight, group_size)
        orig_scales = res.scales.clone()
        codes_fixed = res.codes
        zero_mask = orig_scales == 0
        reference_scales = orig_scales.clone()

    # scale params (learnable reconstruction scale)
    raw_param, zero_mask_t = build_scale_params(orig_scales, zero_mask)
    effective_initial = get_effective_scales(raw_param, zero_mask_t)
    if not torch.allclose(effective_initial, orig_scales, atol=1e-6):
        raise ValueError(f"effective initial mismatch: {effective_initial} vs {orig_scales}")

    # threshold params (if enabled)
    raw_threshold = None
    thr_zero_mask = None
    if threshold_enabled:
        # per_tensor: 1 group, per_group: num_groups
        n_thr = int(orig_scales.numel())
        # reuse zero_mask for threshold (zero ref groups threshold is masked)
        raw_threshold, thr_zero_mask = build_threshold_params(
            n_thr, init_ratio=threshold_init_ratio, eps=threshold_eps, device=orig_scales.device, zero_mask=zero_mask
        )
        thr_initial = get_effective_threshold_ratio(raw_threshold, eps=threshold_eps)
        # step0 parity: threshold 0.5 should reproduce naive codes
        # verify
        from openternary.quant.threshold import hard_threshold_codes

        if scale_granularity == "per_tensor":
            hard_codes_step0 = hard_threshold_codes(
                dummy_weight, float(reference_scales[0].item()), float(thr_initial[0].item())
            )
        else:
            hard_codes_step0 = hard_threshold_codes(dummy_weight, reference_scales, thr_initial, group_size=group_size)
        # codes_fixed is naive codes (threshold 0.5). For synthetic we check parity when init 0.5
        if threshold_init_ratio == 0.5 and not torch.equal(hard_codes_step0, codes_fixed):
            raise ValueError("threshold step0 parity failed: hard codes != naive codes")

    # optimizer: scale + threshold
    if threshold_enabled:
        assert raw_threshold is not None
        # separate param groups for different LRs
        optimizer = torch.optim.Adam(  # type: ignore[union-attr]
            [
                {"params": [raw_param], "lr": float(calib_cfg.lr)},
                {"params": [raw_threshold], "lr": threshold_lr},
            ]
        )
    else:
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
            if threshold_enabled and raw_threshold is not None and "raw_threshold" in ckpt:
                raw_threshold.data = ckpt["raw_threshold"]
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt["step"]) + 1
        elif resume:
            ckpts = sorted(ckpt_dir.glob("step_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"no resumable checkpoint found in {ckpt_dir}")
            latest = ckpts[-1]
            ckpt = torch.load(str(latest), map_location="cpu")
            raw_param.data = ckpt["raw_param"]
            if threshold_enabled and raw_threshold is not None and "raw_threshold" in ckpt:
                raw_threshold.data = ckpt["raw_threshold"]
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
                eff_scale = get_effective_scales(raw_param, zero_mask_t)
                if threshold_enabled:
                    assert raw_threshold is not None and thr_zero_mask is not None
                    thr_ratio = get_effective_threshold_ratio(raw_threshold, eps=threshold_eps)
                    # STE codes
                    from openternary.quant.threshold import ste_threshold_codes

                    if scale_granularity == "per_tensor":
                        thr_scalar = thr_ratio[0]
                        ref_scalar = float(reference_scales[0].item())
                        codes_ste = ste_threshold_codes(
                            dummy_weight, ref_scalar, thr_scalar.view(1), ste_width=threshold_ste_width
                        )
                        w_hat = codes_ste * eff_scale[0]
                    else:
                        codes_ste = ste_threshold_codes(
                            dummy_weight,
                            reference_scales,
                            thr_ratio,
                            group_size=group_size,
                            ste_width=threshold_ste_width,
                        )
                        # w_hat = codes_ste * expanded eff_scale
                        # need to expand eff_scale per-group like dequantize does
                        # use GroupwiseResult style but with ste codes
                        # expand eff_scale to per-element via _expand_per_group
                        from openternary.quant.threshold import _expand_per_group

                        eff_exp = _expand_per_group(eff_scale, tuple(dummy_weight.shape), group_size)
                        w_hat = codes_ste * eff_exp
                else:
                    if scale_granularity == "per_tensor":
                        w_hat = codes_fixed.to(torch.float32) * eff_scale[0]
                    else:
                        res_tmp = GroupwiseResult(
                            codes=codes_fixed,
                            scales=eff_scale,
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
            ckpt_dict: dict[str, typing.Any] = {
                "raw_param": raw_param.detach().cpu(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "loss": step_loss,
                "threshold_enabled": threshold_enabled,
            }
            if threshold_enabled:
                assert raw_threshold is not None
                ckpt_dict["raw_threshold"] = raw_threshold.detach().cpu()
                ckpt_dict["reference_scales"] = reference_scales.detach().cpu()
            # P0: temp→atomic replace + byte size log
            tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
            torch.save(ckpt_dict, str(tmp_ckpt))
            tmp_ckpt.replace(ckpt_path)
            try:
                sz = ckpt_path.stat().st_size
                print(f"[checkpoint] saved {ckpt_path.name} ({sz} bytes)", flush=True)
            except Exception:
                pass

    final_loss = loss_history[-1] if loss_history else 0.0

    # Final evaluation with threshold
    if threshold_enabled:
        assert raw_threshold is not None
        eff_final = get_effective_scales(raw_param, zero_mask_t)
        thr_final = get_effective_threshold_ratio(raw_threshold, eps=threshold_eps)
        from openternary.quant.threshold import hard_threshold_codes

        if scale_granularity == "per_tensor":
            codes_final = hard_threshold_codes(
                dummy_weight, float(reference_scales[0].item()), float(thr_final[0].item())
            )
            w_hat_final = codes_final.to(torch.float32) * eff_final[0]
            codes_init = hard_threshold_codes(dummy_weight, float(reference_scales[0].item()), 0.5)
            w_hat_init = codes_init.to(torch.float32) * orig_scales[0]
            code_fingerprint_before = hashlib.sha256(codes_init.numpy().tobytes()).hexdigest()
            code_fingerprint_after = hashlib.sha256(codes_final.numpy().tobytes()).hexdigest()
            code_change_ratio = float((codes_init != codes_final).float().mean().item())
            zero_before = float((codes_init == 0).float().mean().item())
            zero_after = float((codes_final == 0).float().mean().item())
        else:
            codes_final = hard_threshold_codes(dummy_weight, reference_scales, thr_final, group_size=group_size)
            from openternary.quant.threshold import _expand_per_group

            eff_exp_final = _expand_per_group(eff_final, tuple(dummy_weight.shape), group_size)
            w_hat_final = codes_final.to(torch.float32) * eff_exp_final
            codes_init = hard_threshold_codes(
                dummy_weight, reference_scales, torch.full_like(thr_final, 0.5), group_size=group_size
            )
            eff_exp_init = _expand_per_group(orig_scales, tuple(dummy_weight.shape), group_size)
            w_hat_init = codes_init.to(torch.float32) * eff_exp_init
            code_fingerprint_before = hashlib.sha256(codes_init.numpy().tobytes()).hexdigest()
            code_fingerprint_after = hashlib.sha256(codes_final.numpy().tobytes()).hexdigest()
            code_change_ratio = float((codes_init != codes_final).float().mean().item())
            zero_before = float((codes_init == 0).float().mean().item())
            zero_after = float((codes_final == 0).float().mean().item())
        thr_before = torch.full_like(thr_final, 0.5)
        thr_fingerprint_before = _hash_tensor(thr_before)
        thr_fingerprint_after = _hash_tensor(thr_final.detach().cpu())
        thr_mean = float(thr_final.mean().item())
        thr_min = float(thr_final.min().item())
        thr_max = float(thr_final.max().item())
        thr_std = float(thr_final.float().std(correction=0).item()) if thr_final.numel() > 1 else 0.0
    else:
        eff_final = get_effective_scales(raw_param, zero_mask_t)
        if scale_granularity == "per_tensor":
            w_hat_final = codes_fixed.to(torch.float32) * eff_final[0]
        else:
            res_tmp = GroupwiseResult(
                codes=codes_fixed,
                scales=eff_final,
                shape=tuple(dummy_weight.shape),
                orig_dtype=str(dummy_weight.dtype),
                group_size=group_size,
                grouping_scheme="last-dim-rowwise-v1",
            )
            w_hat_final = dequantize_groupwise(res_tmp)
        if scale_granularity == "per_tensor":
            w_hat_init = codes_fixed.to(torch.float32) * orig_scales[0]
        else:
            res_init = GroupwiseResult(
                codes=codes_fixed,
                scales=orig_scales,
                shape=tuple(dummy_weight.shape),
                orig_dtype=str(dummy_weight.dtype),
                group_size=group_size,
                grouping_scheme="last-dim-rowwise-v1",
            )
            w_hat_init = dequantize_groupwise(res_init)
        code_fingerprint_before = hashlib.sha256(codes_fixed.numpy().tobytes()).hexdigest()
        code_fingerprint_after = code_fingerprint_before
        code_change_ratio = 0.0
        zero_before = float((codes_fixed == 0).float().mean().item())
        zero_after = zero_before
        thr_fingerprint_before = ""
        thr_fingerprint_after = ""
        thr_mean = thr_min = thr_max = thr_std = 0.0
        thr_final = None  # type: ignore[assignment]

    held_losses: list[float] = []
    for inp, tout in zip(held_inputs, held_teacher_outs, strict=False):  # noqa: B905
        y_hat = F.linear(inp, w_hat_final)  # type: ignore[union-attr]
        held_losses.append(float(mse_loss(tout, y_hat).item()))
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
        + (
            int(raw_threshold.numel() - int(thr_zero_mask.sum().item()))
            if threshold_enabled and raw_threshold is not None and thr_zero_mask is not None
            else 0
        ),
        "total_param_count": int(raw_param.numel())
        + (int(raw_threshold.numel()) if threshold_enabled and raw_threshold is not None else 0),
        "used_dummy_simulation": True,
        "teacher_snapshot": None,
        "threshold_enabled": threshold_enabled,
        "threshold_fingerprint_before": thr_fingerprint_before,
        "threshold_fingerprint_after": thr_fingerprint_after,
        "threshold_ratio_mean": thr_mean,
        "threshold_ratio_min": thr_min,
        "threshold_ratio_max": thr_max,
        "threshold_ratio_std": thr_std,
        "code_fingerprint_before": code_fingerprint_before,
        "code_fingerprint_after": code_fingerprint_after,
        "code_change_ratio": code_change_ratio,
        "zero_ratio_before": zero_before,
        "zero_ratio_after": zero_after,
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
