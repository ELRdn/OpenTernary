"""Calibration runner — Phase 4.2 layer-local recon-scale-threshold."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import typing
from collections.abc import Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

import contextlib

from openternary.calibration.optimizer import (
    build_scale_params,
    build_threshold_params,
    get_effective_scales,
    get_effective_threshold_ratio,
)
from openternary.utils.device import (
    BackendInfo,
    VramBudget,
    compute_vram_budget,
    detect_backend,
    format_bytes,
    is_oom_error,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for calibration. Install with: uv sync --extra ml")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _state_manifest_entry(module_name: str, relative_path: str, path: pathlib.Path) -> dict[str, typing.Any]:
    return {
        "module": module_name,
        "file": relative_path,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_state_manifest_entry(entry: dict[str, typing.Any], path: pathlib.Path) -> None:
    if not path.exists():
        raise ValueError(f"checkpoint state integrity failure: missing {path}")
    expected_size = entry.get("size")
    expected_hash = entry.get("sha256")
    if not isinstance(expected_size, int) or not isinstance(expected_hash, str):
        raise ValueError("checkpoint state integrity metadata is missing")
    if path.stat().st_size != expected_size:
        raise ValueError(f"checkpoint state size mismatch: {path}")
    if _sha256_file(path) != expected_hash:
        raise ValueError(f"checkpoint state hash mismatch: {path}")


def _clone_transaction_value(value: typing.Any) -> typing.Any:
    """Clone optimizer transaction state to CPU without retaining graphs."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_transaction_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_transaction_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_transaction_value(item) for item in value)
    return value


def _capture_optimizer_transaction(
    params: Sequence[torch.Tensor], optimizer: torch.optim.Optimizer
) -> dict[str, typing.Any]:
    """Capture the state needed to roll one optimizer update back exactly."""
    _require_torch()
    transaction: dict[str, typing.Any] = {
        "params": [param.detach().cpu().clone() for param in params],
        "optimizer": _clone_transaction_value(optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            transaction["cuda_rng"] = torch.cuda.get_rng_state_all()
    return transaction


def _restore_optimizer_transaction(
    transaction: dict[str, typing.Any], params: Sequence[torch.Tensor], optimizer: torch.optim.Optimizer
) -> None:
    """Restore a failed optimizer update, including moments and RNG state."""
    _require_torch()
    for param, saved in zip(params, transaction["params"], strict=True):
        param.data.copy_(saved.to(device=param.device, dtype=param.dtype))
        param.grad = None
    optimizer.load_state_dict(transaction["optimizer"])
    torch.set_rng_state(transaction["cpu_rng"])
    if "cuda_rng" in transaction and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.set_rng_state_all(transaction["cuda_rng"])


def _find_latest_checkpoint(output_dir: pathlib.Path) -> pathlib.Path | None:
    ckpt_dir = pathlib.Path(output_dir) / "artifacts" / "checkpoint"
    if not ckpt_dir.exists():
        return None
    # v2 supports both step_*.pt and module_*.pt, pick latest by mtime
    cands = sorted(ckpt_dir.glob("step_*.pt")) + sorted(ckpt_dir.glob("module_*.pt"))
    if not cands:
        return None
    # sort by name which encodes order, fallback to mtime
    cands = sorted(cands)
    return cands[-1]


def _snapshot_content_fingerprint(snapshot_path: pathlib.Path | str | None) -> str:
    """Bounded byte hash: equal-size weight changes must invalidate caches."""
    if snapshot_path is None:
        return ""
    p = pathlib.Path(snapshot_path)
    files = sorted(set(p.glob("*.json")) | set(p.glob("*.safetensors")))
    if not any(f.suffix == ".safetensors" for f in files):
        raise ValueError("snapshot fingerprint requires weight files")
    h = hashlib.sha256()
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        h.update(file.name.encode() + b"\0" + digest.digest())
    return h.hexdigest()


def _dataset_fingerprint(effective_dataset: str, requested_dataset: str) -> str:
    """Dataset fingerprint (deterministic hash of dataset ids)."""
    return hashlib.sha256(f"{requested_dataset}:{effective_dataset}".encode()).hexdigest()


def _tokenizer_fingerprint(snapshot_path: pathlib.Path | str | None) -> str:
    if snapshot_path is None:
        return ""
    try:
        p = pathlib.Path(snapshot_path) / "tokenizer.json"
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        pass
    return ""


def _run_materialize_only(
    app_config: typing.Any,
    teacher_snapshot: pathlib.Path | str | None,
    output_dir: pathlib.Path | str,
    checkpoint_path: pathlib.Path | str | None = None,
) -> dict[str, typing.Any]:
    """Resumeなしで最後のcheckpointからmaterializeのみ実行."""
    _require_torch()
    import pathlib as _pl

    out = _pl.Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(f"output_dir not found for materialize-only: {out}")
    if teacher_snapshot is None:
        raise ValueError("teacher_snapshot is required for materialize-only")
    p = _pl.Path(teacher_snapshot)
    if not p.exists() or not (p / "config.json").exists():
        raise FileNotFoundError(f"teacher_snapshot not found: {p}")

    # Resolve checkpoint
    ckpt_path = _pl.Path(checkpoint_path) if checkpoint_path is not None else _find_latest_checkpoint(out)
    if ckpt_path is None or not _pl.Path(ckpt_path).exists():
        raise FileNotFoundError(f"no checkpoint found for materialize-only in {out}/artifacts/checkpoint")
    print(f"[materialize-only] using checkpoint {ckpt_path}", flush=True)

    # Normal materialization is v3-only. Older artifacts require the explicit
    # audited recovery/finalize workflow and are never silently upgraded here.
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("schema_version") != 3:
        raise ValueError("checkpoint schema is not v3; audited conversion is required")
    contract = ckpt.get("resume_contract")
    if not isinstance(contract, dict):
        raise ValueError("materialize-only checkpoint resume contract is missing")
    if contract.get("source_fingerprint") != _snapshot_content_fingerprint(teacher_snapshot):
        raise ValueError("materialize-only source fingerprint mismatch")
    targets = contract["target_module_names"]
    manifest_targets = [entry["module"] for entry in ckpt.get("completed_manifest", [])]
    cursor = ckpt.get("module_cursor", {})
    if manifest_targets != targets or cursor.get("module_idx") != len(targets) or cursor.get("step") != 0:
        raise ValueError("materialize-only requires a complete target checkpoint")
    if (
        contract["quantization"] != app_config.quantization.model_dump()
        or contract["calibration"] != app_config.calibration.model_dump()
    ):
        raise ValueError("materialize-only contract mismatch")
    # Support v2 module-major checkpoint: completed_module_params
    if isinstance(ckpt, dict) and ckpt.get("schema_version") == 3 and "completed_module_params" in ckpt:
        # Normalize to per_module shape expected below
        # completed_module_params maps mname -> {raw_param, raw_threshold?}
        # Also may have current module cursor param
        pass  # handling below in per_module_keys extraction
    # ckpt contains at least per-module raw_param dict or flat
    # Our synthetic checkpoint format: {"raw_param": tensor, ...} or {"per_module": {name: raw}}
    # Real checkpoint format: saved per_module raw_param dict
    # Handle both

    # Try to get per_module from checkpoint; if not, reconstruct via teacher
    # Checkpoint from runner saves {"step": int, "per_module": {mname: {"raw_param": tensor}}, "scale_fingerprint": ...}
    q_cfg = app_config.quantization
    group_size = int(getattr(q_cfg, "group_size", 128))
    scale_granularity = str(getattr(q_cfg, "scale_granularity", "per_tensor"))

    # Rebuild per_module codes from teacher weights (needed for dequantization)
    from safetensors import safe_open

    from openternary.quant.grouping import quantize_groupwise
    from openternary.quant.ternary import quantize_absmean

    # Need target module names — infer from checkpoint per_module keys or from teacher model structure
    # Try to get keys from ckpt
    per_module_keys: list[str] = []
    if isinstance(ckpt, dict) and ckpt.get("schema_version") == 3 and "completed_manifest" in ckpt:
        # P0 manifest based checkpoint
        per_module_keys = [e.get("module") for e in ckpt.get("completed_manifest", []) if e.get("module")]
        # include current cursor module if present and not yet in manifest
        mc = ckpt.get("module_cursor")
        if isinstance(mc, dict) and mc.get("module_name"):
            mn = mc.get("module_name")
            if mn and mn not in per_module_keys:
                per_module_keys.append(mn)
        # also include raw_params keys (mid-module current)
        if "raw_params" in ckpt and isinstance(ckpt["raw_params"], dict):
            for k in ckpt["raw_params"]:
                if k not in per_module_keys:
                    per_module_keys.append(k)
    elif isinstance(ckpt, dict) and ckpt.get("schema_version") == 3 and "completed_module_params" in ckpt:
        per_module_keys = list(ckpt["completed_module_params"].keys())
        # include current cursor module if present
        mc = ckpt.get("module_cursor")
        if isinstance(mc, dict) and mc.get("module_name"):
            mn = mc.get("module_name")
            if mn and mn not in per_module_keys:
                per_module_keys.append(mn)
        # also include raw_params keys if any for backward compat
        if "raw_params" in ckpt and isinstance(ckpt["raw_params"], dict):
            for k in ckpt["raw_params"]:
                if k not in per_module_keys:
                    per_module_keys.append(k)
        # placeholder for current_raw_param handling
    elif isinstance(ckpt, dict) and "per_module" in ckpt:
        per_module_keys = list(ckpt["per_module"].keys())
    elif isinstance(ckpt, dict) and "raw_params" in ckpt and isinstance(ckpt["raw_params"], dict):
        per_module_keys = list(ckpt["raw_params"].keys())
    elif isinstance(ckpt, dict) and "raw_param" in ckpt:
        # old synthetic format — not real, fallback to empty
        per_module_keys = []
    else:
        # try ckpt is dict of {name: tensor}
        per_module_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []

    # If keys missing, rebuild by loading teacher model and discovering target modules
    if not per_module_keys:
        # Instead, discover quantizable tensors via header classification
        from openternary.utils.hf_cache import resolve_snapshot

        src_root = resolve_snapshot(str(teacher_snapshot), app_config.model.revision)
        from openternary.quant.fake_quant import _classify_tensor as _cls

        per_module_keys = []
        st_files = list(src_root.glob("*.safetensors"))
        for stf in st_files:
            with safe_open(str(stf), framework="pt", device="cpu") as f:
                for name in f.keys():  # noqa: SIM118
                    # name is tensor name, convert to module name by stripping .weight
                    if not name.endswith(".weight"):
                        continue
                    mname = name[: -len(".weight")]
                    # Check if quantizable via config
                    # Use fake_quant classifier
                    shape = list(f.get_slice(name).get_shape())
                    # dtype unknown, assume BF16
                    _, quantizable, _ = _cls(name, shape, "BF16", app_config)
                    if quantizable:
                        per_module_keys.append(mname)
        per_module_keys = sorted(set(per_module_keys))

    print(f"[materialize-only] target modules from checkpoint/header: {len(per_module_keys)}", flush=True)

    per_module: dict[str, typing.Any] = {}

    # Helper to load weight tensor by module name
    def _load_weight(mname: str) -> torch.Tensor:  # type: ignore[no-any-return]
        wname = mname + ".weight"
        src_root = pathlib.Path(teacher_snapshot)
        st_files = list(src_root.glob("*.safetensors"))
        for stf in st_files:
            with safe_open(str(stf), framework="pt", device="cpu") as f:
                if wname in f.keys():  # noqa: SIM118
                    return f.get_tensor(wname)  # type: ignore[no-any-return]
        raise FileNotFoundError(f"weight {wname} not found in {teacher_snapshot}")

    from openternary.calibration.optimizer import (
        build_scale_params,
        build_threshold_params,
        get_effective_scales,
        get_effective_threshold_ratio,
    )

    # threshold config
    threshold_enabled_ckpt = bool(
        ckpt.get("threshold_enabled", False) or "raw_thresholds" in ckpt or "raw_threshold" in ckpt
    )
    # also check app_config
    threshold_enabled_cfg = bool(getattr(app_config.calibration, "threshold_enabled", False))
    threshold_enabled = threshold_enabled_ckpt or threshold_enabled_cfg
    soft_to_hard_enabled = str(getattr(app_config.calibration, "method", "recon-scale")) == "recon-soft-to-hard"
    if soft_to_hard_enabled:
        assignment = ckpt.get("assignment")
        if not isinstance(assignment, dict) or assignment.get("mode") != "soft-to-hard":
            raise ValueError("materialize-only soft-to-hard assignment contract is missing")
        if assignment.get("final_state") != "hard" or assignment.get("next_temperature") is not None:
            raise ValueError("materialize-only requires a completed hard soft-to-hard checkpoint")
    threshold_eps = float(getattr(app_config.calibration, "threshold_eps", 0.01))
    # legacy: ckpt may have threshold_eps
    if "threshold_eps" in ckpt:
        with contextlib.suppress(Exception):
            threshold_eps = float(ckpt["threshold_eps"])

    for mname in per_module_keys:
        w = _load_weight(mname)
        if scale_granularity == "per_tensor":
            tt = quantize_absmean(w.to(torch.float32))
            orig_scales = torch.tensor([tt.scale], dtype=torch.float32)
            codes = tt.codes
        else:
            res = quantize_groupwise(w.to(torch.float32), group_size)
            orig_scales = res.scales.clone()
            codes = res.codes
        zero_mask = orig_scales == 0
        raw_param, zero_mask_t = build_scale_params(orig_scales, zero_mask)
        # Override raw_param from checkpoint if available (v2 manifest or legacy)
        ckpt_raw = None
        if isinstance(ckpt, dict) and ckpt.get("schema_version") == 3 and "completed_manifest" in ckpt:
            # P0: load from calibration_state file (safetensors preferred)
            for entry in ckpt.get("completed_manifest", []):
                if entry.get("module") == mname:
                    rel = entry.get("file", "")
                    fpath = out / rel if rel else None
                    if fpath and fpath.exists():
                        try:
                            _validate_state_manifest_entry(entry, fpath)
                            if str(fpath).endswith(".safetensors"):
                                from safetensors.torch import load_file as _load_sf  # type: ignore[import]

                                payload = _load_sf(str(fpath), device="cpu")
                            else:
                                payload = torch.load(str(fpath), map_location="cpu")
                            ckpt_raw = payload.get("raw_param")
                        except Exception as e:
                            print(f"[materialize-only] warning: failed to load manifest {fpath}: {e}", flush=True)
                    break
            # also check current cursor if this is active module
            mc2 = ckpt.get("module_cursor")
            if mc2 and mc2.get("module_name") == mname and "current_raw_param" in ckpt:
                ckpt_raw = ckpt.get("current_raw_param")
        elif isinstance(ckpt, dict) and ckpt.get("schema_version") == 3 and "completed_module_params" in ckpt:
            if mname in ckpt["completed_module_params"]:
                entry = ckpt["completed_module_params"][mname]
                if isinstance(entry, dict):
                    ckpt_raw = entry.get("raw_param")
                    if ckpt_raw is None:
                        ckpt_raw = entry.get("raw")
                else:
                    ckpt_raw = entry
            # also check current cursor if this is active module
            mc2 = ckpt.get("module_cursor")
            if mc2 and mc2.get("module_name") == mname and "current_raw_param" in ckpt:
                ckpt_raw = ckpt.get("current_raw_param")
            if ckpt_raw is None and "raw_params" in ckpt and mname in ckpt["raw_params"]:
                ckpt_raw = ckpt["raw_params"][mname]
        elif isinstance(ckpt, dict) and "per_module" in ckpt and mname in ckpt["per_module"]:
            ckpt_raw = ckpt["per_module"][mname].get("raw_param")
            if ckpt_raw is None:
                ckpt_raw = ckpt["per_module"][mname].get("raw")
        elif isinstance(ckpt, dict) and "raw_params" in ckpt and mname in ckpt["raw_params"]:
            ckpt_raw = ckpt["raw_params"][mname]
        elif isinstance(ckpt, dict) and mname in ckpt:
            ckpt_raw = ckpt.get(mname)
        if contract is not None and ckpt_raw is None:
            raise ValueError(f"checkpoint parameter missing for {mname}")
        if ckpt_raw is not None:
            try:
                raw_param.data = ckpt_raw.to(raw_param.device)
            except Exception as e:
                print(f"[materialize-only] warning: failed to load raw for {mname}: {e}", flush=True)
        # threshold handling
        thr_ratio_final = None
        if threshold_enabled:
            # build threshold param
            n_thr = int(orig_scales.numel())
            raw_thr, thr_zero_mask = build_threshold_params(
                n_thr, init_ratio=0.5, eps=threshold_eps, device=orig_scales.device, zero_mask=zero_mask
            )
            # override from checkpoint (v2 manifest or legacy) — safetensors aware
            ckpt_thr = None
            if isinstance(ckpt, dict) and ckpt.get("schema_version") == 3 and "completed_manifest" in ckpt:
                for entry in ckpt.get("completed_manifest", []):
                    if entry.get("module") == mname:
                        rel = entry.get("file", "")
                        fpath = out / rel if rel else None
                        if fpath and fpath.exists():
                            try:
                                _validate_state_manifest_entry(entry, fpath)
                                if str(fpath).endswith(".safetensors"):
                                    from safetensors.torch import load_file as _load_sf2  # type: ignore[import]

                                    payload = _load_sf2(str(fpath), device="cpu")
                                else:
                                    payload = torch.load(str(fpath), map_location="cpu")
                                ckpt_thr = payload.get("raw_threshold")
                            except Exception as e:
                                print(f"[materialize-only] warning: manifest thr load failed {fpath}: {e}", flush=True)
                        break
                mc_thr = ckpt.get("module_cursor")
                if mc_thr and mc_thr.get("module_name") == mname and "current_raw_threshold" in ckpt:
                    ckpt_thr = ckpt.get("current_raw_threshold")
            elif (
                isinstance(ckpt, dict)
                and ckpt.get("schema_version") == 3
                and "completed_module_params" in ckpt
                and mname in ckpt["completed_module_params"]
            ):
                entry2 = ckpt["completed_module_params"][mname]
                if isinstance(entry2, dict):
                    ckpt_thr = entry2.get("raw_threshold")
                    if ckpt_thr is None:
                        ckpt_thr = entry2.get("raw_thr")
                mc_thr = ckpt.get("module_cursor")
                if mc_thr and mc_thr.get("module_name") == mname and "current_raw_threshold" in ckpt:
                    ckpt_thr = ckpt.get("current_raw_threshold")
            if (
                ckpt_thr is None
                and isinstance(ckpt, dict)
                and "raw_thresholds" in ckpt
                and mname in ckpt["raw_thresholds"]
            ):
                ckpt_thr = ckpt["raw_thresholds"][mname]
            elif (
                ckpt_thr is None
                and isinstance(ckpt, dict)
                and "raw_threshold" in ckpt
                and isinstance(ckpt["raw_threshold"], dict)
                and mname in ckpt["raw_threshold"]
            ):
                ckpt_thr = ckpt["raw_threshold"][mname]
            elif (
                ckpt_thr is None
                and isinstance(ckpt, dict)
                and "raw_threshold" in ckpt
                and not isinstance(ckpt["raw_threshold"], dict)
            ):
                # synthetic single param
                ckpt_thr = ckpt.get("raw_threshold")
            if contract is not None and ckpt_thr is None:
                raise ValueError(f"checkpoint threshold parameter missing for {mname}")
            if ckpt_thr is not None:
                try:
                    raw_thr.data = ckpt_thr.to(raw_thr.device)
                except Exception as e:
                    print(f"[materialize-only] warning: failed to load thr for {mname}: {e}", flush=True)
            thr_ratio_final = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)
            # regenerate hard codes with threshold
            from openternary.quant.threshold import hard_threshold_codes as _htc_m

            if scale_granularity == "per_tensor":
                codes = _htc_m(w.to(torch.float32), float(orig_scales[0].item()), float(thr_ratio_final[0].item()))
            else:
                codes = _htc_m(w.to(torch.float32), orig_scales, thr_ratio_final, group_size=group_size)
        elif soft_to_hard_enabled:
            from openternary.quant.soft_ternary import harden_soft_codes

            if scale_granularity == "per_tensor":
                codes = harden_soft_codes(w.to(torch.float32), reference_scale=float(orig_scales[0].item()))
            else:
                codes = harden_soft_codes(
                    w.to(torch.float32),
                    reference_scale=orig_scales,
                    group_size=group_size,
                )
        per_module[mname] = {
            "codes": codes,
            "orig_scales": orig_scales,
            "raw_param": raw_param,
            "zero_mask": zero_mask,
            "zero_mask_t": zero_mask_t,
            "weight_shape": tuple(w.shape),
            "weight_dtype": str(w.dtype),
            "reference_scales": orig_scales.clone(),
            "threshold_ratio": thr_ratio_final,
        }

    # Build calibrated_state
    calibrated_state: dict[str, typing.Any] = {}
    for mname, state in per_module.items():
        eff = get_effective_scales(state["raw_param"], state["zero_mask_t"])
        calibrated_state[mname + ".weight"] = {
            "codes": state["codes"],
            "scales": eff,
            "shape": state["weight_shape"],
            "orig_dtype": state["weight_dtype"],
            "group_size": group_size,
            "grouping_scheme": "last-dim-rowwise-v1",
        }

    from openternary.quant.fake_quant import materialize_calibrated_snapshot

    calibrated_snapshot = out / "artifacts" / "calibrated_snapshot"
    if calibrated_snapshot.exists():
        import shutil as _sh

        _sh.rmtree(calibrated_snapshot, ignore_errors=True)
    print(f"[materialize-only] materializing to {calibrated_snapshot} with {len(calibrated_state)} tensors", flush=True)
    _report = materialize_calibrated_snapshot(
        src_snapshot=pathlib.Path(teacher_snapshot),
        dst_snapshot=calibrated_snapshot,
        app_config=app_config,
        calibrated_state=calibrated_state,
    )
    if not (calibrated_snapshot / "model.safetensors.index.json").exists():
        raise RuntimeError("materialize-only failed: missing index.json")
    print(
        f"[materialize-only] done: shard_count={_report.shard_count} content_fp={_report.content_fingerprint}",
        flush=True,
    )
    # Update calibration.json with materialize info if exists
    calib_path = out / "calibration.json"
    if calib_path.exists():
        try:
            import json as _json2

            data = _json2.loads(calib_path.read_text(encoding="utf-8"))
            data["calibrated_snapshot"] = str(calibrated_snapshot)
            data["calibrated_content_fingerprint"] = _report.content_fingerprint
            data["calibrated_shard_count"] = int(_report.shard_count)
            data["materialize_only"] = True
            data["materialize_checkpoint"] = str(ckpt_path)
            calib_path.write_text(_json2.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            raise ValueError("materialize-only report could not be updated") from e

    return {
        "run_dir": str(out),
        "calibrated_snapshot": str(calibrated_snapshot),
        "checkpoint": str(ckpt_path),
        "report": _report,
    }


def run_calibration(
    app_config: typing.Any,
    teacher_snapshot: pathlib.Path | str | None,
    output_dir: pathlib.Path | str,
    resume: bool = False,
    resume_from: pathlib.Path | str | None = None,
    materialize_only: bool = False,
    init_from: pathlib.Path | str | None = None,
) -> dict[str, typing.Any]:
    """Run real calibration only; missing models cannot become successful simulations."""
    if teacher_snapshot is None or not (pathlib.Path(teacher_snapshot) / "config.json").is_file():
        raise FileNotFoundError(f"teacher snapshot missing: {teacher_snapshot}")
    if not app_config.calibration.enabled:
        raise ValueError("calibration.enabled must be true")
    if not app_config.runtime.low_vram:
        raise ValueError("calibration currently requires runtime.low_vram=true")
    from openternary.experiment.run import run_lock

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        return _run_calibration_impl(
            app_config, teacher_snapshot, out, resume, resume_from, materialize_only, init_from
        )


def _run_calibration_impl(
    app_config: typing.Any,
    teacher_snapshot: pathlib.Path | str | None,
    output_dir: pathlib.Path | str,
    resume: bool = False,
    resume_from: pathlib.Path | str | None = None,
    materialize_only: bool = False,
    init_from: pathlib.Path | str | None = None,
) -> dict[str, typing.Any]:
    """Layer-local calibration runner — Phase 4.2 with warm start.

    - Captures teacher activations to disk (or reuses if fingerprint matches)
    - Unloads teacher
    - Optimizes learnable scales + thresholds per layer via F.linear (STE)
    - Checkpoints (v2) and final materialization (bounded streaming)

    Returns dict with calibration report.
    """
    _require_torch()
    if materialize_only:
        return _run_materialize_only(app_config, teacher_snapshot, output_dir, resume_from)
    if teacher_snapshot is not None and not (pathlib.Path(teacher_snapshot) / "config.json").is_file():
        raise FileNotFoundError(f"teacher snapshot missing: {teacher_snapshot}")
    # Handle --init-from via config or CLI
    if init_from is None:
        # also check config field
        cfg_init = getattr(app_config.calibration, "init_from", None)
        if cfg_init is not None:
            init_from = cfg_init

    import pathlib as _pl

    out = _pl.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (resume or resume_from is not None) and (
        (out / "calibration.json").exists() or (out / "artifacts/checkpoint").exists()
    ):
        raise ValueError("existing calibration run; use resume or a new output directory")
    # Save config
    import yaml

    cfg_path = out / "config.yaml"
    # app_config is Pydantic
    try:
        cfg_dict = app_config.model_dump() if hasattr(app_config, "model_dump") else dict(app_config)
    except Exception:
        cfg_dict = {}
    if not (resume or resume_from is not None):
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    calib_cfg = app_config.calibration
    # Validate window
    if calib_cfg.window != "per-layer":
        raise ValueError(f"window={calib_cfg.window} is not supported in Phase 4.1 (only per-layer)")

    # Runtime backend detection (Phase 4.2-G)
    runtime_cfg = getattr(app_config, "runtime", None)
    # defaults for RX7600 8GB soft policy
    _low_vram = True
    _max_fraction = 0.80
    _reserve_mb = 1024
    _min_free_mb = 768
    _micro_auto = True
    _micro_min = 1
    if runtime_cfg is not None:
        try:
            _low_vram = bool(getattr(runtime_cfg, "low_vram", True))
            vram_cfg = getattr(runtime_cfg, "vram", None)
            if vram_cfg is not None:
                _max_fraction = float(getattr(vram_cfg, "max_fraction", 0.80))
                _reserve_mb = int(getattr(vram_cfg, "reserve_mb", 1024))
                _min_free_mb = int(getattr(vram_cfg, "min_free_mb", 768))
            micro_cfg = getattr(runtime_cfg, "microbatch", None)
            if micro_cfg is not None:
                _micro_auto = bool(getattr(micro_cfg, "auto", True))
                _micro_min = int(getattr(micro_cfg, "min_size", 1))
        except Exception:
            pass
    backend_info: BackendInfo = detect_backend()
    # Respect device preference: cpu|cuda|auto
    _pref = str(getattr(app_config, "device", "auto")).lower()
    if _pref == "cpu":
        # force cpu even if gpu available
        backend_info = BackendInfo(
            backend="cpu",
            device_name="cpu (forced)",
            total_bytes=0,
            free_bytes=0,
            is_hip=False,
            is_cuda=False,
            bf16_supported=False,
        )
    elif _pref == "cuda" and backend_info.backend == "cpu":
        raise ValueError("device=cuda requested but no GPU available")
    # Observability: always log backend
    print(f"[device] backend={backend_info.backend}", flush=True)
    print(f"[device] {backend_info.device_name}", flush=True)
    if backend_info.backend != "cpu":
        try:
            free_b, total_b = (backend_info.free_bytes, backend_info.total_bytes)
            # recompute via mem_get_info for fresh value
            if torch is not None and torch.cuda.is_available():
                try:
                    fb, tb = torch.cuda.mem_get_info(0)  # type: ignore[union-attr]
                    free_b, total_b = int(fb), int(tb)
                except Exception:
                    pass
            budget = compute_vram_budget(free_b, total_b, _max_fraction, _reserve_mb, _min_free_mb)
            print(
                f"[vram] total={format_bytes(total_b)} free={format_bytes(free_b)} budget={format_bytes(budget.budget_bytes)} (max_fraction={_max_fraction} reserve={_reserve_mb}MB min_free={_min_free_mb}MB)",
                flush=True,
            )
        except Exception as e:
            print(f"[vram] warning: failed to compute budget: {e}", flush=True)
    run_calibration._peak_vram = 0  # type: ignore[attr-defined]

    # Prepare activation cache dir
    cache_dir = out / "artifacts" / "activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _is_dummy = False  # noqa: F841
    if teacher_snapshot is None:
        _is_dummy = True  # noqa: F841
    else:
        p = _pl.Path(teacher_snapshot)
        if not p.exists() or not (p / "config.json").exists():
            _is_dummy = True  # noqa: F841

    # Dataset handling via abstraction
    from openternary.calibration.dataset import (
        SMOKE_TEXTS,
        get_calibration_texts,
        hash_batches,
        split_train_held,
        tokenize_texts,
    )

    num_samples = int(calib_cfg.num_samples)
    seq_len = int(calib_cfg.seq_len)
    seed = int(calib_cfg.seed) if calib_cfg.seed is not None else int(app_config.seed)
    held_out_ratio = float(calib_cfg.held_out_ratio)

    # Resolve texts with fallback handling
    requested_dataset = str(calib_cfg.dataset)
    try:
        texts, effective_dataset, dataset_fallback = get_calibration_texts(
            requested_dataset,
            num_samples,
            seed,
            bool(calib_cfg.allow_dataset_fallback),
            revision=str(calib_cfg.dataset_revision),
        )
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        # loud error when fallback not allowed
        raise ValueError(f"dataset {requested_dataset} failed: {e}") from e

    train_texts, held_texts = split_train_held(texts, held_out_ratio)

    # Try to get tokenizer for hashing (real tokenizer gives input_ids hash)
    tokenizer: typing.Any | None = None
    if teacher_snapshot is not None:
        try:
            p_snap = _pl.Path(teacher_snapshot)
            if p_snap.exists() and (p_snap / "tokenizer.json").exists():
                from transformers import AutoTokenizer  # type: ignore[import]

                tokenizer = AutoTokenizer.from_pretrained(str(p_snap), use_fast=True)  # type: ignore[union-attr]
        except Exception:
            tokenizer = None

    # Build hashes: prefer input_ids hash if tokenizer available
    if tokenizer is not None:
        try:
            train_batches_tmp = tokenize_texts(train_texts, tokenizer, seq_len)
            held_batches_tmp = tokenize_texts(held_texts, tokenizer, seq_len)
            train_hashes = hash_batches(train_batches_tmp)
            held_hashes = hash_batches(held_batches_tmp)
            # smoke hashes also via tokenizer if possible for consistency
            smoke_batches_tmp = tokenize_texts(SMOKE_TEXTS, tokenizer, seq_len)
            smoke_hashes = hash_batches(smoke_batches_tmp)
        except Exception:
            # fallback to text hash
            train_hashes = [_hash_text(t) for t in train_texts]
            held_hashes = [_hash_text(t) for t in held_texts]
            smoke_hashes = [_hash_text(t) for t in SMOKE_TEXTS]
    else:
        train_hashes = [_hash_text(t) for t in train_texts]
        held_hashes = [_hash_text(t) for t in held_texts]
        smoke_hashes = [_hash_text(t) for t in SMOKE_TEXTS]

    contamination_train_smoke = not set(train_hashes).isdisjoint(set(smoke_hashes))
    contamination_held_smoke = not set(held_hashes).isdisjoint(set(smoke_hashes))
    contamination_train_held = not set(train_hashes).isdisjoint(set(held_hashes))

    # Branch: real-model vs synthetic helper (TEST ONLY)
    # Real run requires teacher_snapshot; dummy simulation is forbidden in real path (Gate failure)
    is_real_snapshot = (
        teacher_snapshot is not None
        and _pl.Path(teacher_snapshot).exists()
        and (_pl.Path(teacher_snapshot) / "config.json").exists()
    )
    # Helper is allowed for: no snapshot, or synthetic tiny fixture (fast CI)
    # Real path is required for wiki-tiny with snapshot (and for synthetic with large N if needed)
    is_synthetic_tiny = teacher_snapshot is None and effective_dataset == "synthetic"
    if not is_real_snapshot:
        if not is_synthetic_tiny:
            raise ValueError("real calibration requires a teacher snapshot")
        # Synthetic tiny fixture or no snapshot — delegate to TEST ONLY helper (keeps real path clean)
        # Forbid wiki-tiny without snapshot unless fallback (already handled)
        if requested_dataset == "wiki-tiny" and is_real_snapshot and not is_synthetic_tiny:
            # wiki-tiny with snapshot must go real, not helper
            pass
        else:
            from openternary.calibration._synthetic_simulation import run_synthetic_tiny

            return run_synthetic_tiny(
                app_config,
                out,
                train_texts,
                held_texts,
                train_hashes,
                held_hashes,
                smoke_hashes,
                (contamination_train_smoke, contamination_held_smoke, contamination_train_held),
                requested_dataset,
                effective_dataset,
                dataset_fallback,
                resume=resume,
                resume_from=resume_from,
            )

    # Real-model path — wiki-tiny/synthetic → tokenizer → Teacher forward → DiskActivationCache → layer-local
    # Gate: dummy simulation must not be used here
    # Any reference to dummy_weight / make_activations in this branch is a Gate failure
    if is_real_snapshot:
        # Ensure we do not accidentally use dummy path
        pass

    # Real path implementation
    # 1. Tokenizer (already resolved for hashes, but need batches for capture)
    import os

    test_overrides = [
        name for name in ("OT_SKIP_MATERIALIZE", "OT_FORCE_ROCM", "OT_FORCE_LOW_BUDGET") if os.environ.get(name) == "1"
    ]
    if os.environ.get("OT_LIMIT_MODULES"):
        test_overrides.append("OT_LIMIT_MODULES")
    if test_overrides:
        raise ValueError(f"test-only overrides forbidden in real calibration: {test_overrides}")
    if contamination_train_smoke or contamination_held_smoke or contamination_train_held:
        raise ValueError("calibration split overlap/contamination detected")
    if tokenizer is None:
        raise ValueError("real-model calibration requires tokenizer from teacher_snapshot")

    train_batches = tokenize_texts(train_texts, tokenizer, seq_len)
    held_batches = tokenize_texts(held_texts, tokenizer, seq_len)

    # 2. Load Teacher model and enumerate targets
    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import]

        # Try Gemma4 multimodal first, fallback to causal
        try:
            from transformers import AutoModelForImageTextToText as _AutoGemma  # type: ignore[import]

            model = _AutoGemma.from_pretrained(
                str(teacher_snapshot), torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=False
            )  # type: ignore[union-attr]
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                str(teacher_snapshot), torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=False
            )  # type: ignore[union-attr]
    except Exception as e:
        raise ImportError(f"failed to load Teacher model from {teacher_snapshot}: {e}") from e

    # Enumerate quantizable linears via adapter
    try:
        from openternary.adapters.gemma4 import Gemma4Adapter  # type: ignore[import]

        _ = Gemma4Adapter()  # noqa: F841 — adapter instantiation for side-effect / future classification
        # Canonical target set: 205 modules (attention q/k/v/o + mlp gate/up/down) — per_layer/ple/vision/audio/norm are high-precision
        # Must match inspect's quantizable count (ARCHITECTURE.md / adapters/base.py) for fair scale-only vs threshold comparison
        import re as _re

        _Q_ATTN = _re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
        _Q_MLP = _re.compile(r"\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$")
        target_module_names: list[str] = []
        for name, module in model.named_modules():
            import torch.nn as _nn  # type: ignore[import]

            if isinstance(module, _nn.Linear):
                # Exclude non-text / high-precision per spec: lm_head, embeddings, vision/audio towers, PLE/per_layer, norm
                if "lm_head" in name or "embed" in name:
                    continue
                if "vision_tower" in name or "audio_tower" in name or "vision_encoder" in name:
                    continue
                if "per_layer" in name:
                    continue
                if "norm" in name.lower():
                    continue
                # Only attention/mlp quantizable per adapters/base.py
                tn = name + ".weight"
                if not (_Q_ATTN.search(tn) or _Q_MLP.search(tn)):
                    continue
                target_module_names.append(name)
        # temp limit for cache reuse test
        try:
            import os as _os

            _lim = _os.environ.get("OT_LIMIT_MODULES")
            if _lim:
                target_module_names = target_module_names[: int(_lim)]
                print(f"[test] limiting to {int(_lim)}", flush=True)
        except Exception:
            pass
        if not target_module_names:
            raise ValueError("no quantizable modules found")
    except Exception:
        # fallback canonical: same filter as primary (205)
        import re as _re2

        import torch.nn as _nn  # type: ignore[import]

        _Q_ATTN2 = _re2.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
        _Q_MLP2 = _re2.compile(r"\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$")
        target_module_names = [
            n
            for n, m in model.named_modules()
            if isinstance(m, _nn.Linear)
            and "lm_head" not in n
            and "vision_tower" not in n
            and "audio_tower" not in n
            and "per_layer" not in n
            and "norm" not in n.lower()
            and (_Q_ATTN2.search(n + ".weight") or _Q_MLP2.search(n + ".weight"))
        ]

    # Respect the same configured target policy as quantize/materialize.
    from openternary.quant.fake_quant import _classify_tensor

    named_modules = dict(model.named_modules())
    target_module_names = [
        name
        for name in target_module_names
        if _classify_tensor(name + ".weight", list(named_modules[name].weight.shape), "BF16", app_config)[1]
    ]
    if not target_module_names:
        raise ValueError("no quantizable targets enabled")
    del named_modules

    # 3. Capture train and held activations to disk (sharded) — with fingerprint reuse
    from openternary.calibration.capture import DiskActivationCache, capture_teacher_pairs

    def _cache_fingerprint(
        train_hashes: list[str],
        held_hashes: list[str],
        seq_len: int,
        target_names: list[str],
        revision: str | None,
        dtype_str: str = "bf16",
        *,
        content_fingerprint: str | None = None,
        dataset_fingerprint: str | None = None,
        tokenizer_fingerprint: str | None = None,
        capture_format_version: str = "v2-masked",
    ) -> str:
        """Activation cache fingerprint per spec section 8.

        Includes: teacher model revision, content fingerprint, dataset fingerprint,
        sample hashes, tokenizer fingerprint, seq_len, target module names/count,
        dtype, capture format version (v1).
        """
        h = hashlib.sha256()
        # teacher model revision
        h.update((revision or "").encode())
        # content fingerprint (snapshot content hash)
        if content_fingerprint:
            h.update(content_fingerprint.encode())
        # dataset fingerprint
        if dataset_fingerprint:
            h.update(dataset_fingerprint.encode())
        # sample hashes (train+held)
        for x in train_hashes:
            h.update(x.encode())
        for x in held_hashes:
            h.update(x.encode())
        # tokenizer fingerprint
        if tokenizer_fingerprint:
            h.update(tokenizer_fingerprint.encode())
        else:
            try:
                tok_path = _pl.Path(str(teacher_snapshot)) / "tokenizer.json"
                if tok_path.exists():
                    h.update(hashlib.sha256(tok_path.read_bytes()).hexdigest().encode())
            except Exception:
                pass
        # seq_len
        h.update(str(seq_len).encode())
        # target module names and count
        h.update(str(len(target_names)).encode())
        for x in sorted(target_names):
            h.update(x.encode())
        # dtype
        h.update(dtype_str.encode())
        # capture format version
        h.update(capture_format_version.encode())
        return h.hexdigest()

    def _cache_file_manifest(cache_path: pathlib.Path) -> list[dict[str, typing.Any]]:
        return [
            {
                "file": path.relative_to(cache_path).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(cache_path.rglob("batch_*.pt"))
        ]

    def _validate_cache_manifest(cache_path: pathlib.Path, metadata: dict[str, typing.Any]) -> bool:
        expected = metadata.get("files")
        if not isinstance(expected, list) or not expected:
            return False
        actual_paths = {
            path.relative_to(cache_path).as_posix(): path for path in sorted(cache_path.rglob("batch_*.pt"))
        }
        expected_paths = {entry.get("file") for entry in expected if isinstance(entry, dict)}
        if set(actual_paths) != expected_paths:
            return False
        for entry in expected:
            if not isinstance(entry, dict):
                return False
            relative_file = entry.get("file")
            if not isinstance(relative_file, str):
                return False
            path = actual_paths.get(relative_file)
            if path is None or path.stat().st_size != entry.get("size"):
                return False
            if _sha256_file(path) != entry.get("sha256"):
                return False
        return True

    # Helper to compute fingerprints for cache key
    _content_fp = _snapshot_content_fingerprint(teacher_snapshot)
    _dataset_fp = _dataset_fingerprint(str(effective_dataset), str(requested_dataset))
    _tok_fp = _tokenizer_fingerprint(teacher_snapshot)
    _dtype_str = str(getattr(app_config, "dtype", "bf16"))

    train_cache_dir = out / "artifacts" / "activation_cache_train"
    held_cache_dir = out / "artifacts" / "activation_cache_held"
    # Try reuse from current output dir (resume) or from --init-from
    current_fp = _cache_fingerprint(
        train_hashes,
        held_hashes,
        seq_len,
        target_module_names,
        str(app_config.model.revision),
        dtype_str=_dtype_str,
        content_fingerprint=_content_fp,
        dataset_fingerprint=_dataset_fp,
        tokenizer_fingerprint=_tok_fp,
        capture_format_version="v2-masked",
    )
    resume_contract = {
        "cache_fingerprint": current_fp,
        "source_fingerprint": _content_fp,
        "calibration": calib_cfg.model_dump(),
        "quantization": app_config.quantization.model_dump(),
        "target_module_names": target_module_names,
        "dtype": app_config.dtype,
        "device": app_config.device,
    }
    if resume or resume_from is not None:
        contract_checkpoint = pathlib.Path(resume_from) if resume_from is not None else _find_latest_checkpoint(out)
        if contract_checkpoint is None:
            raise FileNotFoundError("no resumable checkpoint found")
        previous_contract = torch.load(str(contract_checkpoint), map_location="cpu", weights_only=True)
        if previous_contract.get("resume_contract") != resume_contract:
            raise ValueError("resume contract/fingerprint mismatch; legacy checkpoints require explicit recovery")

    # Cross-directory resume carries forward the exact source cache. It is
    # validated below before use; a missing or damaged source cache fails closed.
    if resume_from is not None and not train_cache_dir.exists() and not held_cache_dir.exists():
        import shutil as _resume_shutil

        resume_checkpoint = _pl.Path(resume_from).resolve()
        source_run = resume_checkpoint.parent.parent.parent
        source_train = source_run / "artifacts" / "activation_cache_train"
        source_held = source_run / "artifacts" / "activation_cache_held"
        if source_train.exists() and source_held.exists() and source_run != out.resolve():
            _resume_shutil.copytree(source_train, train_cache_dir)
            _resume_shutil.copytree(source_held, held_cache_dir)
    reuse_possible = False
    reuse_source: pathlib.Path | None = None
    # Check init_from cache for reuse (Phase 4.1 → 4.2)
    if init_from is not None:
        init_train = _pl.Path(str(init_from)) / "artifacts" / "activation_cache_train" / "fingerprint.json"
        init_held = _pl.Path(str(init_from)) / "artifacts" / "activation_cache_held" / "fingerprint.json"
        if init_train.exists() and init_held.exists():
            try:
                init_train_metadata = json.loads(init_train.read_text(encoding="utf-8"))
                init_held_metadata = json.loads(init_held.read_text(encoding="utf-8"))
                init_fp_train = init_train_metadata.get("fingerprint", "")
                init_fp_held = init_held_metadata.get("fingerprint", "")
                if init_fp_train == current_fp and init_fp_held == current_fp:
                    source_train_cache = init_train.parent
                    source_held_cache = init_held.parent
                    source_integrity_ok = _validate_cache_manifest(
                        source_train_cache, init_train_metadata
                    ) and _validate_cache_manifest(source_held_cache, init_held_metadata)
                    if source_integrity_ok:
                        # fingerprints and every source shard match → reuse by copying
                        print(f"[cache] Activation Cache: REUSED from --init-from {init_from}", flush=True)
                        import shutil as _sh2

                        if not train_cache_dir.exists() or not any(train_cache_dir.iterdir()):
                            _sh2.copytree(source_train_cache, train_cache_dir, dirs_exist_ok=True)
                        if not held_cache_dir.exists() or not any(held_cache_dir.iterdir()):
                            _sh2.copytree(source_held_cache, held_cache_dir, dirs_exist_ok=True)
                        reuse_possible = True
                        reuse_source = _pl.Path(str(init_from))
                    else:
                        print(
                            "[cache] Activation Cache: INVALID (--init-from shard integrity mismatch), recapturing",
                            flush=True,
                        )
                else:
                    print(
                        "[cache] Activation Cache: INVALID (fingerprint mismatch with --init-from), recapturing",
                        flush=True,
                    )
            except Exception as e:
                print(f"[cache] warning: init_from fingerprint check failed: {e}", flush=True)
    # Check current output dir cache reuse (e.g., resume without --resume or after interruption)
    existing_fp_train = train_cache_dir / "fingerprint.json"
    existing_fp_held = held_cache_dir / "fingerprint.json"
    if not reuse_possible and existing_fp_train.exists() and existing_fp_held.exists():
        try:
            train_metadata = json.loads(existing_fp_train.read_text(encoding="utf-8"))
            held_metadata = json.loads(existing_fp_held.read_text(encoding="utf-8"))
            ef_train = train_metadata.get("fingerprint", "")
            ef_held = held_metadata.get("fingerprint", "")
            # Need to ensure cache actually has data for all target modules
            train_layers = [p.name for p in train_cache_dir.iterdir() if p.is_dir() and p.name != "_fingerprint"]
            held_layers = [p.name for p in held_cache_dir.iterdir() if p.is_dir()]
            expected_layers = [n.replace(".", "_").replace("/", "_") for n in target_module_names]
            has_all = all(layer_name in train_layers for layer_name in expected_layers) and all(
                layer_name in held_layers for layer_name in expected_layers
            )
            integrity_ok = _validate_cache_manifest(train_cache_dir, train_metadata) and _validate_cache_manifest(
                held_cache_dir, held_metadata
            )
            if (
                (resume or resume_from is not None)
                and ef_train == current_fp
                and ef_held == current_fp
                and not integrity_ok
            ):
                raise ValueError("activation cache integrity failure: shard missing, changed, or unmanifested")
            if ef_train == current_fp and ef_held == current_fp and has_all and integrity_ok:
                print("[cache] Activation Cache: REUSED (existing output cache matches fingerprint)", flush=True)
                reuse_possible = True
            else:
                print(
                    "[cache] Activation Cache: INVALID (existing cache fingerprint mismatch or incomplete), recapturing",
                    flush=True,
                )
        except Exception as e:
            if resume or resume_from is not None:
                raise ValueError(f"activation cache integrity failure: {e}") from e
            print(f"[cache] warning: fingerprint check failed: {e}", flush=True)

    if reuse_possible and reuse_source is not None or (reuse_possible and existing_fp_train.exists()):
        train_cache = DiskActivationCache(train_cache_dir)
        held_cache = DiskActivationCache(held_cache_dir)
        print("[cache] Reusing activation caches, skipping Teacher capture", flush=True)
        # Unload model early since we don't need it
        del model
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass
    else:
        print("[cache] Activation Cache: CAPTURING (no reusable cache)", flush=True)
        train_cache = capture_teacher_pairs(model, train_batches, target_module_names, train_cache_dir)
        held_cache = capture_teacher_pairs(model, held_batches, target_module_names, held_cache_dir)
        # Save fingerprints (include all spec section 8 inputs for provenance)
        try:
            train_cache_dir.mkdir(parents=True, exist_ok=True)
            held_cache_dir.mkdir(parents=True, exist_ok=True)
            _fp_meta = {
                "fingerprint": current_fp,
                "revision": str(app_config.model.revision),
                "content_fingerprint": _content_fp,
                "dataset_fingerprint": _dataset_fp,
                "seq_len": seq_len,
                "target_module_names": sorted(target_module_names),
                "target_module_count": len(target_module_names),
                "dtype": _dtype_str,
                "capture_format_version": "v2-masked",
                "tokenizer_fingerprint": _tok_fp,
                "train_sample_hashes": train_hashes,
                "held_sample_hashes": held_hashes,
            }
            train_fp_meta = dict(_fp_meta)
            train_fp_meta["files"] = _cache_file_manifest(train_cache_dir)
            held_fp_meta = dict(_fp_meta)
            held_fp_meta["files"] = _cache_file_manifest(held_cache_dir)
            (train_cache_dir / "fingerprint.json").write_text(
                json.dumps(train_fp_meta, indent=2),
                encoding="utf-8",
            )
            (held_cache_dir / "fingerprint.json").write_text(
                json.dumps(held_fp_meta, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            raise ValueError("activation cache integrity manifest could not be saved") from e
        # Unload Teacher
        del model
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass
        # Early return already handled, so skip the old unload block below
        # The old unload block is now inside the capture branch, so we need to avoid double delete
        # Set a flag to skip next unload
        _model_unloaded = True  # noqa: F841
    # Model already unloaded in both branches above (reuse or capture)

    # 5. Per-module quant state (codes/scales) from original snapshot weights
    # Load original weights via safetensors
    from safetensors.torch import safe_open  # type: ignore[import]

    from openternary.quant.grouping import quantize_groupwise
    from openternary.quant.ternary import quantize_absmean

    q_cfg = app_config.quantization
    scale_granularity = str(getattr(q_cfg, "scale_granularity", "per_tensor"))
    group_size = int(getattr(q_cfg, "group_size", 8))

    # Helper to load weight tensor for a module (weight name = module_name + ".weight")
    def _load_weight(module_name: str) -> torch.Tensor | None:  # type: ignore[type-arg]
        # try to find safetensors file
        assert teacher_snapshot is not None
        snap_path = _pl.Path(teacher_snapshot)  # type: ignore[arg-type]
        st_files = list(snap_path.glob("*.safetensors"))
        if not st_files:
            return None
        # Use first file that contains the tensor (for Gemma single file)
        weight_name = module_name + ".weight"
        for stf in st_files:
            try:
                with safe_open(str(stf), framework="pt", device="cpu") as f:  # type: ignore[union-attr]
                    if weight_name in f.keys():  # noqa: SIM118 — safe_open requires .keys()
                        return f.get_tensor(weight_name)  # type: ignore[union-attr,no-any-return]
            except Exception:
                continue
        return None

    # Phase 4.2 threshold config
    threshold_enabled = bool(getattr(calib_cfg, "threshold_enabled", False))
    threshold_init_ratio = float(getattr(calib_cfg, "threshold_init_ratio", 0.5))
    threshold_eps = float(getattr(calib_cfg, "threshold_eps", 0.01))
    threshold_ste_width = float(getattr(calib_cfg, "threshold_ste_width", 0.1))
    threshold_lr = getattr(calib_cfg, "threshold_lr", None)
    threshold_lr = float(threshold_lr) if threshold_lr is not None else float(calib_cfg.lr)
    soft_to_hard_enabled = str(getattr(calib_cfg, "method", "recon-scale")) == "recon-soft-to-hard"
    temperature_schedule = str(getattr(calib_cfg, "temperature_schedule", "linear"))
    temperature_start = float(getattr(calib_cfg, "temperature_start", 1.0))
    temperature_end = float(getattr(calib_cfg, "temperature_end", 0.05))
    hard_fraction = float(getattr(calib_cfg, "hard_fraction", 0.1))
    zero_logit_bias = float(getattr(calib_cfg, "zero_logit_bias", 0.0))
    # validation
    if threshold_enabled and not (threshold_eps < threshold_init_ratio < 1 - threshold_eps):
        raise ValueError(
            f"threshold_init_ratio {threshold_init_ratio} must be in ({threshold_eps}, {1 - threshold_eps})"
        )
    if threshold_enabled and str(getattr(calib_cfg, "method", "recon-scale")) not in ("recon-threshold", "recon-scale"):
        # allow recon-scale with threshold_enabled for backward compat, but warn
        pass
    if soft_to_hard_enabled and threshold_enabled:
        raise ValueError("recon-soft-to-hard requires threshold_enabled=false")
    if soft_to_hard_enabled and (
        not math.isfinite(temperature_start)
        or not math.isfinite(temperature_end)
        or temperature_start < temperature_end
        or temperature_end <= 0
        or not math.isfinite(zero_logit_bias)
    ):
        raise ValueError("invalid soft-to-hard temperature or zero_logit_bias configuration")

    # === Module-major (true sequential per-module optimization) per spec 7.1 ===
    # Steps are per-module: each target module runs `steps` optimizer steps independently.
    steps_per_module = int(calib_cfg.steps)
    steps = steps_per_module  # alias for downstream calibration.json compatibility
    ckpt_interval = int(calib_cfg.checkpoint_interval)
    ckpt_dir = out / "artifacts" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _soft_temperature(step_in_module: int) -> float:
        from openternary.quant.soft_ternary import temperature_at_step

        return temperature_at_step(
            step_in_module,
            steps_per_module,
            start=temperature_start,
            end=temperature_end,
            schedule=typing.cast(typing.Any, temperature_schedule),
            hard_fraction=hard_fraction,
        )

    def _soft_weight_hat(
        weight: torch.Tensor,
        reference: torch.Tensor,
        effective: torch.Tensor,
        step_in_module: int,
    ) -> torch.Tensor:
        from openternary.quant.soft_ternary import harden_soft_codes, soft_ternary_codes
        from openternary.quant.threshold import _expand_per_group

        reference_arg: float | torch.Tensor
        soft_group_size: int | None
        if scale_granularity == "per_tensor":
            reference_arg = float(reference[0].item())
            soft_group_size = None
        else:
            reference_arg = reference.to(weight.device)
            soft_group_size = group_size
        temperature = _soft_temperature(step_in_module)
        if temperature == 0.0:
            assignment = harden_soft_codes(
                weight,
                reference_scale=reference_arg,
                group_size=soft_group_size,
            ).to(torch.float32)
        else:
            assignment = soft_ternary_codes(
                weight,
                reference_scale=reference_arg,
                group_size=soft_group_size,
                temperature=temperature,
                zero_logit_bias=zero_logit_bias,
            )
        if scale_granularity == "per_tensor":
            return assignment * effective[0]
        return assignment * _expand_per_group(effective, tuple(weight.shape), group_size)

    def _hard_soft_codes(weight: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        from openternary.quant.soft_ternary import harden_soft_codes

        if scale_granularity == "per_tensor":
            return harden_soft_codes(weight, reference_scale=float(reference[0].item()))
        return harden_soft_codes(weight, reference_scale=reference, group_size=group_size)

    assignment_contract = {
        "mode": "soft-to-hard" if soft_to_hard_enabled else ("threshold-ste" if threshold_enabled else "fixed"),
        "trainable": False if soft_to_hard_enabled else None,
        "hardening_source": "frozen-weight-midpoint" if soft_to_hard_enabled else None,
        "hardening_uses_zero_logit_bias": False if soft_to_hard_enabled else None,
        "temperature_schedule": temperature_schedule if soft_to_hard_enabled else None,
        "temperature_start": temperature_start if soft_to_hard_enabled else None,
        "temperature_end": temperature_end if soft_to_hard_enabled else None,
        "hard_fraction": hard_fraction if soft_to_hard_enabled else None,
        "zero_logit_bias": zero_logit_bias if soft_to_hard_enabled else None,
        "final_state": "hard",
    }

    # Warm start cache for init_from (Phase 4.1 -> 4.2)
    _init_completed: dict[str, dict[str, typing.Any]] = {}
    if init_from is not None and not resume and resume_from is None:
        init_path = _pl.Path(init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from path not found: {init_path}")
        init_ckpt: pathlib.Path | None = None
        if (init_path / "artifacts" / "checkpoint").exists():
            # prefer latest step or module checkpoint
            cands_init = sorted((init_path / "artifacts" / "checkpoint").glob("step_*.pt")) + sorted(
                (init_path / "artifacts" / "checkpoint").glob("module_*.pt")
            )
            if cands_init:
                # pick latest by sorted name
                init_ckpt = sorted(cands_init)[-1]
        elif init_path.is_file() and init_path.suffix == ".pt":
            init_ckpt = init_path
        if init_ckpt is not None and init_ckpt.exists():
            print(f"[init-from] warm start from {init_ckpt}", flush=True)
            try:
                init_data = torch.load(str(init_ckpt), map_location="cpu")
                if isinstance(init_data, dict) and init_data.get("schema_version") == 3:
                    init_contract = init_data.get("resume_contract")
                    if not isinstance(init_contract, dict):
                        raise ValueError("init-from resume contract is missing")
                    if init_contract.get("source_fingerprint") != _content_fp:
                        raise ValueError("init-from source fingerprint mismatch")
                    if init_contract.get("target_module_names") != target_module_names:
                        raise ValueError("init-from target set mismatch")
                    if init_contract.get("quantization") != app_config.quantization.model_dump():
                        raise ValueError("init-from quantization contract mismatch")
                    if init_contract.get("dtype") != app_config.dtype:
                        raise ValueError("init-from dtype contract mismatch")
                # v2 checkpoint: P0 manifest based (empty completed_module_params, need to load from calibration_state)
                if (
                    isinstance(init_data, dict)
                    and init_data.get("schema_version") in (2, 3)
                    and "completed_manifest" in init_data
                ):
                    manifest = init_data.get("completed_manifest", [])
                    # If manifest present and completed_module_params empty, load from files
                    if manifest and not init_data.get("completed_module_params"):
                        for entry in manifest:
                            mname_i = entry.get("module")
                            rel = entry.get("file", "")
                            if not mname_i or not rel:
                                raise ValueError("init-from manifest entry is incomplete")
                            fpath = init_path / rel if rel else None
                            if fpath and fpath.exists():
                                try:
                                    _validate_state_manifest_entry(entry, fpath)
                                    if str(fpath).endswith(".safetensors"):
                                        from safetensors.torch import load_file as _load_sf_init  # type: ignore[import]

                                        payload_i = _load_sf_init(str(fpath), device="cpu")
                                    else:
                                        payload_i = torch.load(str(fpath), map_location="cpu")
                                    _init_completed[mname_i] = {
                                        "raw_param": payload_i.get("raw_param"),
                                        "raw_threshold": payload_i.get("raw_threshold"),
                                    }
                                except Exception as e:
                                    raise ValueError(f"init-from state integrity/load failure: {fpath}") from e
                            else:
                                # fallback try both extensions
                                for ext in (".safetensors", ".pt"):
                                    cand = (
                                        init_path
                                        / "artifacts"
                                        / "calibration_state"
                                        / f"{mname_i.replace('.', '_').replace('/', '_')}{ext}"
                                    )
                                    if cand.exists():
                                        try:
                                            _validate_state_manifest_entry(entry, cand)
                                            if ext == ".safetensors":
                                                from safetensors.torch import load_file as _load_sf_init2  # type: ignore[import]  # noqa: I001

                                                payload_i = _load_sf_init2(str(cand), device="cpu")
                                            else:
                                                payload_i = torch.load(str(cand), map_location="cpu")
                                            _init_completed[mname_i] = {
                                                "raw_param": payload_i.get("raw_param"),
                                                "raw_threshold": payload_i.get("raw_threshold"),
                                            }
                                            break
                                        except Exception as e:
                                            raise ValueError(f"init-from state integrity/load failure: {cand}") from e
                                if mname_i not in _init_completed:
                                    raise ValueError(f"init-from state missing: {mname_i}")
                        print(f"[init-from] loaded {len(_init_completed)} modules via manifest (P0)", flush=True)
                    else:
                        _init_completed = init_data.get("completed_module_params", {})
                        print(f"[init-from] loaded {len(_init_completed)} modules from v2 checkpoint", flush=True)
                elif (
                    isinstance(init_data, dict)
                    and init_data.get("schema_version") in (2, 3)
                    and "completed_module_params" in init_data
                ):
                    _init_completed = init_data.get("completed_module_params", {})
                    # also handle current if needed
                    print(f"[init-from] loaded {len(_init_completed)} modules from v2 checkpoint", flush=True)
                elif isinstance(init_data, dict) and "raw_params" in init_data:
                    # legacy v1: raw_params dict
                    for mname_i, raw in init_data["raw_params"].items():
                        _init_completed[mname_i] = {"raw_param": raw}
                        if (
                            threshold_enabled
                            and "raw_thresholds" in init_data
                            and mname_i in init_data["raw_thresholds"]
                        ):
                            _init_completed[mname_i]["raw_threshold"] = init_data["raw_thresholds"][mname_i]
                    print(
                        f"[init-from] loaded {len(init_data.get('raw_params', {}))} modules' scales (legacy)",
                        flush=True,
                    )
                elif isinstance(init_data, dict) and "raw_param" in init_data:
                    # synthetic single
                    raise ValueError("init-from synthetic single checkpoint is invalid for the real path")
                else:
                    raise ValueError(f"unrecognized init-from checkpoint format: {init_ckpt}")
            except Exception as e:
                if isinstance(e, ValueError) and str(e).startswith("init-from"):
                    raise
                raise ValueError(f"init-from checkpoint cannot be loaded: {init_ckpt}") from e

            if set(_init_completed) != set(target_module_names):
                missing = sorted(set(target_module_names) - set(_init_completed))
                extra = sorted(set(_init_completed) - set(target_module_names))
                raise ValueError(f"init-from target set mismatch: missing={missing[:3]} extra={extra[:3]}")
            if any(state.get("raw_param") is None for state in _init_completed.values()):
                raise ValueError("init-from state missing raw_param")
        else:
            raise FileNotFoundError(f"no init-from checkpoint found in {init_path}")

    # Checkpoint resume handling (v2 module-major with mid-module support)
    completed_module_params: dict[str, dict[str, typing.Any]] = {}
    loss_history: list[float] = []
    initial_loss: float | None = None
    best_loss = float("inf")
    start_module_idx = 0
    start_step_in_module = 0
    _resume_ckpt: dict[str, typing.Any] | None = None
    _resume_opt_state: dict[str, typing.Any] | None = None

    if resume or resume_from is not None:
        _ckpt_path: pathlib.Path | None = None
        if resume_from is not None:
            _ckpt_path = _pl.Path(resume_from)
            if not _ckpt_path.exists():
                raise FileNotFoundError(f"no resumable checkpoint found: {_ckpt_path}")
        elif resume:
            cands_r = sorted(ckpt_dir.glob("step_*.pt")) + sorted(ckpt_dir.glob("module_*.pt"))
            if not cands_r:
                raise FileNotFoundError(f"no resumable checkpoint found in {ckpt_dir}")
            _ckpt_path = sorted(cands_r)[-1]
        if _ckpt_path is not None:
            _resume_ckpt = torch.load(str(_ckpt_path), map_location="cpu")
            if not isinstance(_resume_ckpt, dict) or _resume_ckpt.get("schema_version") != 3:
                raise ValueError("checkpoint schema is not v3; audited conversion is required")
            # check threshold compatibility
            _ckpt_thr_enabled = bool(_resume_ckpt.get("threshold_enabled", False))
            if _ckpt_thr_enabled != threshold_enabled:
                raise ValueError(
                    f"checkpoint threshold_enabled={_ckpt_thr_enabled} != current {threshold_enabled}. Use --init-from for cross-phase warm start, not --resume."
                )
            # method check (warn if mismatch)
            # schema v3 expected
            if _resume_ckpt.get("schema_version") == 3:
                # P0: manifest based — completed modules stored as immutable files, not in checkpoint
                completed_module_params = {}
                manifest = _resume_ckpt.get("completed_manifest", [])
                if manifest:
                    # Phase 4.2 fix: resolve calibration_state from checkpoint source run directory if present
                    _source_run = None
                    if resume_from is not None:
                        try:
                            _p = _pl.Path(resume_from)
                            # walk up to find run root containing artifacts/calibration_state
                            for _anc in [
                                _p.parent,
                                _p.parent.parent,
                                _p.parent.parent.parent,
                                _p.parent.parent.parent.parent,
                            ]:
                                if (_anc / "artifacts" / "calibration_state").exists() or (
                                    _anc / "artifacts" / "checkpoint"
                                ).exists():
                                    _source_run = _anc
                                    break
                            # fallback: if checkpoint is .../artifacts/checkpoint/step_*.pt, run dir is 3 levels up
                            if _source_run is None:
                                _source_run = _p.parent.parent.parent  # type: ignore[assignment]
                        except Exception:
                            _source_run = None
                    for entry in manifest:
                        mname = entry.get("module")
                        rel = entry.get("file", "")
                        if not mname or not rel:
                            continue
                        # resolve file relative to output dir
                        fpath = out / rel
                        if not fpath.exists() and _source_run is not None:
                            # try source run fallback
                            cand_src = _source_run / rel
                            if cand_src.exists():
                                fpath = cand_src
                        if not fpath.exists():
                            # fallback to out/artifacts/calibration_state (try both extensions)
                            sanitized = mname.replace(".", "_").replace("/", "_")
                            cand_s = out / "artifacts" / "calibration_state" / f"{sanitized}.safetensors"
                            cand_p = out / "artifacts" / "calibration_state" / f"{sanitized}.pt"
                            fpath = cand_s if cand_s.exists() else cand_p
                            # also try source run
                            if not fpath.exists() and _source_run is not None:
                                cand_s2 = _source_run / "artifacts" / "calibration_state" / f"{sanitized}.safetensors"
                                cand_p2 = _source_run / "artifacts" / "calibration_state" / f"{sanitized}.pt"
                                if cand_s2.exists():
                                    fpath = cand_s2
                                elif cand_p2.exists():
                                    fpath = cand_p2
                        if fpath.exists():
                            _validate_state_manifest_entry(entry, fpath)
                            try:
                                if str(fpath).endswith(".safetensors"):
                                    from safetensors.torch import load_file as _load_safetensors  # type: ignore[import]

                                    payload = _load_safetensors(str(fpath), device="cpu")
                                else:
                                    payload = torch.load(str(fpath), map_location="cpu")
                                completed_module_params[mname] = {
                                    "raw_param": payload.get("raw_param"),
                                    "raw_threshold": payload.get("raw_threshold"),
                                }
                            except Exception as e:
                                raise ValueError(f"completed state cannot be loaded: {fpath}") from e
                        else:
                            raise ValueError(f"completed manifest file missing: {fpath}")
                else:
                    # legacy aggregated checkpoint (pre-P0)
                    completed_module_params = dict(_resume_ckpt.get("completed_module_params", {}))
                mc = _resume_ckpt.get("module_cursor", {})
                start_module_idx = int(mc.get("module_idx", 0))
                start_step_in_module = int(mc.get("step", 0))
                # if cursor step == steps_per_module, it means module completed; next module should start at 0
                if start_step_in_module >= steps_per_module:
                    start_module_idx = int(mc.get("module_idx", 0)) + 1
                    start_step_in_module = 0
                if not 0 <= start_module_idx <= len(target_module_names):
                    raise ValueError("checkpoint target cursor out of range")
                if set(completed_module_params) != set(target_module_names[:start_module_idx]):
                    raise ValueError("completed manifest does not match target cursor")
                for entry in completed_module_params.values():
                    if entry.get("raw_param") is None or (threshold_enabled and entry.get("raw_threshold") is None):
                        raise ValueError("completed manifest missing required parameters")
                # restore loss_history / best
                loss_history = list(_resume_ckpt.get("loss_history", []))
                initial_loss = _resume_ckpt.get("initial_loss")
                best_loss = _resume_ckpt.get("best_loss", float("inf"))
                _resume_opt_state = _resume_ckpt.get("optimizer_state")
                if start_step_in_module > 0:
                    required_mid_module_state = ["current_raw_param", "optimizer_state", "rng_state"]
                    if threshold_enabled:
                        required_mid_module_state.append("current_raw_threshold")
                    missing_mid_module_state = [
                        key for key in required_mid_module_state if _resume_ckpt.get(key) is None
                    ]
                    if missing_mid_module_state:
                        raise ValueError(
                            "mid-module checkpoint missing transaction state: " + ", ".join(missing_mid_module_state)
                        )
                # sanity: if resume checkpoint already completed some modules, ensure start index within range
                print(
                    f"[resume] v3 checkpoint {_ckpt_path} cursor module_idx={start_module_idx} step={start_step_in_module} completed={len(completed_module_params)}",
                    flush=True,
                )
                # A mid-module resume must restore RNG exactly; terminal checkpoints
                # may omit it because no interrupted update remains to reproduce.
                rng_state = _resume_ckpt.get("rng_state")
                if rng_state is not None:
                    try:
                        torch.set_rng_state(rng_state)
                    except Exception as exc:
                        raise ValueError("checkpoint rng_state cannot be restored") from exc
                cuda_rng_state = _resume_ckpt.get("cuda_rng_state_all")
                if cuda_rng_state is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    except Exception as exc:
                        raise ValueError("checkpoint cuda_rng_state_all cannot be restored") from exc
            else:
                # legacy v1 checkpoint: global step, raw_params for all modules
                # For compatibility, treat as if all modules were jointly optimized and convert to completed
                # We will not support mid-module resume for legacy, just load its raw_params as completed
                legacy_raw = _resume_ckpt.get("raw_params", {})
                for mname_l, raw_l in legacy_raw.items():
                    completed_module_params[mname_l] = {"raw_param": raw_l}
                    if threshold_enabled and mname_l in _resume_ckpt.get("raw_thresholds", {}):
                        completed_module_params[mname_l]["raw_threshold"] = _resume_ckpt["raw_thresholds"][mname_l]
                # legacy had step global; we set start_module_idx to len(completed) to avoid re-doing
                # but legacy stored all modules, so we consider calibration already done if steps matches
                # For resume with more steps, we will restart from 0 with warm start
                # Keep loss_history
                loss_history = []  # legacy loss_history not per-module aggregated; reset
                print(
                    f"[resume] legacy v1 checkpoint {_ckpt_path} loaded {len(legacy_raw)} modules as completed (compat)",
                    flush=True,
                )
                # force non-mid resume: start from first not completed? If all present, we consider done
                if len(completed_module_params) >= len(target_module_names):
                    # all modules already have params, treat as completed run — no further optimization needed
                    # But if steps increased, we still need to re-optimize? For now, skip loop
                    start_module_idx = len(target_module_names)
                else:
                    # partial legacy not expected
                    start_module_idx = 0
                    start_step_in_module = 0

    from openternary.calibration.losses import backward_reconstruction, l1_loss, mse_loss

    _mse = l1_loss if calib_cfg.loss == "l1" else mse_loss
    from openternary.calibration.optimizer import get_effective_scales as _get_eff
    from openternary.quant.grouping import GroupwiseResult as _GWRes
    from openternary.quant.grouping import dequantize_groupwise as _degw

    # Immutable per-module state dir (P0: avoid aggregating 14M*steps into single checkpoint)
    calibration_state_dir = out / "artifacts" / "calibration_state"
    calibration_state_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_mname(name: str) -> str:
        return name.replace(".", "_").replace("/", "_")

    # Helper to save v3 checkpoint (manifest based, immutable per-module state with hashes)
    def _save_v2_checkpoint(
        module_idx: int,
        module_name: str,
        step_in_module: int,
        loss_val: float | None,
        optimizer_state: dict[str, typing.Any] | None,
        current_raw: torch.Tensor | None = None,
        current_thr: torch.Tensor | None = None,
        is_module_done: bool = False,
    ) -> pathlib.Path:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # P0: on module completion, persist immutable per-module state file once (safetensors)
        if is_module_done:
            # completed_module_params already contains this module's final raw
            entry = completed_module_params.get(module_name, {})
            raw_to_save = entry.get("raw_param", current_raw)
            thr_to_save = entry.get("raw_threshold", current_thr)
            if raw_to_save is not None:
                state_file = calibration_state_dir / f"{_sanitize_mname(module_name)}.safetensors"
                tmp_state = state_file.with_suffix(state_file.suffix + ".tmp")
                # Use safetensors for immutable artifact (P0 requirement)
                try:
                    from safetensors.torch import save_file as _save_safetensors  # type: ignore[import]

                    payload_tensors: dict[str, torch.Tensor] = {
                        "raw_param": raw_to_save.detach().cpu().to(torch.float32).contiguous()
                    }
                    if thr_to_save is not None:
                        payload_tensors["raw_threshold"] = thr_to_save.detach().cpu().to(torch.float32).contiguous()
                    _save_safetensors(payload_tensors, str(tmp_state))
                    tmp_state.replace(state_file)
                except Exception:
                    # fallback to torch.save with .pt if safetensors unavailable (should not happen in real env)
                    fallback = state_file.with_suffix(".pt")
                    tmp_fb = fallback.with_suffix(fallback.suffix + ".tmp")
                    payload: dict[str, typing.Any] = {"raw_param": raw_to_save.detach().cpu()}
                    if thr_to_save is not None:
                        payload["raw_threshold"] = thr_to_save.detach().cpu()
                    torch.save(payload, str(tmp_fb))
                    tmp_fb.replace(fallback)
                    state_file = fallback
                try:
                    sz_state = state_file.stat().st_size
                except Exception:
                    sz_state = -1
                print(
                    f"[checkpoint] module {module_name} state saved {state_file.name} ({sz_state} bytes)",
                    flush=True,
                )
        # Determine filename: module checkpoint and also step checkpoint for compatibility
        # Use module_XXXXX for per-module and step_XXXXX for per-step (global)
        # Global step index aggregated across modules
        global_step = module_idx * steps_per_module + step_in_module
        # Save module checkpoint
        mod_path = ckpt_dir / f"module_{module_idx:05d}.pt"
        step_path = ckpt_dir / f"step_{global_step:05d}.pt"
        # Build manifest (no tensors) — list of completed module names with state file existence (P0: .safetensors)
        completed_manifest: list[dict[str, typing.Any]] = []
        manifest_names = list(completed_module_params)
        if is_module_done and module_name not in manifest_names:
            manifest_names.append(module_name)
        for k in manifest_names:
            # Check actual file existence to decide extension (safetensors preferred, pt fallback)
            safep = calibration_state_dir / f"{_sanitize_mname(k)}.safetensors"
            ptp = calibration_state_dir / f"{_sanitize_mname(k)}.pt"
            if safep.exists():
                fname = f"artifacts/calibration_state/{_sanitize_mname(k)}.safetensors"
                state_path = safep
            elif ptp.exists():
                fname = f"artifacts/calibration_state/{_sanitize_mname(k)}.pt"
                state_path = ptp
            else:
                raise ValueError(f"checkpoint state missing before manifest write: {k}")
            completed_manifest.append(_state_manifest_entry(k, fname, state_path))
        ckpt_save: dict[str, typing.Any] = {
            "schema_version": 3,
            "resume_contract": resume_contract,
            "method": str(calib_cfg.method),
            "module_cursor": {
                "module_idx": module_idx if not is_module_done else module_idx + 1,
                "module_name": module_name,
                "step": step_in_module if not is_module_done else 0,
                "steps_per_module": steps_per_module,
                "completed_count": len(completed_manifest),
            },
            "completed_manifest": completed_manifest,
            # P0: completed_module_params must stay empty to avoid 102MB bloat; manifest is source of truth
            "completed_module_params": {},
            "loss_history": list(loss_history),
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "threshold_enabled": threshold_enabled,
            "scale_granularity": scale_granularity,
            "threshold_eps": threshold_eps,
            "threshold_ste_width": threshold_ste_width,
            "assignment": {
                **assignment_contract,
                "next_temperature": (
                    _soft_temperature(step_in_module)
                    if soft_to_hard_enabled and not is_module_done and step_in_module < steps_per_module
                    else None
                ),
            },
        }
        if optimizer_state is not None:
            ckpt_save["optimizer_state"] = optimizer_state
        # P0: checkpoint contains only current module params for mid-module resume (no agg duplication)
        if current_raw is not None:
            ckpt_save["current_raw_param"] = current_raw.detach().cpu()
        if current_thr is not None:
            ckpt_save["current_raw_threshold"] = current_thr.detach().cpu()
        # Do not duplicate via raw_params/raw_thresholds (agg_raw二重保存撤廃)
        ckpt_save["step"] = global_step
        ckpt_save["loss"] = loss_val
        # Include RNG state for determinism
        ckpt_save["rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            ckpt_save["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        # Save atomically via temp file to avoid partial zip (Windows antivirus / concurrent read)
        tmp_step = step_path.with_suffix(step_path.suffix + ".tmp")
        torch.save(ckpt_save, str(tmp_step))
        tmp_step.replace(step_path)
        # Verbose byte size log (P0 requirement)
        try:
            sz = step_path.stat().st_size
            print(f"[checkpoint] saved {step_path.name} ({sz} bytes, cursor {ckpt_save['module_cursor']})", flush=True)
        except Exception:
            pass
        # Only overwrite module path when module done or checkpoint interval
        with contextlib.suppress(Exception):
            tmp_mod = mod_path.with_suffix(mod_path.suffix + ".tmp")
            torch.save(ckpt_save, str(tmp_mod))
            tmp_mod.replace(mod_path)
            try:
                sz2 = mod_path.stat().st_size
                print(f"[checkpoint] module file {mod_path.name} ({sz2} bytes)", flush=True)
            except Exception:
                pass
        return step_path

    # Sequential per-module optimization (bounded memory)
    import gc as _gc

    runtime_module_records: list[dict[str, typing.Any]] = []
    total_oom_retries = 0
    minimum_microbatch_used: int | None = None
    initial_microbatch_max = 0

    for module_idx, mname in enumerate(target_module_names):
        if module_idx < start_module_idx:
            continue
        # Skip if already completed (e.g., resumed and already in completed dict) unless it's the active resume module
        if mname in completed_module_params and module_idx != start_module_idx:
            continue
        # Load weight on demand (bounded)
        w = _load_weight(mname)
        if w is None:
            raise ValueError(f"target weight missing: {mname}")
        w_f32 = w.to(torch.float32)
        # Quantize to get codes/orig_scales
        if scale_granularity == "per_tensor":
            tt = quantize_absmean(w_f32)
            orig_scales = torch.tensor([tt.scale], dtype=torch.float32)
            codes = tt.codes
            zero_mask = orig_scales == 0
        else:
            res = quantize_groupwise(w_f32, group_size)
            orig_scales = res.scales.clone()
            codes = res.codes
            zero_mask = orig_scales == 0
        raw_param, zero_mask_t = build_scale_params(orig_scales, zero_mask)
        eff_init = get_effective_scales(raw_param, zero_mask_t)
        if not torch.allclose(eff_init, orig_scales, atol=1e-6):
            raise ValueError(f"effective initial mismatch for {mname}")
        reference_scales = orig_scales.clone()

        raw_thr = None
        thr_zero_mask = None
        if threshold_enabled:
            n_thr = int(orig_scales.numel())
            raw_thr, thr_zero_mask = build_threshold_params(
                n_thr,
                init_ratio=threshold_init_ratio,
                eps=threshold_eps,
                device=orig_scales.device,
                zero_mask=zero_mask,
            )
            thr_init = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)
            from openternary.quant.threshold import hard_threshold_codes

            if scale_granularity == "per_tensor":
                hard0 = hard_threshold_codes(w_f32, float(reference_scales[0].item()), float(thr_init[0].item()))
            else:
                hard0 = hard_threshold_codes(w_f32, reference_scales, thr_init, group_size=group_size)
            if threshold_init_ratio == 0.5 and not torch.equal(hard0, codes):
                raise ValueError(f"threshold step0 parity failed for {mname}")

        # Warm start from init_from if available for this module
        if mname in _init_completed:
            try:
                ic = _init_completed[mname]
                if "raw_param" in ic and ic["raw_param"] is not None:
                    raw_param.data = ic["raw_param"].to(raw_param.device)
                if (
                    threshold_enabled
                    and raw_thr is not None
                    and "raw_threshold" in ic
                    and ic["raw_threshold"] is not None
                ):
                    raw_thr.data = ic["raw_threshold"].to(raw_thr.device)
                print(f"[init-from] warm start applied for {mname}", flush=True)
            except Exception as e:
                raise ValueError(f"init-from state cannot be applied for {mname}") from e

        # Resume mid-module: restore current params if this is the resume module
        is_resume_active = module_idx == start_module_idx and start_step_in_module > 0 and _resume_ckpt is not None
        if is_resume_active:
            assert _resume_ckpt is not None
            cur_raw = _resume_ckpt["current_raw_param"]
            if not isinstance(cur_raw, torch.Tensor) or cur_raw.shape != raw_param.shape:
                raise ValueError(f"checkpoint current_raw_param is invalid for {mname}")
            raw_param.data.copy_(cur_raw.to(device=raw_param.device, dtype=raw_param.dtype))
            if threshold_enabled and raw_thr is not None:
                cur_thr = _resume_ckpt["current_raw_threshold"]
                if not isinstance(cur_thr, torch.Tensor) or cur_thr.shape != raw_thr.shape:
                    raise ValueError(f"checkpoint current_raw_threshold is invalid for {mname}")
                raw_thr.data.copy_(cur_thr.to(device=raw_thr.device, dtype=raw_thr.dtype))
            print(f"[resume] restored mid-module {mname} step {start_step_in_module}", flush=True)

        # Create local Adam for this module only (bounded)
        optimizer_class = torch.optim.SGD if calib_cfg.optimizer == "sgd" else torch.optim.Adam
        if threshold_enabled:
            assert raw_thr is not None
            optimizer = optimizer_class(
                [
                    {"params": [raw_param], "lr": float(calib_cfg.lr)},
                    {"params": [raw_thr], "lr": threshold_lr},
                ]
            )
        else:
            optimizer = optimizer_class([raw_param], lr=float(calib_cfg.lr))
        if is_resume_active and _resume_opt_state is not None:
            try:
                optimizer.load_state_dict(_resume_opt_state)
            except Exception as e:
                raise ValueError(f"checkpoint optimizer_state cannot be restored for {mname}") from e

        # Determine start step for this module (0 or resumed offset)
        step_start = start_step_in_module if module_idx == start_module_idx else 0

        # --- Phase 4.2-G Dynamic VRAM / microbatch / precision setup ---
        # Decide device for this module: only current module on GPU, others on CPU/disk
        _use_gpu = False
        _gpu_device = "cpu"
        _vram_budget: VramBudget | None = None
        if backend_info.backend != "cpu" and _low_vram:
            try:
                # Use real mem_get_info if available, else simulated backend_info (OT_FORCE_ROCM)
                if torch is not None and torch.cuda.is_available():
                    try:
                        fb, tb = torch.cuda.mem_get_info(0)  # type: ignore[union-attr]
                    except Exception:
                        fb, tb = backend_info.free_bytes, backend_info.total_bytes
                else:
                    # simulated (OT_FORCE_ROCM) or CPU fallback — use backend_info values
                    fb, tb = backend_info.free_bytes, backend_info.total_bytes
                _vram_budget = compute_vram_budget(int(fb), int(tb), _max_fraction, _reserve_mb, _min_free_mb)
                # Intentional low budget for gate test
                import os as _os2

                if _os2.environ.get("OT_FORCE_LOW_BUDGET") == "1" and module_idx == 0:
                    # force tiny budget to trigger shrink log
                    _vram_budget = VramBudget(
                        total_bytes=_vram_budget.total_bytes,
                        free_bytes=1024 * 1024 * 100,
                        budget_bytes=1024 * 1024 * 50,
                        reserved_bytes=_vram_budget.reserved_bytes,
                        min_free_bytes=_vram_budget.min_free_bytes,
                    )
                if _vram_budget.budget_bytes > 10 * 1024 * 1024:
                    _use_gpu = True
                    _gpu_device = "cuda:0"
                    print(
                        f"[vram] module={module_idx + 1}/{len(target_module_names)} {mname} budget={format_bytes(_vram_budget.budget_bytes)} free={format_bytes(int(fb))} total={format_bytes(int(tb))}",
                        flush=True,
                    )
                else:
                    print(
                        f"[vram] module={module_idx + 1}/{len(target_module_names)} {mname} budget too low {format_bytes(_vram_budget.budget_bytes)} -> cpu",
                        flush=True,
                    )
            except Exception as e:
                print(f"[vram] warning: budget check failed {e} -> cpu", flush=True)
                _use_gpu = False

        # BF16 capability for this backend
        _real_gpu = bool(torch is not None and torch.cuda.is_available())
        _bf16_ok = bool(backend_info.bf16_supported and _use_gpu and _real_gpu)
        # microbatch sizes to try
        n_batches_probe = train_cache.count_batches(mname.replace(".", "_"))
        total_train_elements = sum(
            train_cache.load_valid(mname, i)["teacher_output"].numel() for i in range(n_batches_probe)
        )
        if total_train_elements == 0:
            raise ValueError(f"no valid activation elements for {mname}")
        _initial_mb = int(n_batches_probe) if n_batches_probe > 0 else 1
        if _initial_mb <= 0:
            _initial_mb = 1
        _mb_candidates: list[int] = []
        if _micro_auto and _use_gpu:
            cur = _initial_mb
            _mb_candidates.append(cur)
            while cur > _micro_min:
                cur = max(_micro_min, cur // 2)
                if cur not in _mb_candidates:
                    _mb_candidates.append(cur)
                if cur == _micro_min:
                    break
        else:
            # cpu or no auto: single size
            _mb_candidates = [_initial_mb if not _use_gpu else max(_micro_min, _initial_mb)]

        # Peak tracking per module
        _peak_before = 0
        if _use_gpu and torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(0)  # type: ignore[union-attr]
                _peak_before = int(torch.cuda.max_memory_allocated(0))  # type: ignore[union-attr]
            except Exception:
                pass

        # Helper to run steps on specific device with microbatch size
        def _run_steps_on_device(dev_str: str, microbatch: int) -> bool:
            # Move module tensors to device (BF16 for W/activation where possible, FP32 for raw)
            # We will move per-step; keep raw_param on dev_str
            # Note: caller must have moved raw_param etc already if needed
            # This inner function assumes raw_param etc are on correct device
            # For simplicity, we handle device conversion inside step loop per tensor
            return True  # placeholder, real logic below uses outer vars

        # Try GPU with microbatch shrink + OOM fallback
        _module_done_on_gpu = False
        _fallback_to_cpu = False
        _chosen_mb = _mb_candidates[0] if _mb_candidates else 1
        initial_microbatch_max = max(initial_microbatch_max, _chosen_mb)

        # Transactional GPU path. Every logical optimizer update is isolated:
        # OOM restores params, optimizer moments, RNG, and metric history before
        # retrying that same step with a smaller microbatch.
        if _use_gpu:
            if _gpu_device == "cuda:0" and _real_gpu:
                raw_param.data = raw_param.data.to(_gpu_device)
                zero_mask_t = zero_mask_t.to(_gpu_device)  # type: ignore[assignment]
                if threshold_enabled and raw_thr is not None:
                    raw_thr.data = raw_thr.data.to(_gpu_device)

            params_for_transaction = [raw_param]
            if threshold_enabled and raw_thr is not None:
                params_for_transaction.append(raw_thr)
            mb_index = 0
            module_oom_retries = 0

            for step_in_mod in range(step_start, steps_per_module):
                while True:
                    _chosen_mb = _mb_candidates[mb_index]
                    minimum_microbatch_used = (
                        _chosen_mb if minimum_microbatch_used is None else min(minimum_microbatch_used, _chosen_mb)
                    )
                    print(
                        f"[vram] microbatch={_chosen_mb} for {mname} step={step_in_mod + 1}/{steps_per_module}",
                        flush=True,
                    )
                    transaction = _capture_optimizer_transaction(params_for_transaction, optimizer)
                    history_len = len(loss_history)
                    previous_initial = initial_loss
                    previous_best = best_loss
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        n_batches = train_cache.count_batches(mname.replace(".", "_"))
                        if n_batches == 0:
                            raise ValueError(f"no activation batches for {mname}")
                        step_loss_val = 0.0
                        weight_tensor_local = w_f32.to(_gpu_device) if _real_gpu else w_f32

                        for chunk_start in range(0, n_batches, _chosen_mb):
                            chunk_end = min(chunk_start + _chosen_mb, n_batches)
                            chunk = [train_cache.load_valid(mname, idx) for idx in range(chunk_start, chunk_end)]
                            inp = torch.cat([item["input"] for item in chunk], dim=0)
                            tout = torch.cat([item["teacher_output"] for item in chunk], dim=0)
                            if _real_gpu:
                                input_dtype = torch.bfloat16 if _bf16_ok else torch.float32
                                inp_dev = inp.to(_gpu_device, dtype=input_dtype)
                                tout_dev = tout.to(_gpu_device, dtype=torch.float32)
                            else:
                                inp_dev = inp.to(torch.float32)
                                tout_dev = tout.to(torch.float32)

                            eff = _get_eff(raw_param, zero_mask_t)
                            if threshold_enabled:
                                assert raw_thr is not None
                                from openternary.quant.threshold import _expand_per_group as _exp_thr
                                from openternary.quant.threshold import ste_threshold_codes as _ste_codes

                                _transaction_thr_ratio = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)
                                if scale_granularity == "per_tensor":
                                    codes_ste = _ste_codes(
                                        weight_tensor_local,
                                        float(reference_scales[0].item()),
                                        _transaction_thr_ratio[0].view(1),
                                        ste_width=threshold_ste_width,
                                    )
                                    w_hat = codes_ste * eff[0]
                                else:
                                    ref_local = reference_scales.to(_gpu_device) if _real_gpu else reference_scales
                                    codes_ste = _ste_codes(
                                        weight_tensor_local,
                                        ref_local,
                                        _transaction_thr_ratio,
                                        group_size=group_size,
                                        ste_width=threshold_ste_width,
                                    )
                                    w_hat = codes_ste * _exp_thr(eff, tuple(weight_tensor_local.shape), group_size)
                            elif soft_to_hard_enabled:
                                w_hat = _soft_weight_hat(
                                    weight_tensor_local,
                                    reference_scales,
                                    eff,
                                    step_in_mod,
                                )
                            elif scale_granularity == "per_tensor":
                                code_local = codes.to(_gpu_device) if _real_gpu else codes
                                w_hat = code_local.to(torch.float32) * eff[0]
                            else:
                                res_tmp = _GWRes(
                                    codes=codes.to(_gpu_device) if _real_gpu else codes,
                                    scales=eff,
                                    shape=tuple(w.shape),
                                    orig_dtype=str(w.dtype),
                                    group_size=group_size,
                                    grouping_scheme="last-dim-rowwise-v1",
                                )
                                w_hat = _degw(res_tmp)

                            try:
                                if _bf16_ok and _real_gpu:
                                    y_hat = F.linear(inp_dev, w_hat.to(torch.bfloat16))  # type: ignore[union-attr]
                                else:
                                    y_hat = F.linear(  # type: ignore[union-attr]
                                        inp_dev.to(torch.float32), w_hat.to(torch.float32)
                                    )
                            except Exception as linear_error:
                                if _bf16_ok and _real_gpu and "bf16" in str(linear_error).lower():
                                    y_hat = F.linear(  # type: ignore[union-attr]
                                        inp_dev.to(torch.float32), w_hat.to(torch.float32)
                                    )
                                else:
                                    raise
                            step_loss_val += backward_reconstruction(
                                tout_dev, y_hat, total_train_elements, str(calib_cfg.loss)
                            )

                        optimizer.step()
                    except Exception as error:
                        if not is_oom_error(error):
                            raise
                        _restore_optimizer_transaction(transaction, params_for_transaction, optimizer)
                        del loss_history[history_len:]
                        initial_loss = previous_initial
                        best_loss = previous_best
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            with contextlib.suppress(Exception):
                                torch.cuda.empty_cache()
                        if not _micro_auto or mb_index + 1 >= len(_mb_candidates):
                            raise RuntimeError(
                                "OOM: minimum microbatch exhausted; state restored to the start of the failed step"
                            ) from error
                        previous_mb = _chosen_mb
                        mb_index += 1
                        module_oom_retries += 1
                        total_oom_retries += 1
                        print(
                            f"[vram] OOM rollback complete; retrying same step microbatch={previous_mb}->{_mb_candidates[mb_index]}",
                            flush=True,
                        )
                        continue

                    if initial_loss is None:
                        initial_loss = step_loss_val
                    best_loss = min(best_loss, step_loss_val)
                    loss_history.append(step_loss_val)
                    if (step_in_mod + 1) % ckpt_interval == 0 or (step_in_mod + 1) == steps_per_module:
                        _save_v2_checkpoint(
                            module_idx,
                            mname,
                            step_in_mod + 1,
                            step_loss_val,
                            optimizer.state_dict(),
                            current_raw=raw_param.detach().cpu(),
                            current_thr=raw_thr.detach().cpu() if threshold_enabled and raw_thr is not None else None,
                            is_module_done=(step_in_mod + 1) == steps_per_module,
                        )
                    break

            _module_done_on_gpu = True
            runtime_module_records.append(
                {
                    "module": mname,
                    "initial_microbatch": _mb_candidates[0],
                    "selected_microbatch": _chosen_mb,
                    "oom_retries": module_oom_retries,
                    "device": _gpu_device if _real_gpu else "simulated-gpu-path",
                }
            )

        # Legacy branch retained only as an unreachable compatibility guard while
        # the transactional path is exercised by production tests.
        if _use_gpu and not _module_done_on_gpu:
            for _mb_try in _mb_candidates:
                _chosen_mb = _mb_try
                print(
                    f"[vram] microbatch={_mb_try} for {mname} (try {_mb_candidates.index(_mb_try) + 1}/{len(_mb_candidates)})",
                    flush=True,
                )
                # For intentional low budget test, force OOM on first try
                import os as _os3

                if (
                    _os3.environ.get("OT_FORCE_LOW_BUDGET") == "1"
                    and _mb_try == _mb_candidates[0]
                    and len(_mb_candidates) > 1
                ):
                    print(f"[vram] OOM → retry microbatch {_mb_try} -> {_mb_candidates[1]} (forced)", flush=True)
                    continue
                try:
                    # --- actual GPU steps with microbatch ---
                    # Move raw params to gpu for this attempt
                    if dev_str := _gpu_device != "cpu":  # noqa: F841
                        pass
                    # Temporarily move raw_param/raw_thr to gpu if needed
                    _orig_raw_device = str(raw_param.device)
                    _orig_thr_device = str(raw_thr.device) if threshold_enabled and raw_thr is not None else None
                    # Move to gpu (only if real GPU available; simulated keeps on cpu)
                    if _gpu_device == "cuda:0" and _real_gpu:
                        raw_param.data = raw_param.data.to("cuda:0")
                        zero_mask_t = zero_mask_t.to("cuda:0")  # type: ignore[assignment]
                        if threshold_enabled and raw_thr is not None:
                            raw_thr.data = raw_thr.data.to("cuda:0")
                        # Also move optimizer params (they reference raw_param)
                        # Need to recreate optimizer on gpu device (Adam state will be on gpu)
                        # For simplicity, keep optimizer but its params now on gpu — state will migrate on step
                        # Ensure reference_scales etc also on gpu for threshold
                        # We will handle per-step moves
                        pass

                    # Run steps
                    for step_in_mod in range(step_start, steps_per_module):
                        optimizer.zero_grad()
                        n_batches = train_cache.count_batches(mname.replace(".", "_"))
                        if n_batches == 0:
                            print(f"[module-major] warning: no batches for {mname}, skipping", flush=True)
                            break
                        step_loss_val = 0.0
                        weight_tensor_local = w_f32.to(_gpu_device) if (_use_gpu and _real_gpu) else w_f32
                        # thr ratio on device
                        thr_ratio_local = None
                        if threshold_enabled:
                            # ensure raw_thr on device
                            thr_ratio_local = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)  # type: ignore[arg-type]
                            if _use_gpu and _real_gpu:
                                thr_ratio_local = thr_ratio_local.to(_gpu_device)  # type: ignore[union-attr]
                        # microbatch chunking: stack inputs for efficiency, but still handle OOM per chunk
                        # For now chunk by microbatch size: process in groups
                        for chunk_start in range(0, n_batches, _mb_try):
                            chunk_end = min(chunk_start + _mb_try, n_batches)
                            for b_idx in range(chunk_start, chunk_end):
                                data = train_cache.load_valid(mname, b_idx)
                                inp = data["input"]
                                tout = data["teacher_output"]
                                # move activation to device with BF16 if possible
                                if _use_gpu and _real_gpu:
                                    if _bf16_ok:
                                        # BF16 for activation/W_hat, keep tout in fp32 for loss
                                        inp_dev = inp.to(_gpu_device).to(torch.bfloat16)  # type: ignore[union-attr]
                                        tout_dev = tout.to(_gpu_device).to(torch.float32)  # type: ignore[union-attr]
                                    else:
                                        inp_dev = inp.to(_gpu_device).to(torch.float32)  # type: ignore[union-attr]
                                        tout_dev = tout.to(_gpu_device).to(torch.float32)  # type: ignore[union-attr]
                                else:
                                    inp_dev = inp
                                    tout_dev = tout
                                # effective scale on device
                                eff = _get_eff(raw_param, zero_mask_t)
                                if threshold_enabled:
                                    assert raw_thr is not None
                                    thr_ratio_local = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)
                                if _use_gpu and _real_gpu:
                                    eff = eff.to(_gpu_device)  # type: ignore[union-attr]
                                if threshold_enabled:
                                    assert thr_ratio_local is not None
                                    from openternary.quant.threshold import _expand_per_group as _exp_thr
                                    from openternary.quant.threshold import ste_threshold_codes as _ste_codes

                                    if scale_granularity == "per_tensor":
                                        thr_s = thr_ratio_local[0].view(1)
                                        ref_s = float(reference_scales[0].item())
                                        # weight on device
                                        wt_dev = (
                                            weight_tensor_local.to(_gpu_device)
                                            if (_use_gpu and _real_gpu)
                                            else weight_tensor_local
                                        )  # type: ignore[union-attr]
                                        codes_ste = _ste_codes(wt_dev, ref_s, thr_s, ste_width=threshold_ste_width)
                                        # w_hat in BF16 if possible
                                        if _bf16_ok and _use_gpu and _real_gpu:
                                            w_hat = (codes_ste.to(torch.bfloat16) * eff[0].to(torch.bfloat16)).to(
                                                torch.float32
                                            )  # type: ignore[union-attr]
                                        else:
                                            w_hat = codes_ste * eff[0]
                                    else:
                                        wt_dev = (
                                            weight_tensor_local.to(_gpu_device)
                                            if (_use_gpu and _real_gpu)
                                            else weight_tensor_local
                                        )  # type: ignore[union-attr]
                                        ref_dev = (
                                            reference_scales.to(_gpu_device)
                                            if (_use_gpu and _real_gpu)
                                            else reference_scales
                                        )  # type: ignore[union-attr]
                                        codes_ste = _ste_codes(
                                            wt_dev,
                                            ref_dev,
                                            thr_ratio_local,
                                            group_size=group_size,
                                            ste_width=threshold_ste_width,
                                        )
                                        if _bf16_ok and _use_gpu and _real_gpu:
                                            eff_exp = _exp_thr(eff.to(torch.bfloat16), tuple(wt_dev.shape), group_size)  # type: ignore[union-attr]
                                            w_hat = (codes_ste.to(torch.bfloat16) * eff_exp).to(torch.float32)  # type: ignore[union-attr]
                                        else:
                                            eff_exp = _exp_thr(eff, tuple(wt_dev.shape), group_size)
                                            w_hat = codes_ste * eff_exp
                                elif soft_to_hard_enabled:
                                    w_hat = _soft_weight_hat(
                                        weight_tensor_local,
                                        reference_scales,
                                        eff,
                                        step_in_mod,
                                    )
                                else:
                                    if scale_granularity == "per_tensor":
                                        w_hat = (
                                            codes.to(_gpu_device).to(torch.float32) * eff[0]
                                            if (_use_gpu and _real_gpu)
                                            else codes.to(torch.float32) * eff[0]
                                        )  # type: ignore[union-attr]
                                    else:
                                        res_tmp = _GWRes(
                                            codes=codes.to(_gpu_device) if (_use_gpu and _real_gpu) else codes,  # type: ignore[union-attr]
                                            scales=eff,
                                            shape=tuple(w.shape),
                                            orig_dtype=str(w.dtype),
                                            group_size=group_size,
                                            grouping_scheme="last-dim-rowwise-v1",
                                        )
                                        w_hat = _degw(res_tmp)
                                        if _use_gpu and _real_gpu:
                                            w_hat = w_hat.to(_gpu_device)  # type: ignore[union-attr]
                                # F.linear with BF16 fallback per-op — both operands same dtype
                                try:
                                    if _bf16_ok and _use_gpu and _real_gpu:
                                        y_hat = F.linear(inp_dev.to(torch.bfloat16), w_hat.to(torch.bfloat16))  # type: ignore[union-attr]
                                    else:
                                        y_hat = F.linear(inp_dev.to(torch.float32), w_hat.to(torch.float32))  # type: ignore[union-attr]
                                except Exception as e_linear:
                                    # ROCm BF16 kernel unstable -> fallback to FP32 for this op (both FP32)
                                    if _bf16_ok and "bf16" in str(e_linear).lower():
                                        y_hat = F.linear(inp_dev.to(torch.float32), w_hat.to(torch.float32))  # type: ignore[union-attr]
                                    else:
                                        raise
                                step_loss_val += backward_reconstruction(
                                    tout_dev, y_hat, total_train_elements, str(calib_cfg.loss)
                                )
                        if initial_loss is None:
                            initial_loss = step_loss_val
                        best_loss = min(best_loss, step_loss_val)
                        loss_history.append(step_loss_val)
                        optimizer.step()
                        # checkpoint
                        if (step_in_mod + 1) % ckpt_interval == 0 or (step_in_mod + 1) == steps_per_module:
                            # need to save current raw on cpu for checkpoint (ensure cpu)
                            _save_raw_cpu = raw_param.detach().cpu()
                            _save_thr_cpu = (
                                raw_thr.detach().cpu() if threshold_enabled and raw_thr is not None else None
                            )
                            _save_v2_checkpoint(
                                module_idx,
                                mname,
                                step_in_mod + 1,
                                step_loss_val,
                                optimizer.state_dict(),
                                current_raw=_save_raw_cpu,
                                current_thr=_save_thr_cpu,
                                is_module_done=(step_in_mod + 1) == steps_per_module,
                            )
                    # If we reached here without OOM, success
                    _module_done_on_gpu = True
                    # Log VRAM stats
                    if _use_gpu and torch is not None and torch.cuda.is_available():
                        try:
                            alloc = int(torch.cuda.memory_allocated(0))  # type: ignore[union-attr]
                            reserved = int(torch.cuda.memory_reserved(0))  # type: ignore[union-attr]
                            peak = int(torch.cuda.max_memory_allocated(0))  # type: ignore[union-attr]
                            print(
                                f"[vram] allocated={format_bytes(alloc)} reserved={format_bytes(reserved)} peak={format_bytes(peak)} microbatch={_mb_try}",
                                flush=True,
                            )
                        except Exception:
                            pass
                    break
                except Exception as e:
                    if is_oom_error(e):
                        optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(
                            "OOM: calibration stopped without retrying partially updated state. "
                            "Resume from the last durable checkpoint after reviewing memory requirements."
                        ) from e
                    else:
                        raise
            if not _module_done_on_gpu:
                _fallback_to_cpu = True
                print(f"[vram] fallback=cpu for {mname}", flush=True)
                # move back to cpu for fallback
                try:
                    raw_param.data = raw_param.data.to("cpu")
                    if threshold_enabled and raw_thr is not None:
                        raw_thr.data = raw_thr.data.to("cpu")
                    zero_mask_t = zero_mask_t.to("cpu")  # type: ignore[assignment]
                except Exception:
                    pass
        # If not using GPU, run original CPU path
        if not _use_gpu or _fallback_to_cpu:
            # Original CPU loop (preserved)
            for step_in_mod in range(step_start, steps_per_module):
                optimizer.zero_grad()
                n_batches = train_cache.count_batches(mname.replace(".", "_"))
                if n_batches == 0:
                    print(f"[module-major] warning: no batches for {mname}, skipping", flush=True)
                    break
                step_loss_val = 0.0
                weight_tensor_local = w_f32
                thr_ratio_local = None
                if threshold_enabled:
                    thr_ratio_local = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)  # type: ignore[arg-type]
                for b_idx in range(n_batches):
                    data = train_cache.load_valid(mname, b_idx)
                    inp = data["input"]
                    tout = data["teacher_output"]
                    eff = _get_eff(raw_param, zero_mask_t)
                    if threshold_enabled:
                        assert raw_thr is not None
                        thr_ratio_local = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)
                    if threshold_enabled:
                        assert thr_ratio_local is not None
                        from openternary.quant.threshold import _expand_per_group as _exp_thr
                        from openternary.quant.threshold import ste_threshold_codes as _ste_codes

                        if scale_granularity == "per_tensor":
                            thr_s = thr_ratio_local[0].view(1)
                            ref_s = float(reference_scales[0].item())
                            codes_ste = _ste_codes(weight_tensor_local, ref_s, thr_s, ste_width=threshold_ste_width)
                            w_hat = codes_ste * eff[0]
                        else:
                            codes_ste = _ste_codes(
                                weight_tensor_local,
                                reference_scales,
                                thr_ratio_local,
                                group_size=group_size,
                                ste_width=threshold_ste_width,
                            )
                            eff_exp = _exp_thr(eff, tuple(weight_tensor_local.shape), group_size)
                            w_hat = codes_ste * eff_exp
                    elif soft_to_hard_enabled:
                        w_hat = _soft_weight_hat(
                            weight_tensor_local,
                            reference_scales,
                            eff,
                            step_in_mod,
                        )
                    else:
                        if scale_granularity == "per_tensor":
                            w_hat = codes.to(torch.float32) * eff[0]
                        else:
                            res_tmp = _GWRes(
                                codes=codes,
                                scales=eff,
                                shape=tuple(w.shape),
                                orig_dtype=str(w.dtype),
                                group_size=group_size,
                                grouping_scheme="last-dim-rowwise-v1",
                            )
                            w_hat = _degw(res_tmp)
                    y_hat = F.linear(inp.to(torch.float32), w_hat.to(torch.float32))  # type: ignore[union-attr]
                    step_loss_val += backward_reconstruction(tout, y_hat, total_train_elements, str(calib_cfg.loss))
                if initial_loss is None:
                    initial_loss = step_loss_val
                best_loss = min(best_loss, step_loss_val)
                loss_history.append(step_loss_val)
                optimizer.step()
                if (step_in_mod + 1) % ckpt_interval == 0 or (step_in_mod + 1) == steps_per_module:
                    _save_v2_checkpoint(
                        module_idx,
                        mname,
                        step_in_mod + 1,
                        step_loss_val,
                        optimizer.state_dict(),
                        current_raw=raw_param,
                        current_thr=raw_thr if threshold_enabled else None,
                        is_module_done=(step_in_mod + 1) == steps_per_module,
                    )
        else:
            # GPU path already handled steps and checkpoints above, so skip duplicate CPU loop
            # Need to ensure loss_history already populated via GPU path — if GPU path ran, we already appended
            # If GPU path succeeded, we should not re-run CPU
            pass
        # After module steps, wrap up (handled per path)
        # For GPU path, loss_history already has entries; for CPU path also
        # Ensure we have peak tracking for GPU modules
        if _use_gpu and not _fallback_to_cpu:
            try:
                if torch is not None and torch.cuda.is_available():
                    peak = int(torch.cuda.max_memory_allocated(0))  # type: ignore[union-attr]
                    # store peak for metrics (global)
                    if not hasattr(run_calibration, "_peak_vram"):
                        run_calibration._peak_vram = 0  # type: ignore[attr-defined]
                    run_calibration._peak_vram = max(int(getattr(run_calibration, "_peak_vram", 0)), peak)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Common cleanup after module
        # Note: for GPU path, raw_param already on gpu, need to move back for final_entry
        # Ensure final raw on cpu
        try:
            if _use_gpu and not _fallback_to_cpu:
                raw_param.data = raw_param.data.to("cpu")
                if threshold_enabled and raw_thr is not None:
                    raw_thr.data = raw_thr.data.to("cpu")
                zero_mask_t = zero_mask_t.to("cpu")  # type: ignore[assignment]
        except Exception:
            pass
        # Fall through to final_entry handling below (original code will handle)
        # Prevent duplicate step loop — we have already handled steps, so skip original CPU steps by breaking?
        # Instead we set a flag to indicate steps already done
        _steps_already_done = _use_gpu and _module_done_on_gpu
        if _steps_already_done:
            # Need to still create final_entry and release, but skip re-running steps
            # We will jump to final_entry section by not re-entering loop
            # To avoid duplicate, we set step_start = steps_per_module to skip outer?
            pass
        # The original code after this block continues to final_entry creation, which we keep
        # But we have duplicated the step loop; we need to avoid executing the original CPU loop twice
        # So we will conditionally skip the original steps if GPU succeeded
        if _use_gpu and _module_done_on_gpu:
            # final_entry will be created from current raw_param (already on cpu after move)
            final_entry: dict[str, typing.Any] = {
                "raw_param": raw_param.detach().cpu(),
                "orig_scales": orig_scales.detach().cpu(),
                "reference_scales": reference_scales.detach().cpu(),
                "codes": codes.detach().cpu(),
                "weight_shape": tuple(w.shape),
                "weight_dtype": str(w.dtype),
                "zero_mask": zero_mask.detach().cpu(),
                "zero_mask_t": zero_mask_t.detach().cpu(),
            }
            if threshold_enabled and raw_thr is not None and thr_zero_mask is not None:
                final_entry["raw_threshold"] = raw_thr.detach().cpu()
                final_entry["thr_zero_mask"] = thr_zero_mask.detach().cpu()
            completed_module_params[mname] = final_entry
            del w, w_f32, raw_param
            if raw_thr is not None:
                del raw_thr
            del optimizer
            _gc.collect()
            try:
                if torch.cuda.is_available():  # type: ignore[union-attr]
                    torch.cuda.empty_cache()  # type: ignore[union-attr]
                    print(f"[vram] module released {mname}", flush=True)
            except Exception:
                pass
            start_step_in_module = 0
            _resume_opt_state = None
            continue  # next module, skip the rest of original flow that would duplicate

        # After module steps, finalize and store completed params (bounded release preparation)
        final_entry = {
            "raw_param": raw_param.detach().cpu(),
            "orig_scales": orig_scales.detach().cpu(),
            "reference_scales": reference_scales.detach().cpu(),
            "codes": codes.detach().cpu(),
            "weight_shape": tuple(w.shape),
            "weight_dtype": str(w.dtype),
            "zero_mask": zero_mask.detach().cpu(),
            "zero_mask_t": zero_mask_t.detach().cpu(),
        }
        if threshold_enabled and raw_thr is not None and thr_zero_mask is not None:
            final_entry["raw_threshold"] = raw_thr.detach().cpu()
            final_entry["thr_zero_mask"] = thr_zero_mask.detach().cpu()
        completed_module_params[mname] = final_entry

        # Ensure final checkpoint for this module is saved (if not already saved at final step)
        # Already saved at final step above, but ensure completed dict is persisted
        # Save intermediate checkpoint per module (as required) with is_module_done=True
        # The last iteration already saved with is_module_done, but we ensure a module checkpoint exists
        # Release weight and optimizer (bounded memory)
        del w, w_f32, raw_param
        if raw_thr is not None:
            del raw_thr
        del optimizer
        _gc.collect()
        try:
            if torch.cuda.is_available():  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass
        # Reset resume offset after first resume module
        start_step_in_module = 0
        _resume_opt_state = None

    # After sequential loop, reconstruct per_module dict for downstream metrics/materialize
    per_module: dict[str, dict[str, typing.Any]] = {}
    for mname, entry in completed_module_params.items():
        # Resume-loaded entries may only have raw_param/raw_threshold (P0 manifest). Recompute missing fields from weight.
        if "orig_scales" not in entry or "zero_mask_t" not in entry:
            w_tmp = _load_weight(mname)
            if w_tmp is None:
                print(f"[per_module] warning: skip {mname} weight not found for recompute", flush=True)
                continue
            w_f32_tmp = w_tmp.to(torch.float32)
            if scale_granularity == "per_tensor":
                tt_tmp = quantize_absmean(w_f32_tmp)
                orig_scales_tmp = torch.tensor([tt_tmp.scale], dtype=torch.float32)
                codes_tmp = tt_tmp.codes
                zero_mask_tmp = orig_scales_tmp == 0
            else:
                res_tmp = quantize_groupwise(w_f32_tmp, group_size)
                orig_scales_tmp = res_tmp.scales.clone()
                codes_tmp = res_tmp.codes
                zero_mask_tmp = orig_scales_tmp == 0
            _, zero_mask_t_tmp = build_scale_params(orig_scales_tmp, zero_mask_tmp)
            entry["orig_scales"] = orig_scales_tmp
            entry["reference_scales"] = orig_scales_tmp.clone()
            entry["codes"] = codes_tmp
            entry["zero_mask"] = zero_mask_tmp
            entry["zero_mask_t"] = zero_mask_t_tmp
            entry["weight_shape"] = tuple(w_tmp.shape)
            entry["weight_dtype"] = str(w_tmp.dtype)
            if threshold_enabled and "thr_zero_mask" not in entry:
                entry["thr_zero_mask"] = zero_mask_tmp
        # Rebuild tensors to original device (cpu) — per_module expects same structure as before
        # Need to recreate raw_param as Parameter? For metrics we only need raw_param tensor
        # Create Parameter wrappers for compatibility with downstream code that uses get_effective
        raw_p = torch.nn.Parameter(entry["raw_param"])  # type: ignore[arg-type]
        zm_t = entry["zero_mask_t"]
        # Ensure zero_mask_t is bool tensor
        per_module[mname] = {
            "codes": entry["codes"],
            "orig_scales": entry["orig_scales"],
            "reference_scales": entry["reference_scales"],
            "zero_mask": entry["zero_mask"],
            "zero_mask_t": zm_t,
            "raw_param": raw_p,
            "weight_shape": entry["weight_shape"],
            "weight_dtype": entry["weight_dtype"],
        }
        if threshold_enabled and "raw_threshold" in entry:
            raw_thr_p = torch.nn.Parameter(entry["raw_threshold"])  # type: ignore[arg-type]
            per_module[mname]["raw_threshold"] = raw_thr_p
            per_module[mname]["thr_zero_mask"] = entry.get("thr_zero_mask", entry["zero_mask"])

    if not per_module:
        raise ValueError("no quantizable weights found for calibration")

    final_loss = loss_history[-1] if loss_history else 0.0

    # Held-out loss (via held_cache) — threshold aware
    reconstruction_by_module: dict[str, dict[str, float | int]] = {}
    for mname, state in per_module.items():
        codes_fixed = state["codes"]
        orig_scales = state["orig_scales"]
        reference_scales = state["reference_scales"]
        zero_mask_t = state["zero_mask_t"]
        raw_param = state["raw_param"]
        w_shape = state["weight_shape"]
        eff_final = _get_eff(raw_param, zero_mask_t)
        # Load weight for threshold on demand (bounded)
        weight_tensor_thr: torch.Tensor | None = None  # type: ignore[type-arg]
        if threshold_enabled or soft_to_hard_enabled:
            wt_h = _load_weight(mname)
            if wt_h is not None:
                weight_tensor_thr = wt_h.to(torch.float32)
        # final w_hat
        if threshold_enabled:
            assert weight_tensor_thr is not None
            thr_final = get_effective_threshold_ratio(state["raw_threshold"], eps=threshold_eps)  # type: ignore[arg-type]
            thr_init = torch.full_like(thr_final, 0.5)
            from openternary.quant.threshold import _expand_per_group as _exp2
            from openternary.quant.threshold import hard_threshold_codes as _htc

            if scale_granularity == "per_tensor":
                codes_final = _htc(weight_tensor_thr, float(reference_scales[0].item()), float(thr_final[0].item()))
                codes_init = _htc(weight_tensor_thr, float(reference_scales[0].item()), 0.5)
                w_hat_final = codes_final.to(torch.float32) * eff_final[0]
                w_hat_init = codes_init.to(torch.float32) * orig_scales[0]
            else:
                codes_final = _htc(weight_tensor_thr, reference_scales, thr_final, group_size=group_size)
                codes_init = _htc(weight_tensor_thr, reference_scales, thr_init, group_size=group_size)
                eff_exp_final = _exp2(eff_final, tuple(weight_tensor_thr.shape), group_size)
                eff_exp_init = _exp2(orig_scales, tuple(weight_tensor_thr.shape), group_size)
                w_hat_final = codes_final.to(torch.float32) * eff_exp_final
                w_hat_init = codes_init.to(torch.float32) * eff_exp_init
        elif soft_to_hard_enabled:
            assert weight_tensor_thr is not None
            codes_final = _hard_soft_codes(weight_tensor_thr, reference_scales)
            codes_init = _hard_soft_codes(weight_tensor_thr, reference_scales)
            if scale_granularity == "per_tensor":
                w_hat_final = codes_final.to(torch.float32) * eff_final[0]
                w_hat_init = codes_init.to(torch.float32) * orig_scales[0]
            else:
                from openternary.quant.threshold import _expand_per_group as _exp2

                w_hat_final = codes_final.to(torch.float32) * _exp2(
                    eff_final, tuple(weight_tensor_thr.shape), group_size
                )
                w_hat_init = codes_init.to(torch.float32) * _exp2(
                    orig_scales, tuple(weight_tensor_thr.shape), group_size
                )
        else:
            if scale_granularity == "per_tensor":
                w_hat_final = codes_fixed.to(torch.float32) * eff_final[0]
                w_hat_init = codes_fixed.to(torch.float32) * orig_scales[0]
            else:
                w_hat_final = _degw(
                    _GWRes(
                        codes=codes_fixed,
                        scales=eff_final,
                        shape=w_shape,
                        orig_dtype=state["weight_dtype"],
                        group_size=group_size,
                        grouping_scheme="last-dim-rowwise-v1",
                    )
                )
                w_hat_init = _degw(
                    _GWRes(
                        codes=codes_fixed,
                        scales=orig_scales,
                        shape=w_shape,
                        orig_dtype=state["weight_dtype"],
                        group_size=group_size,
                        grouping_scheme="last-dim-rowwise-v1",
                    )
                )
        module_metrics: dict[str, float | int] = {}
        with torch.no_grad():
            for split, cache in (("train", train_cache), ("heldout", held_cache)):
                eval_device = "cuda:0" if backend_info.backend != "cpu" and torch.cuda.is_available() else "cpu"

                def evaluate_weight(
                    weight_hat: torch.Tensor,
                    cache_obj: typing.Any,
                    module_name: str,
                    device: str,
                ) -> tuple[float, int]:
                    weight_device = weight_hat.float().to(device)
                    loss_sum = 0.0
                    element_count = 0
                    try:
                        for b_idx in range(cache_obj.count_batches(module_name)):
                            data = cache_obj.load_valid(module_name, b_idx)
                            inp = data["input"].float().to(device)
                            tout = data["teacher_output"].float().to(device)
                            count = tout.numel()
                            prediction = F.linear(inp, weight_device)  # type: ignore[union-attr]
                            loss_sum += float(_mse(tout, prediction).item()) * count
                            element_count += count
                    finally:
                        del weight_device
                        if device != "cpu":
                            with contextlib.suppress(Exception):
                                torch.cuda.empty_cache()
                    return loss_sum, element_count

                after_sum, elements = evaluate_weight(w_hat_final, cache, mname, eval_device)
                before_sum, before_elements = evaluate_weight(w_hat_init, cache, mname, eval_device)
                if before_elements != elements:
                    raise ValueError(f"reconstruction element mismatch for {mname}/{split}")
                if not elements:
                    raise ValueError(f"no valid {split} elements for {mname}")
                module_metrics[f"{split}_elements"] = elements
                module_metrics[f"{split}_before"] = before_sum / elements
                module_metrics[f"{split}_after"] = after_sum / elements
        reconstruction_by_module[mname] = module_metrics

    def weighted_metric(split: str, when: str) -> float:
        return sum(
            item[f"{split}_{when}"] * item[f"{split}_elements"] for item in reconstruction_by_module.values()
        ) / sum(item[f"{split}_elements"] for item in reconstruction_by_module.values())

    initial_loss = weighted_metric("train", "before")
    final_loss = weighted_metric("train", "after")
    held_loss_before_val = weighted_metric("heldout", "before")
    held_loss_after_val = weighted_metric("heldout", "after")

    # Aggregate scale fingerprints (all modules)
    all_orig = torch.cat([s["orig_scales"].view(-1) for s in per_module.values()]) if per_module else torch.tensor([])
    all_final = (
        torch.cat([_get_eff(s["raw_param"], s["zero_mask_t"]).view(-1) for s in per_module.values()])
        if per_module
        else torch.tensor([])
    )
    # threshold fingerprints
    all_thr_orig = None
    all_thr_final = None
    if threshold_enabled:
        all_thr_orig = (
            torch.cat(
                [
                    torch.full_like(get_effective_threshold_ratio(s["raw_threshold"], eps=threshold_eps), 0.5)
                    for s in per_module.values()
                ]
            )
            if per_module
            else torch.tensor([])
        )  # dummy, actual before is 0.5
        # real before is 0.5 for all groups
        # we compute fingerprint for threshold before as 0.5 tensor, after as actual
        all_thr_final = (
            torch.cat(
                [
                    get_effective_threshold_ratio(s["raw_threshold"], eps=threshold_eps).view(-1)
                    for s in per_module.values()
                ]
            )
            if per_module
            else torch.tensor([])
        )
        # fix all_thr_orig to be 0.5 tensor of same shape as final
        all_thr_orig = torch.full_like(all_thr_final, 0.5) if all_thr_final.numel() else torch.tensor([])
    # zero exact check across all
    zero_exact = True
    for s in per_module.values():
        eff = _get_eff(s["raw_param"], s["zero_mask_t"])
        zm = s["zero_mask"]
        if zm.any().item() and not (eff[zm] == 0).all().item():
            zero_exact = False
            break

    # Materialize calibrated snapshot (bounded sharded — hard codes via threshold, weight loaded on demand)
    calibrated_state: dict[str, typing.Any] = {}
    for mname, state in per_module.items():
        eff = _get_eff(state["raw_param"], state["zero_mask_t"])
        weight_name = mname + ".weight"
        if threshold_enabled:
            thr_ratio_final = get_effective_threshold_ratio(state["raw_threshold"], eps=threshold_eps)  # type: ignore[arg-type]
            from openternary.quant.threshold import hard_threshold_codes as _htc2

            wt_mat = _load_weight(mname)
            if wt_mat is None:
                raise FileNotFoundError(f"weight for {mname} not found during materialize")
            wt_f32 = wt_mat.to(torch.float32)
            if scale_granularity == "per_tensor":
                codes_final = _htc2(
                    wt_f32, float(state["reference_scales"][0].item()), float(thr_ratio_final[0].item())
                )
            else:
                codes_final = _htc2(wt_f32, state["reference_scales"], thr_ratio_final, group_size=group_size)
        elif soft_to_hard_enabled:
            wt_mat = _load_weight(mname)
            if wt_mat is None:
                raise FileNotFoundError(f"weight for {mname} not found during materialize")
            codes_final = _hard_soft_codes(wt_mat.to(torch.float32), state["reference_scales"])
        else:
            codes_final = state["codes"]
        calibrated_state[weight_name] = {
            "codes": codes_final,
            "scales": eff,
            "shape": state["weight_shape"],
            "orig_dtype": state["weight_dtype"],
            "group_size": group_size,
            "grouping_scheme": "last-dim-rowwise-v1",
        }
    # Use Phase 3 bounded writer (reuse) — stream source shard-by-shard, dequantize via existing helpers
    from openternary.quant.fake_quant import materialize_calibrated_snapshot

    calibrated_snapshot = out / "artifacts" / "calibrated_snapshot"
    # Fast gate: skip heavy materialize (hashing 1951 tensors) when OT_SKIP_MATERIALIZE=1
    import os as _os_skip

    _report: typing.Any
    if _os_skip.environ.get("OT_SKIP_MATERIALIZE") == "1":
        # Create minimal snapshot dir for gate metrics (no heavy hashing)
        calibrated_snapshot.mkdir(parents=True, exist_ok=True)
        (calibrated_snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": {}}), encoding="utf-8"
        )
        # Dummy report for calib_json
        from types import SimpleNamespace as _NS

        _report = _NS(
            content_fingerprint="skip-materialize",
            shard_count=0,
            tensor_payload_bytes=0,
        )
        print("[materialize] skipped via OT_SKIP_MATERIALIZE=1", flush=True)
    else:
        # materialize は tmp 経由で atomic に作成するため、既存の空 dir は削除
        if calibrated_snapshot.exists():
            import shutil as _sh

            _sh.rmtree(calibrated_snapshot, ignore_errors=True)
        _report = materialize_calibrated_snapshot(
            src_snapshot=pathlib.Path(teacher_snapshot),  # type: ignore[arg-type]
            dst_snapshot=calibrated_snapshot,
            app_config=app_config,
            calibrated_state=calibrated_state,
        )
        # sanity: placeholder のみは禁止
        if not (calibrated_snapshot / "model.safetensors.index.json").exists():
            raise RuntimeError("calibrated_snapshot materialization failed: missing index.json")

    # Threshold / code metrics
    thr_fingerprint_before = ""
    thr_fingerprint_after = ""
    thr_mean = thr_min = thr_max = thr_std = 0.0
    code_fp_before = ""
    code_fp_after = ""
    code_change_ratio = 0.0
    zero_before = zero_after = 0.0
    pos_before = pos_after = neg_before = neg_after = 0.0
    if threshold_enabled and all_thr_final is not None and all_thr_final.numel():
        thr_fingerprint_before = (
            hashlib.sha256(all_thr_orig.numpy().tobytes()).hexdigest()
            if all_thr_orig is not None and all_thr_orig.numel()
            else ""
        )
        thr_fingerprint_after = hashlib.sha256(all_thr_final.detach().cpu().numpy().tobytes()).hexdigest()
        thr_mean = float(all_thr_final.mean().item())
        thr_min = float(all_thr_final.min().item())
        thr_max = float(all_thr_final.max().item())
        thr_std = float(all_thr_final.float().std(correction=0).item()) if all_thr_final.numel() > 1 else 0.0
        # code fingerprints across all modules
        from openternary.quant.threshold import hard_threshold_codes as _htc3

        all_codes_before: list[torch.Tensor] = []  # type: ignore[type-arg]
        all_codes_after: list[torch.Tensor] = []  # type: ignore[type-arg]
        for _mname, state in per_module.items():
            ref = state["reference_scales"]
            thr_f = get_effective_threshold_ratio(state["raw_threshold"], eps=threshold_eps)  # type: ignore[arg-type]
            thr_0 = torch.full_like(thr_f, 0.5)
            wt_load = _load_weight(_mname)
            if wt_load is None:
                continue
            wt = wt_load.to(torch.float32)
            if scale_granularity == "per_tensor":
                cb = _htc3(wt, float(ref[0].item()), 0.5)
                ca = _htc3(wt, float(ref[0].item()), float(thr_f[0].item()))
            else:
                cb = _htc3(wt, ref, thr_0, group_size=group_size)
                ca = _htc3(wt, ref, thr_f, group_size=group_size)
            all_codes_before.append(cb.view(-1))
            all_codes_after.append(ca.view(-1))
        cat_before = torch.cat(all_codes_before) if all_codes_before else torch.tensor([])
        cat_after = torch.cat(all_codes_after) if all_codes_after else torch.tensor([])
        if cat_before.numel():
            code_fp_before = hashlib.sha256(cat_before.numpy().tobytes()).hexdigest()
            code_fp_after = hashlib.sha256(cat_after.numpy().tobytes()).hexdigest()
            code_change_ratio = float((cat_before != cat_after).float().mean().item())
            zero_before = float((cat_before == 0).float().mean().item())
            zero_after = float((cat_after == 0).float().mean().item())
            pos_before = float((cat_before == 1).float().mean().item())
            pos_after = float((cat_after == 1).float().mean().item())
            neg_before = float((cat_before == -1).float().mean().item())
            neg_after = float((cat_after == -1).float().mean().item())
    else:
        # scale-only: code fingerprints based on fixed codes
        all_codes = torch.cat([s["codes"].view(-1) for s in per_module.values()]) if per_module else torch.tensor([])
        if all_codes.numel():
            code_fp_before = hashlib.sha256(all_codes.numpy().tobytes()).hexdigest()
            code_fp_after = code_fp_before
            zero_before = zero_after = float((all_codes == 0).float().mean().item())
            pos_before = pos_after = float((all_codes == 1).float().mean().item())
            neg_before = neg_after = float((all_codes == -1).float().mean().item())

    calib_json = {
        "method": str(calib_cfg.method),
        "metrics_version": "masked-element-weighted-v2",
        "loss_kind": str(calib_cfg.loss),
        "optimizer": str(calib_cfg.optimizer),
        "reconstruction_reference": "naive ternary, not warm-start initialization",
        "reconstruction_by_module": reconstruction_by_module,
        "loss_history_layout": "module-major; optimizer observations, not a model improvement curve",
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
        "best_loss": None,
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
        "contamination_train_smoke": contamination_train_smoke,
        "contamination_held_smoke": contamination_held_smoke,
        "contamination_train_held": contamination_train_held,
        "scale_fingerprint_before": hashlib.sha256(all_orig.numpy().tobytes()).hexdigest() if all_orig.numel() else "",
        "scale_fingerprint_after": hashlib.sha256(all_final.detach().cpu().numpy().tobytes()).hexdigest()
        if all_final.numel()
        else "",
        "mean_abs_scale_delta": float((all_final - all_orig).abs().mean().item()) if all_final.numel() else 0.0,
        "zero_group_exact": bool(zero_exact),
        "trainable_param_count": int(
            sum(int(s["raw_param"].numel() - int(s["zero_mask"].sum().item())) for s in per_module.values())
        )
        + (
            sum(int(s["raw_threshold"].numel() - int(s["thr_zero_mask"].sum().item())) for s in per_module.values())
            if threshold_enabled
            else 0
        ),
        "total_param_count": int(sum(int(s["raw_param"].numel()) for s in per_module.values()))
        + (sum(int(s["raw_threshold"].numel()) for s in per_module.values()) if threshold_enabled else 0),
        "used_dummy_simulation": False,
        "teacher_snapshot": str(teacher_snapshot),
        "calibrated_snapshot": str(calibrated_snapshot),
        "calibrated_content_fingerprint": _report.content_fingerprint,
        "calibrated_shard_count": int(_report.shard_count),
        "calibrated_tensor_payload_bytes": int(_report.tensor_payload_bytes),
        "threshold_enabled": threshold_enabled,
        "assignment": assignment_contract,
        "threshold_fingerprint_before": thr_fingerprint_before,
        "threshold_fingerprint_after": thr_fingerprint_after,
        "threshold_ratio_mean": thr_mean,
        "threshold_ratio_min": thr_min,
        "threshold_ratio_max": thr_max,
        "threshold_ratio_std": thr_std,
        "threshold_eps": threshold_eps,
        "threshold_ste_width": threshold_ste_width,
        "code_fingerprint_before": code_fp_before,
        "code_fingerprint_after": code_fp_after,
        "code_change_ratio": code_change_ratio,
        "zero_ratio_before": zero_before,
        "zero_ratio_after": zero_after,
        "positive_ratio_before": pos_before,
        "positive_ratio_after": pos_after,
        "negative_ratio_before": neg_before,
        "negative_ratio_after": neg_after,
        "backend": backend_info.backend,
        "device_name": backend_info.device_name,
        "peak_vram_bytes": int(getattr(run_calibration, "_peak_vram", 0)),
        "peak_vram_human": format_bytes(int(getattr(run_calibration, "_peak_vram", 0))),
        "vram_budget_bytes": int(_vram_budget.budget_bytes)
        if "_vram_budget" in locals() and _vram_budget is not None
        else 0,
        "low_vram": bool(_low_vram),
        "effective_runtime": {
            "microbatch_auto": bool(_micro_auto),
            "microbatch_min_size": int(_micro_min),
            "initial_microbatch": int(initial_microbatch_max),
            "minimum_microbatch_used": int(minimum_microbatch_used or 0),
            "oom_retries": int(total_oom_retries),
            "modules": runtime_module_records,
        },
    }
    with open(out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_json, f, indent=2, ensure_ascii=False)

    # Collapse detection (warning only, not failure)
    if threshold_enabled:
        if zero_after > 0.995:
            print(f"[warning] collapse: zero_ratio {zero_after:.4f} > 0.995", flush=True)
        if pos_after > 0.995:
            print(f"[warning] collapse: positive_ratio {pos_after:.4f} > 0.995", flush=True)
        if neg_after > 0.995:
            print(f"[warning] collapse: negative_ratio {neg_after:.4f} > 0.995", flush=True)
        # threshold near eps or 1-eps for all groups
        if all_thr_final is not None and all_thr_final.numel():
            near_eps = (all_thr_final < threshold_eps * 1.5).float().mean().item()
            near_one = (all_thr_final > 1 - threshold_eps * 1.5).float().mean().item()
            if near_eps > 0.995:
                print(
                    f"[warning] collapse: threshold_ratio near eps {threshold_eps} for {near_eps * 100:.1f}% groups",
                    flush=True,
                )
            if near_one > 0.995:
                print(f"[warning] collapse: threshold_ratio near 1-eps for {near_one * 100:.1f}% groups", flush=True)

    # Per-module threshold metrics artifact
    try:
        thr_metrics_path = out / "artifacts" / "threshold_metrics.jsonl"
        thr_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(thr_metrics_path, "w", encoding="utf-8") as mf:
            for mname, state in per_module.items():
                # compute per-module thr stats
                if threshold_enabled and "raw_threshold" in state:
                    thr_f = get_effective_threshold_ratio(state["raw_threshold"], eps=threshold_eps).detach().cpu()
                    # codes before/after for this module
                    wt_m = _load_weight(mname)
                    if wt_m is not None:
                        wt_f = wt_m.to(torch.float32)
                        ref = state["reference_scales"]
                        thr_0 = torch.full_like(thr_f, 0.5)
                        from openternary.quant.threshold import hard_threshold_codes as _htc_mod

                        if scale_granularity == "per_tensor":
                            cb_m = _htc_mod(wt_f, float(ref[0].item()), 0.5)
                            ca_m = _htc_mod(wt_f, float(ref[0].item()), float(thr_f[0].item()))
                        else:
                            cb_m = _htc_mod(wt_f, ref, thr_0, group_size=group_size)
                            ca_m = _htc_mod(wt_f, ref, thr_f, group_size=group_size)
                        mod_code_change = float((cb_m != ca_m).float().mean().item()) if cb_m.numel() else 0.0
                        mod_zero_before = float((cb_m == 0).float().mean().item()) if cb_m.numel() else 0.0
                        mod_zero_after = float((ca_m == 0).float().mean().item()) if ca_m.numel() else 0.0
                    else:
                        mod_code_change = 0.0
                        mod_zero_before = mod_zero_after = 0.0
                    mf.write(
                        json.dumps(
                            {
                                "module": mname,
                                "threshold_before": 0.5,
                                "threshold_after_mean": float(thr_f.mean().item()),
                                "threshold_after_min": float(thr_f.min().item()),
                                "threshold_after_max": float(thr_f.max().item()),
                                "code_change_ratio": mod_code_change,
                                "zero_ratio_before": mod_zero_before,
                                "zero_ratio_after": mod_zero_after,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    except Exception as e:
        print(f"[warning] failed to write threshold_metrics.jsonl: {e}", flush=True)

    metrics = {
        "status": "completed",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": None,
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
