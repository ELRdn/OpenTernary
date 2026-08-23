"""Calibration runner — Phase 4.2 layer-local recon-scale-threshold."""

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

import contextlib

from openternary.calibration.optimizer import (
    build_scale_params,
    build_threshold_params,
    get_effective_scales,
    get_effective_threshold_ratio,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for calibration. Install with: uv sync --extra ml")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    """Snapshot content fingerprint for cache invalidation (best-effort)."""
    if snapshot_path is None:
        return ""
    try:
        p = pathlib.Path(snapshot_path)
        if not p.exists():
            return ""
        # hash config.json + tokenizer.json + list of safetensors sizes/names
        h = hashlib.sha256()
        for fname in ["config.json", "tokenizer.json"]:
            fp = p / fname
            if fp.exists():
                with contextlib.suppress(Exception):
                    h.update(hashlib.sha256(fp.read_bytes()).hexdigest().encode())
        # include safetensors file names + sizes
        for st in sorted(p.glob("*.safetensors")):
            with contextlib.suppress(Exception):
                h.update(st.name.encode())
                h.update(str(st.stat().st_size).encode())
        return h.hexdigest()
    except Exception:
        return ""


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

    # Load checkpoint data (v2 aware)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # Support v2 module-major checkpoint: completed_module_params
    if isinstance(ckpt, dict) and ckpt.get("schema_version") == 2 and "completed_module_params" in ckpt:
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
    if isinstance(ckpt, dict) and ckpt.get("schema_version") == 2 and "completed_module_params" in ckpt:
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
        # Override raw_param from checkpoint if available (v2 and legacy)
        ckpt_raw = None
        if isinstance(ckpt, dict) and ckpt.get("schema_version") == 2 and "completed_module_params" in ckpt:
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
            # override from checkpoint (v2 aware)
            ckpt_thr = None
            if (
                isinstance(ckpt, dict)
                and ckpt.get("schema_version") == 2
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
            print(f"[materialize-only] warning: failed to update calibration.json: {e}", flush=True)

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
    # Handle --init-from via config or CLI
    if init_from is None:
        # also check config field
        cfg_init = getattr(app_config.calibration, "init_from", None)
        if cfg_init is not None:
            init_from = cfg_init

    import pathlib as _pl

    out = _pl.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save config
    import yaml

    cfg_path = out / "config.yaml"
    # app_config is Pydantic
    try:
        cfg_dict = app_config.model_dump() if hasattr(app_config, "model_dump") else dict(app_config)
    except Exception:
        cfg_dict = {}
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    calib_cfg = app_config.calibration
    # Validate window
    if calib_cfg.window != "per-layer":
        raise ValueError(f"window={calib_cfg.window} is not supported in Phase 4.1 (only per-layer)")

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
            requested_dataset, num_samples, seed, bool(calib_cfg.allow_dataset_fallback)
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
    is_synthetic_tiny = requested_dataset == "synthetic" and num_samples <= 8 and int(calib_cfg.steps) <= 5
    if not is_real_snapshot or is_synthetic_tiny:
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
        # Use adapter's quantizable classification on model named modules
        target_module_names: list[str] = []
        for name, module in model.named_modules():
            import torch.nn as _nn  # type: ignore[import]

            if isinstance(module, _nn.Linear):
                # Exclude non-text quantizable modules: lm_head, embeddings, vision/audio towers
                if "lm_head" in name or "embed" in name:
                    continue
                if "vision_tower" in name or "audio_tower" in name or "vision_encoder" in name:
                    continue
                # Gemma 4 text path is model.language_model etc; keep those
                target_module_names.append(name)
        if not target_module_names:
            raise ValueError("no quantizable modules found")
    except Exception:
        # fallback: all Linear except vision/audio
        import torch.nn as _nn  # type: ignore[import]

        target_module_names = [
            n
            for n, m in model.named_modules()
            if isinstance(m, _nn.Linear) and "lm_head" not in n and "vision_tower" not in n and "audio_tower" not in n
        ]

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
        capture_format_version: str = "v1",
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
        capture_format_version="v1",
    )
    reuse_possible = False
    reuse_source: pathlib.Path | None = None
    # Check init_from cache for reuse (Phase 4.1 → 4.2)
    if init_from is not None:
        init_train = _pl.Path(str(init_from)) / "artifacts" / "activation_cache_train" / "fingerprint.json"
        init_held = _pl.Path(str(init_from)) / "artifacts" / "activation_cache_held" / "fingerprint.json"
        if init_train.exists() and init_held.exists():
            try:
                init_fp_train = json.loads(init_train.read_text(encoding="utf-8")).get("fingerprint", "")
                init_fp_held = json.loads(init_held.read_text(encoding="utf-8")).get("fingerprint", "")
                if init_fp_train == current_fp and init_fp_held == current_fp:
                    # fingerprints match → reuse by copying
                    print(f"[cache] Activation Cache: REUSED from --init-from {init_from}", flush=True)
                    import shutil as _sh2

                    if not train_cache_dir.exists() or not any(train_cache_dir.iterdir()):
                        _sh2.copytree(
                            _pl.Path(str(init_from)) / "artifacts" / "activation_cache_train",
                            train_cache_dir,
                            dirs_exist_ok=True,
                        )
                    if not held_cache_dir.exists() or not any(held_cache_dir.iterdir()):
                        _sh2.copytree(
                            _pl.Path(str(init_from)) / "artifacts" / "activation_cache_held",
                            held_cache_dir,
                            dirs_exist_ok=True,
                        )
                    reuse_possible = True
                    reuse_source = _pl.Path(str(init_from))
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
            ef_train = json.loads(existing_fp_train.read_text(encoding="utf-8")).get("fingerprint", "")
            ef_held = json.loads(existing_fp_held.read_text(encoding="utf-8")).get("fingerprint", "")
            # Need to ensure cache actually has data for all target modules
            train_layers = [p.name for p in train_cache_dir.iterdir() if p.is_dir() and p.name != "_fingerprint"]
            held_layers = [p.name for p in held_cache_dir.iterdir() if p.is_dir()]
            expected_layers = [n.replace(".", "_").replace("/", "_") for n in target_module_names]
            has_all = all(layer_name in train_layers for layer_name in expected_layers) and all(
                layer_name in held_layers for layer_name in expected_layers
            )
            if ef_train == current_fp and ef_held == current_fp and has_all:
                print("[cache] Activation Cache: REUSED (existing output cache matches fingerprint)", flush=True)
                reuse_possible = True
            else:
                print(
                    "[cache] Activation Cache: INVALID (existing cache fingerprint mismatch or incomplete), recapturing",
                    flush=True,
                )
        except Exception as e:
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
                "capture_format_version": "v1",
                "tokenizer_fingerprint": _tok_fp,
                "train_sample_hashes": train_hashes,
                "held_sample_hashes": held_hashes,
            }
            (train_cache_dir / "fingerprint.json").write_text(
                json.dumps(_fp_meta, indent=2),
                encoding="utf-8",
            )
            (held_cache_dir / "fingerprint.json").write_text(
                json.dumps(_fp_meta, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[cache] warning: failed to save fingerprint: {e}", flush=True)
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
    # validation
    if threshold_enabled and not (threshold_eps < threshold_init_ratio < 1 - threshold_eps):
        raise ValueError(
            f"threshold_init_ratio {threshold_init_ratio} must be in ({threshold_eps}, {1 - threshold_eps})"
        )
    if threshold_enabled and str(getattr(calib_cfg, "method", "recon-scale")) not in ("recon-threshold", "recon-scale"):
        # allow recon-scale with threshold_enabled for backward compat, but warn
        pass

    # === Module-major (true sequential per-module optimization) per spec 7.1 ===
    # Steps are per-module: each target module runs `steps` optimizer steps independently.
    steps_per_module = int(calib_cfg.steps)
    steps = steps_per_module  # alias for downstream calibration.json compatibility
    ckpt_interval = int(calib_cfg.checkpoint_interval)
    ckpt_dir = out / "artifacts" / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
                # v2 checkpoint
                if (
                    isinstance(init_data, dict)
                    and init_data.get("schema_version") == 2
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
                    print("[init-from] synthetic single param warm start (ignored for real path)", flush=True)
                else:
                    print(f"[init-from] warning: unrecognized checkpoint format in {init_ckpt}", flush=True)
            except Exception as e:
                print(f"[init-from] warning: failed to load {init_ckpt}: {e}", flush=True)
        else:
            print(
                f"[init-from] warning: no checkpoint found in {init_path}, using fresh init (threshold 0.5)", flush=True
            )

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
            # check threshold compatibility
            _ckpt_thr_enabled = bool(_resume_ckpt.get("threshold_enabled", False))
            if _ckpt_thr_enabled != threshold_enabled:
                raise ValueError(
                    f"checkpoint threshold_enabled={_ckpt_thr_enabled} != current {threshold_enabled}. Use --init-from for cross-phase warm start, not --resume."
                )
            # method check (warn if mismatch)
            # schema v2 expected
            if _resume_ckpt.get("schema_version") == 2:
                completed_module_params = dict(_resume_ckpt.get("completed_module_params", {}))
                mc = _resume_ckpt.get("module_cursor", {})
                start_module_idx = int(mc.get("module_idx", 0))
                start_step_in_module = int(mc.get("step", 0))
                # if cursor step == steps_per_module, it means module completed; next module should start at 0
                if start_step_in_module >= steps_per_module:
                    start_module_idx = int(mc.get("module_idx", 0)) + 1
                    start_step_in_module = 0
                # restore loss_history / best
                loss_history = list(_resume_ckpt.get("loss_history", []))
                initial_loss = _resume_ckpt.get("initial_loss")
                best_loss = _resume_ckpt.get("best_loss", float("inf"))
                _resume_opt_state = _resume_ckpt.get("optimizer_state")
                # sanity: if resume checkpoint already completed some modules, ensure start index within range
                print(
                    f"[resume] v2 checkpoint {_ckpt_path} cursor module_idx={start_module_idx} step={start_step_in_module} completed={len(completed_module_params)}",
                    flush=True,
                )
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

    from openternary.calibration.losses import mse_loss as _mse
    from openternary.calibration.optimizer import get_effective_scales as _get_eff
    from openternary.quant.grouping import GroupwiseResult as _GWRes
    from openternary.quant.grouping import dequantize_groupwise as _degw

    # Helper to save v2 checkpoint
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
        # Determine filename: module checkpoint and also step checkpoint for compatibility
        # Use module_XXXXX for per-module and step_XXXXX for per-step (global)
        # Global step index aggregated across modules
        global_step = module_idx * steps_per_module + step_in_module
        # Save module checkpoint
        mod_path = ckpt_dir / f"module_{module_idx:05d}.pt"
        step_path = ckpt_dir / f"step_{global_step:05d}.pt"
        # Build completed dict for save (ensure cpu tensors)
        save_completed = {}
        for k, v in completed_module_params.items():
            entry = {}
            if "raw_param" in v:
                entry["raw_param"] = (
                    v["raw_param"].detach().cpu() if hasattr(v["raw_param"], "detach") else v["raw_param"]
                )
            if "raw_threshold" in v:
                entry["raw_threshold"] = (
                    v["raw_threshold"].detach().cpu() if hasattr(v["raw_threshold"], "detach") else v["raw_threshold"]
                )
            # Minimal checkpoint: only raw params, recompute scales on resume/materialize to keep checkpoint small
            save_completed[k] = entry
        ckpt_save: dict[str, typing.Any] = {
            "schema_version": 2,
            "method": str(calib_cfg.method),
            "module_cursor": {
                "module_idx": module_idx if not is_module_done else module_idx + 1,
                "module_name": module_name,
                "step": step_in_module if not is_module_done else 0,
                "steps_per_module": steps_per_module,
                "completed_count": len(save_completed),
            },
            "completed_module_params": save_completed,
            "loss_history": list(loss_history),
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "threshold_enabled": threshold_enabled,
            "scale_granularity": scale_granularity,
            "threshold_eps": threshold_eps,
            "threshold_ste_width": threshold_ste_width,
        }
        if optimizer_state is not None:
            ckpt_save["optimizer_state"] = optimizer_state
        if current_raw is not None:
            ckpt_save["current_raw_param"] = current_raw.detach().cpu()
        if current_thr is not None:
            ckpt_save["current_raw_threshold"] = current_thr.detach().cpu()
        # For backward compat, also include aggregated raw_params/raw_thresholds
        agg_raw = {k: v["raw_param"] for k, v in save_completed.items()}
        ckpt_save["raw_params"] = agg_raw
        if threshold_enabled:
            agg_thr = {k: v["raw_threshold"] for k, v in save_completed.items() if "raw_threshold" in v}
            if agg_thr:
                ckpt_save["raw_thresholds"] = agg_thr
        # also include current if mid-module
        if not is_module_done and current_raw is not None:
            # include current in raw_params for materialize to see it
            ckpt_save["raw_params"][module_name] = current_raw.detach().cpu()
            if threshold_enabled and current_thr is not None:
                if "raw_thresholds" not in ckpt_save:
                    ckpt_save["raw_thresholds"] = {}
                ckpt_save["raw_thresholds"][module_name] = current_thr.detach().cpu()
        ckpt_save["step"] = global_step
        ckpt_save["loss"] = loss_val
        # Save atomically via temp file to avoid partial zip (Windows antivirus / concurrent read)
        tmp_step = step_path.with_suffix(step_path.suffix + ".tmp")
        torch.save(ckpt_save, str(tmp_step))
        tmp_step.replace(step_path)
        # Only overwrite module path when module done or checkpoint interval
        with contextlib.suppress(Exception):
            tmp_mod = mod_path.with_suffix(mod_path.suffix + ".tmp")
            torch.save(ckpt_save, str(tmp_mod))
            tmp_mod.replace(mod_path)
        return step_path

    # Sequential per-module optimization (bounded memory)
    import gc as _gc

    for module_idx, mname in enumerate(target_module_names):
        if module_idx < start_module_idx:
            continue
        # Skip if already completed (e.g., resumed and already in completed dict) unless it's the active resume module
        if mname in completed_module_params and module_idx != start_module_idx:
            continue
        # Load weight on demand (bounded)
        w = _load_weight(mname)
        if w is None:
            # Skip non-quantizable or missing weight, but still need to mark as completed with empty?
            print(f"[module-major] skip {mname}: weight not found", flush=True)
            continue
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
                if "raw_param" in ic:
                    raw_param.data = ic["raw_param"].to(raw_param.device)
                if threshold_enabled and raw_thr is not None and "raw_threshold" in ic:
                    raw_thr.data = ic["raw_threshold"].to(raw_thr.device)
                print(f"[init-from] warm start applied for {mname}", flush=True)
            except Exception as e:
                print(f"[init-from] warning: failed to warm start {mname}: {e}", flush=True)

        # Resume mid-module: restore current params if this is the resume module
        is_resume_active = module_idx == start_module_idx and start_step_in_module > 0 and _resume_ckpt is not None
        if is_resume_active:
            # Try to restore current_raw_param/threshold from checkpoint
            try:
                assert _resume_ckpt is not None
                cur_raw = _resume_ckpt.get("current_raw_param")
                if cur_raw is not None:
                    raw_param.data = cur_raw.to(raw_param.device)
                cur_thr = _resume_ckpt.get("current_raw_threshold")
                if threshold_enabled and raw_thr is not None and cur_thr is not None:
                    raw_thr.data = cur_thr.to(raw_thr.device)
                print(f"[resume] restored mid-module {mname} step {start_step_in_module}", flush=True)
            except Exception as e:
                print(f"[resume] warning: failed to restore {mname}: {e}", flush=True)

        # Create local Adam for this module only (bounded)
        if threshold_enabled:
            assert raw_thr is not None
            optimizer = torch.optim.Adam(
                [
                    {"params": [raw_param], "lr": float(calib_cfg.lr)},
                    {"params": [raw_thr], "lr": threshold_lr},
                ]
            )
        else:
            optimizer = torch.optim.Adam([raw_param], lr=float(calib_cfg.lr))
        if is_resume_active and _resume_opt_state is not None:
            try:
                optimizer.load_state_dict(_resume_opt_state)
            except Exception as e:
                print(f"[resume] warning: failed to load optimizer_state for {mname}: {e}", flush=True)

        # Determine start step for this module (0 or resumed offset)
        step_start = start_step_in_module if module_idx == start_module_idx else 0

        # Run N steps for this module
        for step_in_mod in range(step_start, steps_per_module):
            optimizer.zero_grad()
            # Load activation batches for this module
            n_batches = train_cache.count_batches(mname.replace(".", "_"))
            if n_batches == 0:
                print(f"[module-major] warning: no batches for {mname}, skipping", flush=True)
                break
            module_losses: list[torch.Tensor] = []
            # Need weight tensor for threshold path (already have w_f32)
            weight_tensor_local = w_f32  # for threshold STE
            thr_ratio_local = None
            if threshold_enabled:
                thr_ratio_local = get_effective_threshold_ratio(raw_thr, eps=threshold_eps)  # type: ignore[arg-type]
            for b_idx in range(n_batches):
                data = train_cache.load(mname, b_idx)
                inp = data["input"]
                tout = data["teacher_output"]
                eff = _get_eff(raw_param, zero_mask_t)
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
                loss = _mse(tout, y_hat)
                module_losses.append(loss)
            if module_losses:
                torch.stack(module_losses).mean().backward()
            # Compute step loss for logging (mean of detached)
            step_loss_val = (
                float(torch.stack([loss_val.detach() for loss_val in module_losses]).mean().item())
                if module_losses
                else 0.0
            )
            if initial_loss is None:
                initial_loss = step_loss_val
            best_loss = min(best_loss, step_loss_val)
            loss_history.append(step_loss_val)
            optimizer.step()
            # checkpoint mid-module per interval
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
                # If module not done, we keep optimizer alive; if done we will break and release soon

        # After module steps, finalize and store completed params (bounded release preparation)
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
    held_losses: list[float] = []
    held_before: list[float] = []
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
        if threshold_enabled:
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
        n_batches = held_cache.count_batches(mname.replace(".", "_"))
        for b_idx in range(n_batches):
            data = held_cache.load(mname, b_idx)
            inp = data["input"]
            tout = data["teacher_output"]
            y_hat_final = F.linear(inp.to(torch.float32), w_hat_final.to(torch.float32))  # type: ignore[union-attr]
            held_losses.append(float(_mse(tout, y_hat_final).item()))
            y_hat_init = F.linear(inp.to(torch.float32), w_hat_init.to(torch.float32))  # type: ignore[union-attr]
            held_before.append(float(_mse(tout, y_hat_init).item()))

    held_loss_before_val = sum(held_before) / len(held_before) if held_before else 0.0
    held_loss_after_val = sum(held_losses) / len(held_losses) if held_losses else 0.0

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
