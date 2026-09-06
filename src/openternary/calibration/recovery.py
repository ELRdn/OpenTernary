"""Finalization-only recovery path — Phase 4.2.

optimization loop / Teacher capture / activation cache / materialization を再実行せず、
既存 run の checkpoint / calibration_state / calibrated_snapshot から
calibration.json / metrics.json を原子的に復元する。

Integrity gate を全 PASS した時のみ書き込みを行う（fail closed）。
"""

from __future__ import annotations

import json
import pathlib
import typing
from dataclasses import dataclass

import yaml

# 標準定数（本番run gemma4-e2b-g128-threshold-real-rocm 準拠）
EXPECTED_TARGETS = 205
EXPECTED_STATE_FILES = 205
EXPECTED_STEPS_PER_MODULE = 50
EXPECTED_TOTAL_STEPS = 10250  # 205 * 50
EXPECTED_TENSOR_ENTRIES = 1951

CHECKPOINT_NAME = "step_10250.pt"


class RecoveryError(RuntimeError):
    """復旧失敗（integrity gate 不通過等）."""


class IntegrityError(RecoveryError):
    """Integrity gate fail closed."""


@dataclass
class IntegrityReport:
    passed: bool
    details: dict[str, typing.Any]
    warnings: list[str]
    errors: list[str]


def _read_yaml(p: pathlib.Path) -> dict[str, typing.Any]:
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _find_checkpoint(run_dir: pathlib.Path, resume_from: pathlib.Path | str | None = None) -> pathlib.Path:
    if resume_from is not None:
        cp = pathlib.Path(resume_from)
        # 相対パスなら run_dir からの相対解決も試す
        if not cp.is_absolute() and not cp.exists():
            alt = run_dir / cp
            if alt.exists():
                cp = alt
        if not cp.exists():
            raise FileNotFoundError(f"checkpoint not found: {cp}")
        return cp
    ckpt_dir = run_dir / "artifacts" / "checkpoint"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
    # step_*.pt / module_*.pt の両方を探索し、step_10250.pt を優先
    exact = ckpt_dir / CHECKPOINT_NAME
    if exact.exists():
        return exact
    cands = sorted(ckpt_dir.glob("step_*.pt"))
    if cands:
        return sorted(cands)[-1]
    cands = sorted(ckpt_dir.glob("module_*.pt"))
    if cands:
        return sorted(cands)[-1]
    raise FileNotFoundError(f"no checkpoint found in {ckpt_dir}")


def _load_checkpoint(ckpt_path: pathlib.Path) -> dict[str, typing.Any]:
    try:
        import torch  # type: ignore[import]
    except ImportError as e:
        raise ImportError("torch is required to read checkpoint") from e
    data = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint format invalid (not dict): {ckpt_path}")
    return data


def _count_state_files(run_dir: pathlib.Path) -> tuple[int, list[pathlib.Path]]:
    state_dir = run_dir / "artifacts" / "calibration_state"
    if not state_dir.exists():
        return 0, []
    # P0 は .safetensors が正、フォールバックで .pt も数えるが優先は safetensors
    safes = sorted(state_dir.glob("*.safetensors"))
    if safes:
        return len(safes), safes
    pts = sorted(state_dir.glob("*.pt"))
    return len(pts), pts


def _check_snapshot(snapshot_dir: pathlib.Path) -> dict[str, typing.Any]:
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"snapshot not found: {snapshot_dir}")
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"snapshot index not found: {index_path}")
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError(f"index weight_map invalid: {index_path}")
    tensor_entries = len(weight_map)
    # shards existence
    shards = sorted(set(weight_map.values()))
    missing: list[str] = []
    for s in shards:
        if not (snapshot_dir / s).exists():
            missing.append(s)
    # header open check
    header_errors: list[str] = []
    if not missing:
        from openternary.utils.safetensors_header import parse_safetensors_header

        for s in shards:
            sf = snapshot_dir / s
            try:
                parse_safetensors_header(sf)
            except Exception as e:
                header_errors.append(f"{s}: {e}")
            # さらに safe_open でも確認（torch 無しでも header で十分だが念のため）
            try:
                from safetensors import safe_open  # type: ignore[import]

                with safe_open(str(sf), framework="np") as f:  # type: ignore[union-attr]
                    _ = list(f.keys())
            except Exception:
                # safe_open が無い環境では header のみで判定
                pass
    return {
        "index_path": str(index_path),
        "tensor_entries": tensor_entries,
        "shard_count": len(shards),
        "shards": shards,
        "missing_shards": missing,
        "header_errors": header_errors,
        "weight_map": weight_map,
    }


def integrity_gate(
    run_dir: pathlib.Path | str,
    checkpoint_path: pathlib.Path | str | None = None,
    expected_targets: int | None = None,
    expected_state_files: int | None = None,
    expected_steps_per_module: int | None = None,
    expected_total_steps: int | None = None,
    expected_tensor_entries: int | None = None,
) -> IntegrityReport:
    """Integrity gate — PASS 時のみ復旧を許可.

    チェック項目:
    - config.yaml 存在
    - state files == expected_state_files
    - checkpoint cursor module_idx == expected_targets
    - steps_per_module == expected_steps_per_module
    - total expected steps == expected_total_steps
    - snapshot index tensor entries == expected_tensor_entries
    - all shards exist
    - all headers open OK
    """
    # resolve defaults at call time so monkeypatch of EXPECTED_* works
    if expected_targets is None:
        expected_targets = EXPECTED_TARGETS
    if expected_state_files is None:
        expected_state_files = EXPECTED_STATE_FILES
    if expected_steps_per_module is None:
        expected_steps_per_module = EXPECTED_STEPS_PER_MODULE
    if expected_total_steps is None:
        expected_total_steps = EXPECTED_TOTAL_STEPS
    if expected_tensor_entries is None:
        expected_tensor_entries = EXPECTED_TENSOR_ENTRIES
    run_dir = pathlib.Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, typing.Any] = {}

    # config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        errors.append(f"config.yaml missing: {config_path}")
        details["config_exists"] = False
    else:
        details["config_exists"] = True
        try:
            cfg = _read_yaml(config_path)
            details["config"] = cfg
            # steps_per_module を config からも確認（較正 steps）
            try:
                steps_cfg = int(cfg.get("calibration", {}).get("steps", expected_steps_per_module))
                details["config_steps_per_module"] = steps_cfg
                if steps_cfg != expected_steps_per_module:
                    errors.append(
                        f"steps_per_module mismatch: config {steps_cfg} != expected {expected_steps_per_module}"
                    )
            except Exception:
                pass
        except Exception as e:
            errors.append(f"config.yaml parse failed: {e}")

    # checkpoint
    try:
        ckpt_path = _find_checkpoint(run_dir, checkpoint_path)
        details["checkpoint_path"] = str(ckpt_path)
        ckpt = _load_checkpoint(ckpt_path)
        details["checkpoint_keys"] = list(ckpt.keys())
        # cursor module_idx
        cursor = ckpt.get("module_cursor", {})
        if isinstance(cursor, dict):
            module_idx = int(cursor.get("module_idx", -1))
            details["checkpoint_module_idx"] = module_idx
            if module_idx != expected_targets:
                errors.append(f"checkpoint module_idx {module_idx} != expected {expected_targets}")
            steps_per_mod_ckpt = int(cursor.get("steps_per_module", expected_steps_per_module))
            details["checkpoint_steps_per_module"] = steps_per_mod_ckpt
            if steps_per_mod_ckpt != expected_steps_per_module:
                errors.append(f"checkpoint steps_per_module {steps_per_mod_ckpt} != expected {expected_steps_per_module}")
            # completed count
            manifest = ckpt.get("completed_manifest", [])
            if isinstance(manifest, list):
                details["completed_manifest_count"] = len(manifest)
        else:
            # legacy checkpoint may have no module_cursor; fallback to step count
            warnings.append("checkpoint has no module_cursor (legacy)")
            # try to infer from loss_history length? not enough
        # global step check
        global_step = ckpt.get("step")
        if global_step is not None:
            details["checkpoint_global_step"] = int(global_step)
            # expected total steps = 10250 (最後は step_10250 だが 0-index なら 10249)
            # 本番は step_10250.pt で global step 10250 として保存されている想定
            # 厳密には module_idx*steps_per_module + step_in_module == total steps
            # ここでは総ステップが total_steps と一致するか寛容にチェック
            # manifest ありなら module_idx が主、無ければ global_step で判定
            if manifest is not None and isinstance(manifest, list) and len(manifest) == expected_targets:
                # manifest がある場合は global_step は auxiliary、厳密一致を要求しない
                pass
            else:
                # legacy: global_step が total_steps-1 または total_steps
                if int(global_step) not in (expected_total_steps, expected_total_steps - 1):
                    warnings.append(f"checkpoint global step {global_step} != expected {expected_total_steps}")
        # threshold_enabled 等は情報として保持
        details["checkpoint_threshold_enabled"] = ckpt.get("threshold_enabled")
    except Exception as e:
        errors.append(f"checkpoint check failed: {e}")
        details["checkpoint_error"] = str(e)

    # state files
    try:
        count, files = _count_state_files(run_dir)
        details["state_files"] = count
        details["state_file_list"] = [str(f) for f in files[:5]]  # sample
        if count != expected_state_files:
            errors.append(f"state_files {count} != expected {expected_state_files}")
    except Exception as e:
        errors.append(f"state_files check failed: {e}")

    # snapshot
    try:
        snap_dir = run_dir / "artifacts" / "calibrated_snapshot"
        snap_info = _check_snapshot(snap_dir)
        details["snapshot"] = {k: v for k, v in snap_info.items() if k != "weight_map"}
        details["snapshot_tensor_entries"] = snap_info["tensor_entries"]
        if snap_info["tensor_entries"] != expected_tensor_entries:
            errors.append(
                f"snapshot tensor entries {snap_info['tensor_entries']} != expected {expected_tensor_entries}"
            )
        if snap_info["missing_shards"]:
            errors.append(f"missing shards: {snap_info['missing_shards']}")
        if snap_info["header_errors"]:
            errors.append(f"header errors: {snap_info['header_errors']}")
        if snap_info["shard_count"] == 0:
            errors.append("snapshot shard_count == 0")
        # snapshot 12 shards / 9.54GB は本番の期待だが、テント的には shard_count>0 なら OK、
        # 本番では 12 を期待するが汎用的には expected 固定でチェックしない（tensor_entries が主）
    except Exception as e:
        errors.append(f"snapshot check failed: {e}")
        details["snapshot_error"] = str(e)

    # 総合判定
    passed = len(errors) == 0
    return IntegrityReport(passed=passed, details=details, warnings=warnings, errors=errors)


def _atomic_write_json(path: pathlib.Path, data: dict[str, typing.Any]) -> None:
    import contextlib

    path.parent.mkdir(parents=True, exist_ok=True)
    # 一時ファイルに書き込み後 rename（atomic）
    tmp = path.with_suffix(path.suffix + ".tmp")
    # 既存 tmp があれば削除
    if tmp.exists():
        with contextlib.suppress(Exception):
            tmp.unlink()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _collect_recovery_data(
    run_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    ckpt: dict[str, typing.Any],
    snap_info: dict[str, typing.Any],
    state_count: int,
) -> tuple[dict[str, typing.Any], list[str]]:
    """checkpoint / config / logs から復元可能なフィールドを収集.

    存在しない値は null + recovery_warning とする。
    """
    warnings: list[str] = []
    # config 読み込み
    config_path = run_dir / "config.yaml"
    cfg_yaml = _read_yaml(config_path) if config_path.exists() else {}
    calib_cfg = cfg_yaml.get("calibration", {}) if isinstance(cfg_yaml, dict) else {}
    quant_cfg = cfg_yaml.get("quantization", {}) if isinstance(cfg_yaml, dict) else {}

    # checkpoint からの復元（loss 系）
    loss_history = ckpt.get("loss_history")
    initial_loss = ckpt.get("initial_loss")
    best_loss = ckpt.get("best_loss")
    final_loss = ckpt.get("loss")
    if final_loss is None and loss_history:
        try:
            final_loss = loss_history[-1]
        except Exception:
            final_loss = None

    # heldout は checkpoint に無い場合が多い（runner の held 計算後 calibration.json に保存）
    # checkpoint には heldout_loss_before/after が無いため null 扱い
    heldout_before = ckpt.get("heldout_loss_before")
    heldout_after = ckpt.get("heldout_loss_after")
    # runner の checkpoint には heldout_* が無い -> warnings
    if heldout_before is None:
        warnings.append("heldout_before not in checkpoint -> null")
    if heldout_after is None:
        warnings.append("heldout_after not in checkpoint -> null")

    # 各種 fingerprint / scale stats
    scale_fingerprint_before = ckpt.get("scale_fingerprint_before")
    scale_fingerprint_after = ckpt.get("scale_fingerprint_after")
    # threshold / code fingerprint
    threshold_fingerprint_before = ckpt.get("threshold_fingerprint_before")
    threshold_fingerprint_after = ckpt.get("threshold_fingerprint_after")
    code_fingerprint_before = ckpt.get("code_fingerprint_before")
    code_fingerprint_after = ckpt.get("code_fingerprint_after")
    code_change_ratio = ckpt.get("code_change_ratio")
    threshold_ratio_mean = ckpt.get("threshold_ratio_mean")

    # ログ由来の統計（logs/checkpoint 内の statistics）があれば読む — best effort
    # 存在しない場合は warning
    # ここでは checkpoint に scale 統計が無ければ warnings
    if scale_fingerprint_before is None:
        warnings.append("scale_fingerprint_before not in checkpoint -> null")
    if scale_fingerprint_after is None:
        warnings.append("scale_fingerprint_after not in checkpoint -> null")

    # backend / device / peak_vram
    backend = ckpt.get("backend")
    device_name = ckpt.get("device_name") or ckpt.get("device")
    peak_vram = ckpt.get("peak_vram_bytes") or ckpt.get("peak_vram")
    if backend is None:
        warnings.append("backend not in checkpoint -> null")
    if device_name is None:
        warnings.append("device_name not in checkpoint -> null")
    if peak_vram is None:
        warnings.append("peak_vram not in checkpoint -> null")

    # calibration_state 由来の scale stats を state ファイルから計算できるか試す
    # ここでは大規模 weight ロードが必要なためやらず null とする（recovery_warning）
    # ただし state_count は既知
    if threshold_fingerprint_before is None:
        warnings.append("threshold_fingerprint_before not in checkpoint -> null")
    if threshold_fingerprint_after is None:
        warnings.append("threshold_fingerprint_after not in checkpoint -> null")
    if code_fingerprint_before is None:
        warnings.append("code_fingerprint_before not in checkpoint -> null")
    if code_change_ratio is None:
        warnings.append("code_change_ratio not in checkpoint -> null")

    # config 由来
    method = calib_cfg.get("method") or ckpt.get("method") or "recon-threshold"
    dataset = calib_cfg.get("dataset") or ckpt.get("dataset") or "unknown"
    num_samples = calib_cfg.get("num_samples")
    seq_len = calib_cfg.get("seq_len")
    steps = calib_cfg.get("steps", EXPECTED_STEPS_PER_MODULE)
    lr = calib_cfg.get("lr")

    # train/heldout/sample hashes 等も checkpoint に無い場合 null
    train_hashes = ckpt.get("train_sample_hashes") or ckpt.get("train_hashes")
    held_hashes = ckpt.get("heldout_sample_hashes") or ckpt.get("held_hashes")
    smoke_hashes = ckpt.get("smoke_sample_hashes")
    # contamination flags
    contamination_train_smoke = ckpt.get("contamination_train_smoke")
    contamination_held_smoke = ckpt.get("contamination_held_smoke")

    # 環境情報（best effort）
    import contextlib

    env_path = run_dir / "environment.json"
    if env_path.exists():
        with contextlib.suppress(Exception):
            _env_data = json.loads(env_path.read_text(encoding="utf-8"))
            _ = _env_data  # keep reference for future use, silence F841

    # 復元データ本体
    data: dict[str, typing.Any] = {
        "method": method,
        "dataset": dataset,
        "num_samples": num_samples,
        "seq_len": seq_len,
        "steps": steps,
        "steps_per_module": ckpt.get("module_cursor", {}).get("steps_per_module", EXPECTED_STEPS_PER_MODULE)
        if isinstance(ckpt.get("module_cursor"), dict)
        else EXPECTED_STEPS_PER_MODULE,
        "lr": lr,
        "loss_history": loss_history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "heldout_loss_before": heldout_before,
        "heldout_loss_after": heldout_after,
        "train_sample_hashes": train_hashes,
        "heldout_sample_hashes": held_hashes,
        "smoke_sample_hashes": smoke_hashes,
        "contamination_train_smoke": contamination_train_smoke,
        "contamination_held_smoke": contamination_held_smoke,
        "scale_fingerprint_before": scale_fingerprint_before,
        "scale_fingerprint_after": scale_fingerprint_after,
        "threshold_fingerprint_before": threshold_fingerprint_before,
        "threshold_fingerprint_after": threshold_fingerprint_after,
        "code_fingerprint_before": code_fingerprint_before,
        "code_fingerprint_after": code_fingerprint_after,
        "code_change_ratio": code_change_ratio,
        "threshold_ratio_mean": threshold_ratio_mean,
        "threshold_eps": calib_cfg.get("threshold_eps") or ckpt.get("threshold_eps"),
        "threshold_ste_width": calib_cfg.get("threshold_ste_width") or ckpt.get("threshold_ste_width"),
        "backend": backend,
        "device_name": device_name,
        "peak_vram_bytes": peak_vram,
        "quantization": {
            "group_size": quant_cfg.get("group_size"),
            "scale_granularity": quant_cfg.get("scale_granularity"),
            "grouping_scheme": quant_cfg.get("grouping_scheme"),
        },
        "snapshot": {
            "tensor_entries": snap_info.get("tensor_entries"),
            "shard_count": snap_info.get("shard_count"),
            "shards": snap_info.get("shards"),
        },
        "recovery_warnings": warnings,
    }
    return data, warnings


def finalize_run(
    run_dir: pathlib.Path | str,
    checkpoint_path: pathlib.Path | str | None = None,
    expected_targets: int | None = None,
    expected_state_files: int | None = None,
    expected_steps_per_module: int | None = None,
    expected_total_steps: int | None = None,
    expected_tensor_entries: int | None = None,
    dry_run: bool = False,
) -> dict[str, typing.Any]:
    """既存 run に対して finalization-only 復旧を実行.

    - optimization / capture / cache / materialization を一切実行しない
    - integrity gate PASS 時のみ calibration.json / metrics.json を atomic write
    - 既存 snapshot / checkpoint / state の mtime/hash は変更しない

    Returns:
        dict with details on success
    Raises:
        IntegrityError if gate fails
    """
    run_dir = pathlib.Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    # 1. integrity gate
    report = integrity_gate(
        run_dir,
        checkpoint_path,
        expected_targets,
        expected_state_files,
        expected_steps_per_module,
        expected_total_steps,
        expected_tensor_entries,
    )
    if not report.passed:
        raise IntegrityError(f"integrity gate FAILED: {report.errors} warnings={report.warnings} details={report.details}")

    if dry_run:
        return {"status": "dry_run_passed", "report": report.details, "warnings": report.warnings}

    # 2. resolve checkpoint
    ckpt_path = _find_checkpoint(run_dir, checkpoint_path)
    ckpt = _load_checkpoint(ckpt_path)

    # 3. snapshot info / state count
    snap_dir = run_dir / "artifacts" / "calibrated_snapshot"
    snap_info = _check_snapshot(snap_dir)
    state_count, _ = _count_state_files(run_dir)

    # 4. gather recovery data
    recovery_data, recovery_warnings = _collect_recovery_data(run_dir, ckpt_path, ckpt, snap_info, state_count)

    # 5. build calibration.json
    # 環境 run_id 取得
    run_id = "unknown"
    env_path = run_dir / "environment.json"
    if env_path.exists():
        try:
            env_data = json.loads(env_path.read_text(encoding="utf-8"))
            run_id = str(env_data.get("run_id", run_id))
        except Exception:
            pass

    # 既存 logs があれば読む — loss/statistics の補完（best effort）
    # logs/checkpoint 内のログは現行 runner が保存しないため警告扱い

    recovery_meta = {
        "recovered": True,
        "recovery_mode": "finalization-only",
        "optimization_reexecuted": False,
        "materialization_reexecuted": False,
        "source_checkpoint": ckpt_path.name,
        "state_files": state_count,
        "snapshot_tensor_entries": snap_info["tensor_entries"],
        "snapshot_integrity": "pass",
    }

    # calibration.json 本文 — source artifact に無い値は null + warning とする
    calibration_payload: dict[str, typing.Any] = {
        "run_id": run_id,
        "method": recovery_data.get("method"),
        "dataset": recovery_data.get("dataset"),
        "num_samples": recovery_data.get("num_samples"),
        "seq_len": recovery_data.get("seq_len"),
        "steps": recovery_data.get("steps"),
        "steps_per_module": recovery_data.get("steps_per_module"),
        "lr": recovery_data.get("lr"),
        "loss_history": recovery_data.get("loss_history"),
        "initial_loss": recovery_data.get("initial_loss"),
        "final_loss": recovery_data.get("final_loss"),
        "best_loss": recovery_data.get("best_loss"),
        "heldout_loss_before": recovery_data.get("heldout_loss_before"),
        "heldout_loss_after": recovery_data.get("heldout_loss_after"),
        "train_sample_hashes": recovery_data.get("train_sample_hashes"),
        "heldout_sample_hashes": recovery_data.get("heldout_sample_hashes"),
        "smoke_sample_hashes": recovery_data.get("smoke_sample_hashes"),
        "contamination_train_smoke": recovery_data.get("contamination_train_smoke"),
        "contamination_held_smoke": recovery_data.get("contamination_held_smoke"),
        "scale_fingerprint_before": recovery_data.get("scale_fingerprint_before"),
        "scale_fingerprint_after": recovery_data.get("scale_fingerprint_after"),
        "threshold_fingerprint_before": recovery_data.get("threshold_fingerprint_before"),
        "threshold_fingerprint_after": recovery_data.get("threshold_fingerprint_after"),
        "code_fingerprint_before": recovery_data.get("code_fingerprint_before"),
        "code_fingerprint_after": recovery_data.get("code_fingerprint_after"),
        "code_change_ratio": recovery_data.get("code_change_ratio"),
        "threshold_ratio_mean": recovery_data.get("threshold_ratio_mean"),
        "threshold_eps": recovery_data.get("threshold_eps"),
        "threshold_ste_width": recovery_data.get("threshold_ste_width"),
        "backend": recovery_data.get("backend"),
        "device_name": recovery_data.get("device_name"),
        "peak_vram_bytes": recovery_data.get("peak_vram_bytes"),
        "quantization": recovery_data.get("quantization"),
        "snapshot": recovery_data.get("snapshot"),
        "recovery_warnings": recovery_warnings,
        **recovery_meta,
    }
    # null 正規化 — Python None は JSON null になる
    # 既に None のものはそのまま

    # metrics.json — status completed は gate PASS 時のみ
    metrics_payload: dict[str, typing.Any] = {
        "status": "completed",
        "run_id": run_id,
        "initial_loss": recovery_data.get("initial_loss"),
        "final_loss": recovery_data.get("final_loss"),
        "best_loss": recovery_data.get("best_loss"),
        "heldout_before": recovery_data.get("heldout_loss_before"),
        "heldout_after": recovery_data.get("heldout_loss_after"),
        "recovery_warnings": recovery_warnings,
        **recovery_meta,
    }

    # atomic write（既存 run 直下）
    # mtime/hash を変えない対象（checkpoint/state/snapshot）は触らない
    _atomic_write_json(run_dir / "calibration.json", calibration_payload)
    _atomic_write_json(run_dir / "metrics.json", metrics_payload)

    return {
        "status": "recovered",
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "state_files": state_count,
        "snapshot_tensor_entries": snap_info["tensor_entries"],
        "calibration_json": str(run_dir / "calibration.json"),
        "metrics_json": str(run_dir / "metrics.json"),
        "recovery_meta": recovery_meta,
        "recovery_warnings": recovery_warnings,
    }


__all__ = [
    "EXPECTED_TARGETS",
    "EXPECTED_STATE_FILES",
    "EXPECTED_STEPS_PER_MODULE",
    "EXPECTED_TOTAL_STEPS",
    "EXPECTED_TENSOR_ENTRIES",
    "CHECKPOINT_NAME",
    "IntegrityError",
    "RecoveryError",
    "IntegrityReport",
    "integrity_gate",
    "finalize_run",
]
