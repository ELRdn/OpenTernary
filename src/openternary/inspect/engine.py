"""Inspection engine — raw headerを正本とする exact inventory."""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
from typing import Any

from openternary import __version__
from openternary.adapters.gemma4 import get_adapter
from openternary.config.schema import AppConfig
from openternary.utils.hf_cache import resolve_snapshot
from openternary.utils.safetensors_header import parse_safetensors_header

_DTYPE_NBYTES: dict[str, int] = {
    "F32": 4,
    "F64": 8,
    "F16": 2,
    "BF16": 2,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}


def _find_safetensors_file(snapshot: pathlib.Path) -> pathlib.Path:
    """snapshot内のmodel.safetensorsを解決."""
    primary = snapshot / "model.safetensors"
    if primary.exists():
        return primary
    candidates = sorted(snapshot.glob("*.safetensors"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No *.safetensors found in {snapshot}")


def _load_config_json(snapshot: pathlib.Path) -> dict[str, Any]:
    p = snapshot / "config.json"
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object: {p}")
    return data  # type: ignore[return-value]


def _collect_tensors(snapshot: pathlib.Path) -> list[dict[str, Any]]:
    """headerからtensor一覧を収集 (payload非ロード)."""
    safetensors_path = _find_safetensors_file(snapshot)
    header = parse_safetensors_header(safetensors_path)
    items: list[dict[str, Any]] = []
    for name in sorted(header.keys()):
        info = header[name]
        shape = info["shape"]  # type: ignore[assignment]
        dtype = info["dtype"]  # type: ignore[assignment]
        assert isinstance(shape, list)
        assert isinstance(dtype, str)
        items.append({"name": name, "shape": shape, "dtype": dtype})
    return items


def _compute_fingerprint(deterministic: dict[str, Any]) -> str:
    canonical = json.dumps(deterministic, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_fingerprint_payload(
    config: AppConfig,
    arch: dict[str, Any],
    classified: list[dict[str, Any]],
    summary: dict[str, Any],
    dtype_report: dict[str, Any],
    memory_estimate: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Phase 1と同一のfingerprint対象payloadを構築（Phase 2でも不変）."""
    return {
        "version": 1,
        "model": {"id": config.model.id, "revision": config.model.revision, "adapter": "gemma4"},
        "architecture": arch,
        "quantization_policy": {
            "target": {"attention": config.quantization.target.attention, "mlp": config.quantization.target.mlp},
            "group_size": config.quantization.group_size,
            "exclude_patterns": ["embed_tokens", "lm_head", ".*\\.norm$", "vision_tower.*", "audio_tower.*", "ple"],
        },
        "tensors": classified,
        "summary": summary,
        "dtype_report": dtype_report,
        "memory_estimate": memory_estimate,
        "warnings": warnings,
        "adapter_version": f"{__version__}.gemma4.v1",
    }


def run_inspection(
    config: AppConfig,
    snapshot_path: pathlib.Path | None = None,
    load_weights: bool = False,
) -> dict[str, Any]:
    """Inspectionを実行し、inspection.json相当のdictを返す.

    Args:
        config: 解決済みAppConfig
        snapshot_path: 解決済みsnapshot Path. Noneならconfig.model.id/revisionからcache-only解決
        load_weights: Trueならtensor payloadをstreamingで読み統計付加 (ml extra必要)
    """
    # snapshot解決
    if snapshot_path is None:
        snapshot_path = resolve_snapshot(config.model.id, config.model.revision)
    else:
        snapshot_path = pathlib.Path(snapshot_path)

    raw_config = _load_config_json(snapshot_path)
    # get_adapterは通常HF IDを想定するが、CLIでローカルpathをmodel.idに上書きした場合は
    # config.jsonのmodel_typeからfallbackする (cache-only UXのため)
    try:
        adapter = get_adapter(
            config.model.id,
            target_attention=config.quantization.target.attention,
            target_mlp=config.quantization.target.mlp,
        )
    except ValueError:
        # snapshotのconfigから推定 — gemma4ならGemma4Adapter
        mt = str(raw_config.get("model_type", "")).lower()
        archs = raw_config.get("architectures", [])
        arch_str = " ".join(str(x).lower() for x in archs) if isinstance(archs, list) else ""
        if "gemma" in mt or "gemma" in arch_str or "gemma4" in mt:
            from openternary.adapters.gemma4 import Gemma4Adapter

            adapter = Gemma4Adapter(
                target_attention=config.quantization.target.attention,
                target_mlp=config.quantization.target.mlp,
            )
        else:
            raise
    arch = adapter.architecture_info(raw_config)

    raw_tensors = _collect_tensors(snapshot_path)
    classified: list[dict[str, Any]] = []
    warnings: list[str] = []

    # 層数整合性warn
    text_layers = (
        raw_config.get("text_config", {}).get("num_hidden_layers")
        if isinstance(raw_config.get("text_config"), dict)
        else None
    )
    if isinstance(text_layers, int):
        max_idx = -1
        for t in raw_tensors:
            name: str = t["name"]  # type: ignore[assignment]
            # model.layers.N.
            import re

            m = re.search(r"model\.layers\.(\d+)\.", name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx >= 0 and max_idx + 1 != text_layers:
            warnings.append(
                f"layer count mismatch: text_config.num_hidden_layers={text_layers} vs max layer index {max_idx} (+1={max_idx + 1})"
            )

    # 分類
    for t in raw_tensors:
        info = adapter.classify(t["name"], t["shape"], t["dtype"])  # type: ignore[arg-type]
        classified.append(
            {
                "name": info.name,
                "shape": info.shape,
                "dtype": info.dtype,
                "param_count": info.param_count,
                "role": info.role,
                "quantizable": info.quantizable,
                "exclude_reason": info.exclude_reason,
            }
        )
    # 既にsortedだが念のため
    classified.sort(key=lambda x: x["name"])

    total_tensors = len(classified)
    total_params = sum(c["param_count"] for c in classified)
    quantizable_tensors = sum(1 for c in classified if c["quantizable"])
    quantizable_params = sum(c["param_count"] for c in classified if c["quantizable"])
    excluded_tensors = total_tensors - quantizable_tensors

    # dtype集計
    dtype_counter: collections.Counter[str] = collections.Counter()
    for c in classified:
        dtype_counter[c["dtype"]] += c["param_count"]
    by_dtype = dict(dtype_counter)

    # memory estimate (header由来のみ)
    bf16_bytes = 0
    for c in classified:
        nbytes = _DTYPE_NBYTES.get(c["dtype"], 2)  # 不明は2で近似
        bf16_bytes += c["param_count"] * nbytes
    bf16_gb = round(bf16_bytes / (1024**3), 4)
    fp32_gb = round(total_params * 4 / (1024**3), 4)

    summary = {
        "total_tensors": total_tensors,
        "quantizable_tensors": quantizable_tensors,
        "excluded_tensors": excluded_tensors,
        "total_params": total_params,
        "quantizable_params": quantizable_params,
        "quantizable_ratio": round(quantizable_params / total_params, 6) if total_params else 0.0,
    }
    dtype_report = {"by_dtype": by_dtype, "source": "safetensors_header"}
    # fingerprint対象はPhase 1と同一（ternaryはNoneのまま）
    fingerprint_memory = {
        "bf16_GB": bf16_gb,
        "fp32_GB": fp32_gb,
        "ternary": None,
        "notes": "packing format not defined yet; Phase 2+ will split theoretical_entropy_bits / packed_weight_bits / scale_overhead",
    }
    fingerprint_payload = _build_fingerprint_payload(
        config, arch, classified, summary, dtype_report, fingerprint_memory, warnings
    )
    fingerprint = _compute_fingerprint(fingerprint_payload)

    # Phase 2 ternary estimator — fingerprint対象外（別hash）
    from openternary.quant.accounting import estimate_whole_model

    ternary_estimate = estimate_whole_model(classified, scale_dtype="fp32")
    ternary_estimator: dict[str, Any] = {
        "version": "1",
        "scheme": "absmean-per-tensor",
        "packing": "2bit-v1",
        "scale_dtype": "fp32",
        "estimate": ternary_estimate,
        "fingerprint": _compute_fingerprint(ternary_estimate),
    }
    # 表示用memory_estimateはternaryを含める
    display_memory = {
        "bf16_GB": bf16_gb,
        "fp32_GB": fp32_gb,
        "ternary": ternary_estimate,
        "notes": "Phase 2: ternary estimate uses per-tensor ceil packing (2bit-v1), fp32 scale. See ternary_estimator.",
    }

    deterministic = fingerprint_payload

    # --load-weights時はstreaming statsを付加
    weight_stats: dict[str, Any] | None = None
    if load_weights:
        weight_stats = _stream_weight_stats(snapshot_path, classified)

    result: dict[str, Any] = {
        **deterministic,
        "memory_estimate": display_memory,
        "inspection_fingerprint": fingerprint,
        "ternary_estimator": ternary_estimator,
        "weight_stats": weight_stats,
        "provenance": {
            "snapshot_path": str(snapshot_path),
            "revision": config.model.revision,
        },
    }
    return result


def _stream_weight_stats(
    snapshot: pathlib.Path,
    classified: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tensor単位でpayloadを読み統計を取得 (torch+safetensors必要)."""
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("safetensors is required for --load-weights. Install with: uv sync --extra ml") from e
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("torch is required for --load-weights. Install with: uv sync --extra ml") from e

    safetensors_path = _find_safetensors_file(snapshot)
    per_tensor: list[dict[str, Any]] = []
    global_min: float | None = None
    global_max: float | None = None

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:  # type: ignore[call-arg]
        for entry in classified:
            name = entry["name"]
            try:
                tensor = f.get_tensor(name)  # type: ignore[attr-defined]
            except Exception:
                per_tensor.append({"name": name, "error": "not found in safetensors"})
                continue
            # torch tensor stats
            t_float = tensor.float()
            t_min = float(torch.min(t_float).item())
            t_max = float(torch.max(t_float).item())
            t_mean = float(torch.mean(t_float).item())
            t_std = float(torch.std(t_float).item()) if t_float.numel() > 1 else 0.0
            zero_ratio = float((tensor == 0).float().mean().item()) if tensor.numel() > 0 else 0.0
            abs_mean = float(torch.mean(torch.abs(t_float)).item())
            if global_min is None or t_min < global_min:
                global_min = t_min
            if global_max is None or t_max > global_max:
                global_max = t_max
            per_tensor.append(
                {
                    "name": name,
                    "min": t_min,
                    "max": t_max,
                    "mean": t_mean,
                    "std": t_std,
                    "zero_ratio": zero_ratio,
                    "abs_mean": abs_mean,
                }
            )
            del tensor, t_float

    return {"per_tensor": per_tensor, "global_min": global_min, "global_max": global_max}
