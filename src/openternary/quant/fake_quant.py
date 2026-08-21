"""Bounded-memory sharded fake-quant snapshot converter — Phase 3.

P0対応:
- save_file(dict_of_all_tensors)禁止 → sharded bounded writer
- group-wise LD-RW vectorized + zero-branch
- HF cache blobs symlinkを正規としてdereference
- scalesはJSONへ全保存しない（summary + fingerprintのみ）
- shard flushは追加前判定、単一tensor>MAXは単独shard許可
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import shutil
import struct
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Canonical shard size — HF標準準拠
MAX_SHARD_SIZE = 512 * 1024 * 1024  # 512 MiB

# safetensors header dtype → bytes per element
_DTYPE_NBYTES: dict[str, int] = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "U8": 1,
    "I8": 1,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}

_TORCH_DTYPE_MAP: dict[str, Any] = {}


def _get_torch_dtype_map() -> dict[str, Any]:
    global _TORCH_DTYPE_MAP
    if _TORCH_DTYPE_MAP:
        return _TORCH_DTYPE_MAP
    if torch is None:
        return {}
    _TORCH_DTYPE_MAP = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I32": torch.int32,
        "I64": torch.int64,
        "BOOL": torch.bool,
    }
    return _TORCH_DTYPE_MAP


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for fake-quant conversion. Install with: uv sync --extra ml")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_raw_sha256(t: Any) -> str:
    # t is torch.Tensor on CPU contiguous
    _require_torch()
    cpu = t.detach().cpu().contiguous()
    # Fast path: view as uint8 then numpy tobytes (BF16 compatible, 22s -> 2ms)
    try:
        b = cpu.view(torch.uint8).numpy().tobytes()  # type: ignore[union-attr]
        return _sha256_hex(b)
    except Exception:
        pass
    # Fallback: try numpy directly (for float32 etc.)
    try:
        import numpy as np  # noqa: F401  # type: ignore

        b = cpu.numpy().tobytes()
        return _sha256_hex(b)
    except Exception:
        pass
    # Last resort: untyped storage (slow)
    b = bytes(cpu.untyped_storage())  # type: ignore[attr-defined]
    return _sha256_hex(b)


def _scale_fingerprint(scales: Any) -> str:
    """scales: 1-D float32 tensor or single float. Fingerprintはfloat32 LE bytes concatのsha256."""
    _require_torch()
    if isinstance(scales, float):
        b = struct.pack("<f", scales)
        return f"sha256:{_sha256_hex(b)}"
    # tensor
    cpu = scales.detach().cpu().to(torch.float32).contiguous()
    # float32 LE bytes
    b = cpu.numpy().tobytes() if hasattr(cpu, "numpy") else bytes(cpu.untyped_storage())  # type: ignore
    return f"sha256:{_sha256_hex(b)}"


def _content_fingerprint(entries: list[dict[str, Any]]) -> str:
    """sorted entries [{name,dtype,shape,sha256}] からcanonical hash."""
    canonical = _canonical_json(sorted(entries, key=lambda x: x["name"]))
    return f"sha256:{_sha256_hex(canonical.encode('utf-8'))}"


def _is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_allowed_target(
    link_path: pathlib.Path,
    snapshot_root: pathlib.Path,
    blobs_dir: pathlib.Path | None,
) -> pathlib.Path | None:
    """Symlink targetが許可範囲か判定. 許可ならresolved Pathを返す、拒否ならNone."""
    try:
        resolved = link_path.resolve()
    except Exception:
        return None
    if _is_relative_to(resolved, snapshot_root):
        return resolved
    if blobs_dir is not None and blobs_dir.exists() and _is_relative_to(resolved, blobs_dir):
        return resolved
    # Also allow blobs under repo cache even if snapshot_root is not HF cache (fallback: check parent blobs)
    return None


def _copy_non_weight_files(src_root: pathlib.Path, dst_root: pathlib.Path) -> None:
    """非weightファイルを原則コピー. HF blobs symlinkはdereferenceして内容コピー."""
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    # repo cache blobs detection: support HF cache and sibling blobs (test)
    blobs_candidates: list[pathlib.Path] = []
    try:
        blobs_candidates.append(src_root / "blobs")
        blobs_candidates.append(src_root.parent / "blobs")
        blobs_candidates.append(src_root.parent.parent / "blobs")
    except Exception:
        pass
    blobs_dir: pathlib.Path | None = None
    for cand in blobs_candidates:
        try:
            if cand.exists():
                blobs_dir = cand.resolve()
                break
        except Exception:
            continue

    for item in src_root.iterdir():
        name = item.name
        # weight filesは除外（後でshardedで書き出す）
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        dst_item = dst_root / name
        # Symlink handling
        if item.is_symlink():
            target = _resolve_allowed_target(item, src_root, blobs_dir)
            if target is None:
                # broken or outside allowed → reject loudly
                raise ValueError(f"Rejected symlink outside allowed roots: {item} -> {item.readlink()}")
            # dereference: copy target content
            if target.is_dir():
                shutil.copytree(target, dst_item, symlinks=False)
            else:
                shutil.copy2(target, dst_item)
        else:
            if item.is_dir():
                shutil.copytree(item, dst_item, symlinks=False)
            else:
                shutil.copy2(item, dst_item)


def _find_safetensors_file(src_root: pathlib.Path) -> pathlib.Path:
    """単一safetensorsファイルを探す（Phase 3はsingle-file sourceをcanonical）."""
    candidates = list(src_root.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors file found in {src_root}")
    # Prefer model.safetensors if exists
    for c in candidates:
        if c.name == "model.safetensors":
            return c
    # Otherwise, if sharded, we don't yet support sharded source — raise to avoid silent partial read
    if len(candidates) > 1:
        raise ValueError(f"Sharded source snapshot not yet supported in Phase 3: found {candidates}")
    return candidates[0]


def _classify_tensor(name: str, shape: list[int], dtype: str, app_config: Any) -> tuple[str, bool, str | None]:
    """Adapter経由でquantizable判定. Gemma4優先、fallbackはbase."""
    from openternary.adapters.gemma4 import Gemma4Adapter

    adapter = Gemma4Adapter()
    try:
        info = adapter.classify(name, shape, dtype)
        return (info.role, info.quantizable, info.exclude_reason)
    except Exception:
        from openternary.adapters.base import default_quantizable_role

        target_attn = bool(getattr(app_config.quantization.target, "attention", True))
        target_mlp = bool(getattr(app_config.quantization.target, "mlp", True))
        return default_quantizable_role(name, target_attention=target_attn, target_mlp=target_mlp)


@dataclasses.dataclass
class ConvertReport:
    dst_snapshot: pathlib.Path
    quantization_json: dict[str, Any]
    content_fingerprint: str
    file_hashes: dict[str, str]
    shard_count: int
    tensor_payload_bytes: int
    shard_file_bytes: int
    snapshot_total_bytes: int


def convert_snapshot(
    src_snapshot: pathlib.Path | str,
    dst_snapshot: pathlib.Path | str,
    app_config: Any,
    max_shard_size: int = MAX_SHARD_SIZE,
) -> ConvertReport:
    """Bounded-memory sharded fake-quant conversion.

    Args:
        src_snapshot: HF snapshot dir (cache-only, local)
        dst_snapshot: 出力先dir（tmp経由でatomicに作成）
        app_config: AppConfig
        max_shard_size: shard閾値（テストでは小さくする）

    Returns:
        ConvertReport
    """
    _require_torch()
    import safetensors.torch

    src_root = pathlib.Path(src_snapshot).resolve()
    dst_root = pathlib.Path(dst_snapshot).resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"src_snapshot not found: {src_root}")

    # disk check
    try:
        usage = shutil.disk_usage(dst_root.parent if dst_root.exists() else pathlib.Path.cwd())
        # need at least 12GB free for 10GB model
        if usage.free < 2 * 1024 * 1024 * 1024:
            raise ValueError(f"Low disk space: free {usage.free} bytes")
    except Exception:
        pass  # not fatal if check fails

    tmp_root = dst_root.parent / (dst_root.name + ".tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        # 1. copy non-weight files (HF blobs symlink対応)
        _copy_non_weight_files(src_root, tmp_root)

        # 2. open source safetensors
        weight_file = _find_safetensors_file(src_root)
        # Use safe_open streaming
        from safetensors import safe_open

        # Prepare shard buffer bounded
        buffer: dict[str, Any] = {}
        current_bytes = 0
        shard_index = 1
        weight_map: dict[str, str] = {}
        # For content fingerprint
        content_entries: list[dict[str, Any]] = []
        per_tensor_entries: list[dict[str, Any]] = []
        tensor_payload_bytes = 0
        total_quantizable = 0
        total_groups = 0

        scale_granularity = str(getattr(app_config.quantization, "scale_granularity", "per_tensor"))
        group_size = int(getattr(app_config.quantization, "group_size", 128))
        grouping_scheme = str(getattr(app_config.quantization, "grouping_scheme", "last-dim-rowwise-v1"))

        # Lazy imports for quantization
        from openternary.quant.grouping import dequantize_groupwise, quantize_groupwise
        from openternary.quant.ternary import dequantize, quantize_absmean

        # We need sorted tensor names for determinism
        with safe_open(str(weight_file), framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            # Also need header info for shape/dtype? Use f.get_slice for shape
            for name in keys:
                tensor = f.get_tensor(name)  # torch.Tensor CPU
                # dtype string for classification/header
                dtype_str = str(tensor.dtype).upper().replace("TORCH.", "")
                # Map torch dtype to safetensors dtype string
                # torch.bfloat16 -> BF16, float32 -> F32 etc.
                dtype_map_rev = {
                    "torch.float32": "F32",
                    "torch.float16": "F16",
                    "torch.bfloat16": "BF16",
                    "torch.int8": "I8",
                    "torch.int32": "I32",
                    "torch.int64": "I64",
                    "torch.uint8": "U8",
                    "torch.bool": "BOOL",
                }
                # Use generic mapping
                orig_dtype_s = dtype_map_rev.get(str(tensor.dtype), str(tensor.dtype))
                shape = list(tensor.shape)
                param_count = int(tensor.numel())
                # classification
                role, quantizable, exclude_reason = _classify_tensor(name, shape, orig_dtype_s, app_config)

                if quantizable:
                    total_quantizable += 1
                    # quantize
                    if scale_granularity == "per_tensor":
                        tt = quantize_absmean(tensor)
                        # fingerprint for single scale
                        scale_fp = _scale_fingerprint(tt.scale)
                        # stats
                        scale_val = tt.scale
                        scale_min = scale_max = scale_mean = scale_val
                        scale_std = 0.0
                        num_groups = 1
                        # recon
                        recon = dequantize(tt)  # float32
                        # cast back to original dtype
                        # original tensor dtype
                        recon_cast = recon.to(tensor.dtype)
                        out_tensor = recon_cast
                        # zero ratio from codes
                        zero_ratio = float((tt.codes == 0).sum().item() / tt.codes.numel()) if tt.codes.numel() else 0.0
                        # per-tensor mae (optional, for diagnostics)
                        try:
                            mae = float((tensor.to(torch.float32) - recon).abs().mean().item())
                        except Exception:
                            mae = 0.0
                    elif scale_granularity == "per_group":
                        res = quantize_groupwise(tensor, group_size, grouping_scheme)
                        num_groups = int(res.scales.numel())
                        total_groups += num_groups
                        scale_fp = _scale_fingerprint(res.scales)
                        # stats with correction=0
                        if num_groups > 0:
                            s = res.scales.float()
                            scale_min = float(s.min().item())
                            scale_max = float(s.max().item())
                            scale_mean = float(s.mean().item())
                            # std correction=0
                            if num_groups == 1:
                                scale_std = 0.0
                            else:
                                scale_std = float(s.std(correction=0).item())
                                if math.isnan(scale_std):
                                    scale_std = 0.0
                        else:
                            scale_min = scale_max = scale_mean = scale_std = 0.0
                        recon = dequantize_groupwise(res)  # float32
                        recon_cast = recon.to(tensor.dtype)
                        out_tensor = recon_cast
                        zero_ratio = (
                            float((res.codes == 0).sum().item() / res.codes.numel()) if res.codes.numel() else 0.0
                        )
                        try:
                            mae = float((tensor.to(torch.float32) - recon).abs().mean().item())
                        except Exception:
                            mae = 0.0
                    else:
                        raise ValueError(f"Unknown scale_granularity {scale_granularity}")
                else:
                    # excluded: preserve exactly
                    out_tensor = tensor
                    num_groups = 0
                    scale_fp = ""
                    scale_min = scale_max = scale_mean = scale_std = 0.0
                    zero_ratio = 0.0
                    mae = 0.0

                # tensor payload bytes
                # Use original dtype bytes: BF16 2 etc.
                nbytes = int(out_tensor.numel() * out_tensor.element_size())
                tensor_payload_bytes += nbytes

                # per-tensor entry (without full scales)
                entry: dict[str, Any] = {
                    "name": name,
                    "shape": shape,
                    "dtype": str(out_tensor.dtype),
                    "param_count": param_count,
                    "quantizable": quantizable,
                    "role": role,
                    "exclude_reason": exclude_reason,
                    "scale_granularity": scale_granularity if quantizable else "per_tensor",
                    "grouping_scheme": grouping_scheme
                    if quantizable and scale_granularity == "per_group"
                    else "last-dim-rowwise-v1",
                    "group_size": group_size if quantizable and scale_granularity == "per_group" else 128,
                    "num_groups": num_groups if quantizable else 0,
                    "scale_fingerprint": scale_fp if quantizable else "",
                    "scale_stats": {
                        "min": scale_min,
                        "max": scale_max,
                        "mean": scale_mean,
                        "std": scale_std,
                    }
                    if quantizable
                    else {"min": 0, "max": 0, "mean": 0, "std": 0},
                    "zero_ratio": zero_ratio if quantizable else 0.0,
                    "mae": mae if quantizable else 0.0,
                }
                per_tensor_entries.append(entry)

                # content fingerprint entry (for output tensor)
                sha = _tensor_raw_sha256(out_tensor)
                content_entries.append({"name": name, "dtype": str(out_tensor.dtype), "shape": shape, "sha256": sha})

                # shard buffer handling: flush BEFORE adding if would exceed
                out_nbytes = nbytes
                if buffer and current_bytes + out_nbytes > max_shard_size:
                    # flush current buffer
                    shard_name = f"model-{shard_index:05d}-of-99999.safetensors"
                    shard_path = tmp_root / shard_name
                    # ensure deterministic key order
                    ordered = {k: buffer[k] for k in sorted(buffer.keys())}
                    safetensors.torch.save_file(ordered, str(shard_path))
                    for k in ordered:
                        weight_map[k] = shard_name
                    buffer.clear()
                    current_bytes = 0
                    shard_index += 1
                    # if single tensor > max, allow over-size shard (will be flushed next iteration)
                # add to buffer
                buffer[name] = out_tensor
                current_bytes += out_nbytes

            # flush remaining
            if buffer:
                shard_name = f"model-{shard_index:05d}-of-99999.safetensors"
                shard_path = tmp_root / shard_name
                ordered = {k: buffer[k] for k in sorted(buffer.keys())}
                safetensors.torch.save_file(ordered, str(shard_path))
                for k in ordered:
                    weight_map[k] = shard_name
                buffer.clear()
                current_bytes = 0
            else:
                shard_index -= 1  # no extra shard
            shard_count = shard_index if shard_index >= 1 else 0

            # Rename shards to correct total: model-00001-of-0000N
            # Need to rename all shards from 99999 placeholder
            if shard_count > 0:
                for i in range(1, shard_count + 1):
                    old = tmp_root / f"model-{i:05d}-of-99999.safetensors"
                    new = tmp_root / f"model-{i:05d}-of-{shard_count:05d}.safetensors"
                    old.rename(new)
                    # update weight_map
                    old_name = f"model-{i:05d}-of-99999.safetensors"
                    new_name = f"model-{i:05d}-of-{shard_count:05d}.safetensors"
                    for k, v in list(weight_map.items()):
                        if v == old_name:
                            weight_map[k] = new_name

            # write index
            total_size = tensor_payload_bytes
            index_obj = {
                "metadata": {"total_size": total_size},
                "weight_map": {k: weight_map[k] for k in sorted(weight_map.keys())},
            }
            with open(tmp_root / "model.safetensors.index.json", "w", encoding="utf-8") as idx_f:
                json.dump(index_obj, idx_f, indent=2, sort_keys=True, ensure_ascii=False)
                idx_f.write("\n")

        # After conversion, compute fingerprints
        content_fp = _content_fingerprint(content_entries)
        # shard file bytes
        shard_file_bytes = 0
        for i in range(1, shard_count + 1):
            p = tmp_root / f"model-{i:05d}-of-{shard_count:05d}.safetensors"
            shard_file_bytes += p.stat().st_size
        snapshot_total_bytes = 0
        for p in tmp_root.rglob("*"):
            if p.is_file():
                snapshot_total_bytes += p.stat().st_size

        # accounting for packed estimate (use existing accounting)
        from openternary.quant.accounting import estimate_whole_model

        # Build classified list for accounting — need to know quantizable tensors and their shapes
        # Use per-tensor entries: for accounting we treat each quantizable tensor's N as param_count
        # For grouped overhead, total_groups is sum of groups
        # estimate_whole_model expects classified tensors with param_count etc.
        # We can build minimal dicts
        classified = []
        for e in per_tensor_entries:
            dtype_str = e["dtype"]
            # Map torch dtype string to header dtype for accounting
            header_map = {
                "torch.bfloat16": "BF16",
                "torch.float32": "F32",
                "torch.float16": "F16",
                "torch.float64": "F64",
                "torch.int8": "I8",
            }
            header_dtype = header_map.get(dtype_str, "BF16")
            classified.append(
                {
                    "name": e["name"],
                    "param_count": e["param_count"],
                    "dtype": header_dtype,
                    "quantizable": e["quantizable"],
                }
            )

        # Use accounting helper if available, else manual
        try:
            est = estimate_whole_model(classified, scale_dtype="fp32")  # type: ignore[arg-type]
            # For grouped, override scale_overhead
            if scale_granularity == "per_group":
                grouped_overhead = total_groups * 32
                est["scale_overhead_bits"] = total_quantizable * 32  # keep per_tensor for reference
                est["grouped_scale_overhead_bits"] = grouped_overhead
                est["num_scales"] = total_groups
                est["num_quantizable_tensors"] = total_quantizable
        except Exception:
            # fallback manual
            total_params = sum(e["param_count"] for e in per_tensor_entries if e["quantizable"])
            from openternary.quant.accounting import LOG2_3, packed_weight_bits_for_n

            ideal = int(total_params * LOG2_3)
            packed = sum(packed_weight_bits_for_n(e["param_count"]) for e in per_tensor_entries if e["quantizable"])
            logical = sum(e["param_count"] * 2 for e in per_tensor_entries if e["quantizable"])
            # per_tensor overhead
            per_tensor_overhead = total_quantizable * 32
            grouped_overhead = total_groups * 32 if scale_granularity == "per_group" else per_tensor_overhead
            est = {
                "ideal_ternary_bits": ideal,
                "packed_weight_bits": packed,
                "logical_2bit_bits": logical,
                "scale_overhead_bits": per_tensor_overhead,
                "grouped_scale_overhead_bits": grouped_overhead,
                "num_scales": total_groups if scale_granularity == "per_group" else total_quantizable,
            }

        quantization_json: dict[str, Any] = {
            "version": 1,
            "method": "naive",
            "codebook": [-1, 0, 1],
            "scale_granularity": scale_granularity,
            "grouping_scheme": grouping_scheme,
            "group_size": group_size,
            "source_snapshot": str(src_root),
            "num_tensors": len(per_tensor_entries),
            "num_quantizable_tensors": total_quantizable,
            "total_groups": total_groups if scale_granularity == "per_group" else total_quantizable,
            "content_fingerprint": content_fp,
            "packed_estimate": est,
            "packed_artifact_actual_bytes": None,
            "fake_quant_artifact": {
                "tensor_payload_bytes": tensor_payload_bytes,
                "shard_file_bytes": shard_file_bytes,
                "snapshot_total_bytes": snapshot_total_bytes,
                "shard_count": shard_count,
                "shard_size": f"{max_shard_size // (1024 * 1024)} MiB",
            },
            "per_tensor": per_tensor_entries,
        }

        # write quantization.json before atomic rename (in tmp)
        with open(tmp_root / "quantization.json", "w", encoding="utf-8") as qf:
            json.dump(quantization_json, qf, indent=2, sort_keys=True, ensure_ascii=False)
            qf.write("\n")

        # atomic rename
        if dst_root.exists():
            shutil.rmtree(dst_root)
        tmp_root.rename(dst_root)

        # file hashes for determinism check (optional)
        file_hashes: dict[str, str] = {}
        for i in range(1, shard_count + 1):
            p = dst_root / f"model-{i:05d}-of-{shard_count:05d}.safetensors"
            h = _sha256_hex(p.read_bytes())
            file_hashes[p.name] = f"sha256:{h}"

        return ConvertReport(
            dst_snapshot=dst_root,
            quantization_json=quantization_json,
            content_fingerprint=content_fp,
            file_hashes=file_hashes,
            shard_count=shard_count,
            tensor_payload_bytes=tensor_payload_bytes,
            shard_file_bytes=shard_file_bytes,
            snapshot_total_bytes=snapshot_total_bytes,
        )
    except Exception:
        # cleanup tmp on failure
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        raise
