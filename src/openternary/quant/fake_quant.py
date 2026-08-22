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


def _atomic_rename(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Windows でも PermissionError を避けるためリトライ付きで tmp->dst を rename."""
    import time

    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    # retry for antivirus / file handle delay on Windows
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            src.rename(dst)
            return
        except PermissionError as e:
            last_exc = e
            time.sleep(0.05 * (attempt + 1))
        except OSError:
            # fallback to shutil.move
            try:
                shutil.move(str(src), str(dst))
                return
            except Exception as e2:
                last_exc = e2
                time.sleep(0.05 * (attempt + 1))
    # final attempt with shutil.move
    try:
        shutil.move(str(src), str(dst))
        return
    except Exception:
        pass
    if last_exc:
        raise last_exc


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
        _atomic_rename(tmp_root, dst_root)

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


def _collect_safetensors_files(src_root: pathlib.Path) -> list[pathlib.Path]:
    """src 内の全 .safetensors を収集（single/sharded 両対応、sorted）。"""
    files = sorted(src_root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors file found in {src_root}")
    return files


def _get_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return -1.0


def _materialize_log(msg: str) -> None:
    import sys

    print(msg, file=sys.stderr, flush=True)
    # also try logging if available
    try:
        import logging

        logging.getLogger("openternary.materialize").info(msg)
    except Exception:
        pass


def materialize_calibrated_snapshot(
    src_snapshot: pathlib.Path | str,
    dst_snapshot: pathlib.Path | str,
    app_config: Any,
    calibrated_state: dict[str, Any],
    max_shard_size: int = MAX_SHARD_SIZE,
) -> ConvertReport:
    """Calibrated scales からの bounded sharded snapshot materialize.

    Phase 4.1 real-model 用。Source を shard 単位で stream し、
    calibrated_state に含まれる quantized target は `codes*scale` を
    `dequantize`/`dequantize_groupwise` で再構成、非 target は source を
    そのままコピーする。original weight shard の丸ごとコピーは行わない。

    Args:
        src_snapshot: Teacher HF snapshot dir
        dst_snapshot: 出力先 dir（tmp 経由で atomic に作成）
        app_config: AppConfig（scale_granularity / group_size / grouping_scheme 参照）
        calibrated_state: {tensor_name: {codes, scales, shape, orig_dtype, group_size, grouping_scheme}}
        max_shard_size: shard 閾値

    Returns:
        ConvertReport
    """
    _require_torch()
    import gc

    import safetensors.torch

    src_root = pathlib.Path(src_snapshot).resolve()
    dst_root = pathlib.Path(dst_snapshot).resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"src_snapshot not found: {src_root}")

    tmp_root = dst_root.parent / (dst_root.name + ".tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        _materialize_log(f"[materialize] copy non-weight files: {src_root} -> {tmp_root}")
        _copy_non_weight_files(src_root, tmp_root)
        _materialize_log(
            f"[materialize] non-weight copy done, tmp size ~{sum(p.stat().st_size for p in tmp_root.rglob('*') if p.is_file()) / (1024**3):.3f} GB, rss={_get_rss_mb():.1f} MB"
        )

        from safetensors import safe_open

        buffer: dict[str, Any] = {}
        current_bytes = 0
        shard_index = 1
        weight_map: dict[str, str] = {}
        content_entries: list[dict[str, Any]] = []
        per_tensor_entries: list[dict[str, Any]] = []
        tensor_payload_bytes = 0
        total_quantizable = 0
        total_groups = 0

        scale_granularity = str(getattr(app_config.quantization, "scale_granularity", "per_tensor"))
        group_size_cfg = int(getattr(app_config.quantization, "group_size", 128))
        grouping_scheme_cfg = str(getattr(app_config.quantization, "grouping_scheme", "last-dim-rowwise-v1"))

        from openternary.quant.grouping import GroupwiseResult, dequantize_groupwise  # noqa: WPS433
        from openternary.quant.ternary import TernaryTensor  # noqa: WPS433
        from openternary.quant.ternary import dequantize as dequantize_per_tensor  # noqa: WPS433

        st_files = _collect_safetensors_files(src_root)
        _materialize_log(
            f"[materialize] source shards: {len(st_files)} files, max_shard={max_shard_size // (1024 * 1024)} MiB, scale_granularity={scale_granularity}, group_size={group_size_cfg}"
        )
        for _sf in st_files:
            _materialize_log(f"[materialize] source shard: {_sf.name} size={_sf.stat().st_size / (1024**3):.3f} GB")
        # For sharded source, process shard-by-shard to keep memory bounded and avoid reopening per tensor
        # Collect all tensor names per shard in order, but global order is sorted for determinism
        # We will iterate shards in sorted order, and within each shard iterate sorted keys. Then we sort buffer keys at flush time.
        # To guarantee deterministic global order, we also collect all_names sorted but process via per-shard open.
        all_shard_keys: list[tuple[pathlib.Path, list[str]]] = []
        for stf in st_files:
            with safe_open(str(stf), framework="pt", device="cpu") as f:
                keys = sorted(f.keys())
                all_shard_keys.append((stf, keys))
        total_tensors = sum(len(k) for _, k in all_shard_keys)
        _materialize_log(f"[materialize] total tensors to materialize: {total_tensors}")

        # Helper to process one tensor with logging and memory hygiene
        tensor_idx = 0

        for stf, keys in all_shard_keys:
            _materialize_log(
                f"[materialize] opening source shard {stf.name} with {len(keys)} tensors, rss={_get_rss_mb():.1f} MB"
            )
            with safe_open(str(stf), framework="pt", device="cpu") as f:
                for name in keys:
                    tensor_idx += 1
                    try:
                        _materialize_log(
                            f"[materialize] [{tensor_idx}/{total_tensors}] start name={name} shard={stf.name} buffer_bytes={current_bytes} shard_idx={shard_index} rss={_get_rss_mb():.1f} MB"
                        )
                        tensor = f.get_tensor(name)
                        # Dtype/shape info before processing
                        dtype_str_before = str(tensor.dtype)
                        shape_before = list(tensor.shape)
                        nbytes_before = int(tensor.numel() * tensor.element_size())
                        _materialize_log(
                            f"[materialize]   before dequantize: shape={shape_before} dtype={dtype_str_before} bytes={nbytes_before}"
                        )
                    except Exception as e:
                        _materialize_log(f"[materialize] ERROR fetching tensor {name} from {stf}: {e}")
                        raise
                    # from here, wrap per-tensor processing in try to log which tensor caused crash
                    try:
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
                        orig_dtype_s = dtype_map_rev.get(str(tensor.dtype), str(tensor.dtype))
                        shape = list(tensor.shape)
                        param_count = int(tensor.numel())
                        role, quantizable_by_adapter, exclude_reason = _classify_tensor(
                            name, shape, orig_dtype_s, app_config
                        )

                        calibrated_entry = calibrated_state.get(name)
                        if calibrated_entry is not None:
                            total_quantizable += 1
                            codes = calibrated_entry["codes"]
                            scales = calibrated_entry["scales"]
                            if not hasattr(scales, "numel"):
                                scales_t = torch.tensor([float(scales)], dtype=torch.float32)
                            else:
                                scales_t = scales.detach().cpu().to(torch.float32).contiguous()
                                if scales_t.dim() == 0:
                                    scales_t = scales_t.view(1)
                            num_groups = int(scales_t.numel())
                            total_groups += num_groups
                            scale_fp = _scale_fingerprint(scales_t)
                            if num_groups > 0:
                                s = scales_t.float()
                                scale_min = float(s.min().item())
                                scale_max = float(s.max().item())
                                scale_mean = float(s.mean().item())
                                if num_groups == 1:
                                    scale_std = 0.0
                                else:
                                    scale_std = float(s.std(correction=0).item())
                                    if math.isnan(scale_std):
                                        scale_std = 0.0
                            else:
                                scale_min = scale_max = scale_mean = scale_std = 0.0

                            if scale_granularity == "per_tensor":
                                scale_val = float(scales_t[0].item()) if num_groups else 0.0
                                tt = TernaryTensor(
                                    codes=codes.detach().cpu().to(torch.int8),
                                    scale=scale_val,
                                    shape=tuple(codes.shape),
                                    orig_dtype=str(tensor.dtype),
                                )
                                _materialize_log(f"[materialize]   dequant per_tensor scale={scale_val:.6f}")
                                recon = dequantize_per_tensor(tt)
                                # free tt.codes reference early for giant tensors
                                del tt
                                recon_cast = recon.to(tensor.dtype)
                                # free recon (float32) if not needed after cast
                                del recon
                                gc.collect()
                                out_tensor = recon_cast
                                del recon_cast
                                zero_ratio = float((codes == 0).sum().item() / codes.numel()) if codes.numel() else 0.0
                                try:
                                    # Use float32 tensor without extra copy: recon already freed, compute mae via out_tensor
                                    mae = float(
                                        (tensor.to(torch.float32) - out_tensor.to(torch.float32)).abs().mean().item()
                                    )
                                except Exception:
                                    mae = 0.0
                            else:
                                g_size = int(calibrated_entry.get("group_size", group_size_cfg))
                                g_scheme = str(calibrated_entry.get("grouping_scheme", grouping_scheme_cfg))
                                res = GroupwiseResult(
                                    codes=codes.detach().cpu().to(torch.int8),
                                    scales=scales_t,
                                    shape=tuple(shape),
                                    orig_dtype=str(tensor.dtype),
                                    group_size=g_size,
                                    grouping_scheme=g_scheme,
                                )
                                _materialize_log(
                                    f"[materialize]   dequant per_group groups={num_groups} g_size={g_size}"
                                )
                                recon = dequantize_groupwise(res)
                                del res
                                recon_cast = recon.to(tensor.dtype)
                                del recon
                                gc.collect()
                                out_tensor = recon_cast
                                del recon_cast
                                zero_ratio = float((codes == 0).sum().item() / codes.numel()) if codes.numel() else 0.0
                                try:
                                    mae = float(
                                        (tensor.to(torch.float32) - out_tensor.to(torch.float32)).abs().mean().item()
                                    )
                                except Exception:
                                    mae = 0.0
                            # Free calibrated tensors refs early
                            del scales_t
                            _materialize_log(
                                f"[materialize]   after dequantize: out shape={list(out_tensor.shape)} dtype={out_tensor.dtype} rss={_get_rss_mb():.1f} MB"
                            )
                            entry: dict[str, Any] = {
                                "name": name,
                                "shape": shape,
                                "dtype": str(out_tensor.dtype),
                                "param_count": param_count,
                                "quantizable": True,
                                "role": role,
                                "exclude_reason": None,
                                "scale_granularity": scale_granularity,
                                "grouping_scheme": grouping_scheme_cfg
                                if scale_granularity == "per_group"
                                else "last-dim-rowwise-v1",
                                "group_size": group_size_cfg if scale_granularity == "per_group" else 128,
                                "num_groups": num_groups,
                                "scale_fingerprint": scale_fp,
                                "scale_stats": {
                                    "min": scale_min,
                                    "max": scale_max,
                                    "mean": scale_mean,
                                    "std": scale_std,
                                },
                                "zero_ratio": zero_ratio,
                                "mae": mae,
                            }
                            # free codes ref after use (calibrated_state still holds original, but local codes var freed)
                            del codes
                            del scales
                        else:
                            out_tensor = tensor
                            entry = {
                                "name": name,
                                "shape": shape,
                                "dtype": str(out_tensor.dtype),
                                "param_count": param_count,
                                "quantizable": False,
                                "role": role,
                                "exclude_reason": exclude_reason,
                                "scale_granularity": "per_tensor",
                                "grouping_scheme": "last-dim-rowwise-v1",
                                "group_size": 128,
                                "num_groups": 0,
                                "scale_fingerprint": "",
                                "scale_stats": {"min": 0, "max": 0, "mean": 0, "std": 0},
                                "zero_ratio": 0.0,
                                "mae": 0.0,
                            }
                    except Exception as exc:
                        _materialize_log(f"[materialize] ERROR processing tensor {name} shape={shape} : {exc}")
                        raise

                    # ---- content hash with memory hygiene ----
                    try:
                        # For giant tensors, hash without holding extra duplicate: compute and immediately free bytes
                        # _tensor_raw_sha256 internally does view->numpy->tobytes which copies 800MB for embed; we keep but log and free quickly
                        _materialize_log(f"[materialize]   hashing tensor {name} ...")
                        sha = _tensor_raw_sha256(out_tensor)
                        _materialize_log(f"[materialize]   hash done {sha[:12]}...")
                    except Exception as exc2:
                        _materialize_log(f"[materialize] ERROR hashing {name}: {exc2}")
                        raise

                    per_tensor_entries.append(entry)
                    nbytes = int(out_tensor.numel() * out_tensor.element_size())
                    tensor_payload_bytes += nbytes
                    content_entries.append(
                        {"name": name, "dtype": str(out_tensor.dtype), "shape": shape, "sha256": sha}
                    )
                    _materialize_log(
                        f"[materialize]   buffer before add: current={current_bytes} + nbytes={nbytes} -> {current_bytes + nbytes} / max {max_shard_size} rss={_get_rss_mb():.1f} MB"
                    )

                    if buffer and current_bytes + nbytes > max_shard_size:
                        shard_name = f"model-{shard_index:05d}-of-99999.safetensors"
                        shard_path = tmp_root / shard_name
                        ordered = {k: buffer[k] for k in sorted(buffer.keys())}
                        _materialize_log(
                            f"[materialize]   flushing shard {shard_name} with {len(ordered)} tensors, bytes={current_bytes} rss={_get_rss_mb():.1f} MB"
                        )
                        safetensors.torch.save_file(ordered, str(shard_path))
                        _materialize_log(f"[materialize]   flushed {shard_path} size={shard_path.stat().st_size} bytes")
                        for k in ordered:
                            weight_map[k] = shard_name
                        # free buffer tensors explicitly
                        for _k in list(buffer.keys()):
                            del buffer[_k]
                        buffer.clear()
                        gc.collect()
                        current_bytes = 0
                        shard_index += 1
                        _materialize_log(f"[materialize]   after flush rss={_get_rss_mb():.1f} MB")
                    buffer[name] = out_tensor
                    current_bytes += nbytes
                    # Free original tensor reference if different from out_tensor, to avoid duplicate for non-target
                    if tensor is not out_tensor:
                        del tensor
                    # out_tensor now owned by buffer, will be freed on flush
                    _materialize_log(
                        f"[materialize] [{tensor_idx}/{total_tensors}] done name={name} buffer={current_bytes} rss={_get_rss_mb():.1f} MB"
                    )
                    # Periodic gc for giant tensors
                    if nbytes > 100 * 1024 * 1024:
                        gc.collect()

                # end for name in keys
            _materialize_log(f"[materialize] finished shard {stf.name}")

        _materialize_log(
            f"[materialize] all tensors processed, flushing remaining buffer {len(buffer)} tensors, bytes={current_bytes}"
        )

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
            shard_index -= 1
        shard_count = shard_index if shard_index >= 1 else 0

        if shard_count > 0:
            for i in range(1, shard_count + 1):
                old = tmp_root / f"model-{i:05d}-of-99999.safetensors"
                new = tmp_root / f"model-{i:05d}-of-{shard_count:05d}.safetensors"
                old.rename(new)
                old_name = f"model-{i:05d}-of-99999.safetensors"
                new_name = f"model-{i:05d}-of-{shard_count:05d}.safetensors"
                for k, v in list(weight_map.items()):
                    if v == old_name:
                        weight_map[k] = new_name

        total_size = tensor_payload_bytes
        index_obj = {
            "metadata": {"total_size": total_size},
            "weight_map": {k: weight_map[k] for k in sorted(weight_map.keys())},
        }
        with open(tmp_root / "model.safetensors.index.json", "w", encoding="utf-8") as idx_f:
            json.dump(index_obj, idx_f, indent=2, sort_keys=True, ensure_ascii=False)
            idx_f.write("\n")

        content_fp = _content_fingerprint(content_entries)
        shard_file_bytes = 0
        for i in range(1, shard_count + 1):
            p = tmp_root / f"model-{i:05d}-of-{shard_count:05d}.safetensors"
            shard_file_bytes += p.stat().st_size
        snapshot_total_bytes = 0
        for p in tmp_root.rglob("*"):
            if p.is_file():
                snapshot_total_bytes += p.stat().st_size

        from openternary.quant.accounting import estimate_whole_model

        classified = []
        for ent in per_tensor_entries:
            dtype_str = ent["dtype"]
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
                    "name": ent["name"],
                    "param_count": ent["param_count"],
                    "dtype": header_dtype,
                    "quantizable": ent["quantizable"],
                }
            )

        try:
            est = estimate_whole_model(classified, scale_dtype="fp32")  # type: ignore[arg-type]
            if scale_granularity == "per_group":
                grouped_overhead = total_groups * 32
                # keep per_tensor for reference
                est["scale_overhead_bits"] = sum(1 for e in per_tensor_entries if e["quantizable"]) * 32
                est["grouped_scale_overhead_bits"] = grouped_overhead
                est["num_scales"] = total_groups
                est["num_quantizable_tensors"] = sum(1 for e in per_tensor_entries if e["quantizable"])
        except Exception:
            total_params = sum(e["param_count"] for e in per_tensor_entries if e["quantizable"])
            from openternary.quant.accounting import LOG2_3, packed_weight_bits_for_n

            ideal = int(total_params * LOG2_3)
            packed = sum(packed_weight_bits_for_n(e["param_count"]) for e in per_tensor_entries if e["quantizable"])
            logical = sum(e["param_count"] * 2 for e in per_tensor_entries if e["quantizable"])
            per_tensor_overhead = sum(1 for e in per_tensor_entries if e["quantizable"]) * 32
            grouped_overhead = total_groups * 32 if scale_granularity == "per_group" else per_tensor_overhead
            est = {
                "ideal_ternary_bits": ideal,
                "packed_weight_bits": packed,
                "logical_2bit_bits": logical,
                "scale_overhead_bits": per_tensor_overhead,
                "grouped_scale_overhead_bits": grouped_overhead,
                "num_scales": total_groups
                if scale_granularity == "per_group"
                else sum(1 for e in per_tensor_entries if e["quantizable"]),
            }

        quantization_json: dict[str, Any] = {
            "version": 1,
            "method": "naive-calibrated",
            "codebook": [-1, 0, 1],
            "scale_granularity": scale_granularity,
            "grouping_scheme": grouping_scheme_cfg,
            "group_size": group_size_cfg,
            "source_snapshot": str(src_root),
            "num_tensors": len(per_tensor_entries),
            "num_quantizable_tensors": sum(1 for e in per_tensor_entries if e["quantizable"]),
            "total_groups": total_groups,
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

        with open(tmp_root / "quantization.json", "w", encoding="utf-8") as qf:
            json.dump(quantization_json, qf, indent=2, sort_keys=True, ensure_ascii=False)
            qf.write("\n")

        _atomic_rename(tmp_root, dst_root)

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
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        raise
