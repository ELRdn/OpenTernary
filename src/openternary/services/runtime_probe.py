"""Explicit small-runtime probes; no model download, load, or repair."""

from __future__ import annotations

import time
from typing import Any

from openternary.services.errors import HardwareError


def probe_runtime(device: str = "cpu") -> dict[str, Any]:
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("runtime probe device must be cpu, cuda or mps")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise HardwareError("GPU runtime explicitly requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise HardwareError("MPS runtime explicitly requested but unavailable")
    resolved = {"cuda": "cuda:0", "mps": "mps:0"}.get(device, "cpu")
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                values = torch.arange(64, device=resolved, dtype=torch.float32).reshape(8, 8) / 64
                result = values.to(dtype) @ values.to(dtype).T
                if resolved.startswith("cuda"):
                    torch.cuda.synchronize()
                elif resolved.startswith("mps"):
                    torch.mps.synchronize()
                finite = bool(torch.isfinite(result).all().item())
                if not finite or str(result.device) != resolved or result.dtype != dtype:
                    raise RuntimeError("runtime probe device/dtype/finite check failed")
            results.append(
                {
                    "dtype": str(dtype),
                    "status": "passed",
                    "device": resolved,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )
        except (RuntimeError, NotImplementedError) as exc:
            results.append({"dtype": str(dtype), "status": "failed", "device": resolved, "reason": str(exc)})
    return {
        "scope": "synthetic_tensor",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": resolved,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "checks": results,
        "status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
        "native_low_bit_kernel": "not_probed",
        "model_validation": "not_run",
    }
