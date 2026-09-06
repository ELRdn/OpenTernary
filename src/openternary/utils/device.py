"""Runtime backend detection and VRAM budgeting — Phase 4.2-G AMD GPU."""

from __future__ import annotations

import dataclasses

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class BackendInfo:
    """Detected runtime backend."""

    backend: str  # "cpu" | "cuda-nvidia" | "cuda-rocm"
    device_name: str
    total_bytes: int
    free_bytes: int
    is_hip: bool
    is_cuda: bool
    bf16_supported: bool


def detect_backend() -> BackendInfo:
    """Detect runtime backend without importing heavy ROCm APIs.

    PyTorch ROCm も API 上は torch.cuda を使用するため、backend 識別は
    torch.version.hip / torch.version.cuda / torch.cuda.get_device_name で行う。
    新たに rocm device type を tensor へ直接渡さない。
    """
    if torch is None:
        return BackendInfo(
            backend="cpu",
            device_name="cpu",
            total_bytes=0,
            free_bytes=0,
            is_hip=False,
            is_cuda=False,
            bf16_supported=False,
        )
    # allow force for testing microbatch shrink without real GPU
    import os

    force = os.environ.get("OT_FORCE_ROCM") or os.environ.get("OT_FORCE_GPU")
    if force == "1":
        # simulate RX7600 8GB for gate testing on CPU-only host
        return BackendInfo(
            backend="cuda-rocm",
            device_name="AMD Radeon RX 7600 / gfx1102 (simulated)",
            total_bytes=8 * 1024**3,
            free_bytes=6 * 1024**3,
            is_hip=True,
            is_cuda=False,
            bf16_supported=True,
        )

    try:
        is_cuda_available = bool(torch.cuda.is_available())
    except Exception:
        is_cuda_available = False

    if not is_cuda_available:
        return BackendInfo(
            backend="cpu",
            device_name="cpu",
            total_bytes=0,
            free_bytes=0,
            is_hip=False,
            is_cuda=False,
            bf16_supported=False,
        )

    is_hip = bool(getattr(torch.version, "hip", None) is not None)  # type: ignore[union-attr]
    is_cuda = bool(getattr(torch.version, "cuda", None) is not None)  # type: ignore[union-attr]
    try:
        dev_name = str(torch.cuda.get_device_name(0))  # type: ignore[union-attr]
    except Exception:
        dev_name = "cuda:0"

    # BF16 capability: ROCm 7 may have limited kernels, but torch.cuda.is_bf16_supported exists in newer torch
    bf16_ok = False
    try:
        # torch.cuda.is_bf16_supported is available in torch>=2.0 for Ampere+ / RDNA
        if hasattr(torch.cuda, "is_bf16_supported"):
            bf16_ok = bool(torch.cuda.is_bf16_supported())  # type: ignore[union-attr]
        else:
            # fallback: try small bf16 matmul on gpu
            bf16_ok = "gfx" in dev_name.lower() or "Radeon" in dev_name or is_hip
    except Exception:
        bf16_ok = False

    # free/total via mem_get_info
    free_b = 0
    total_b = 0
    try:
        free_b, total_b = torch.cuda.mem_get_info(0)  # type: ignore[union-attr]
    except Exception:
        try:
            free_b, total_b = torch.cuda.mem_get_info()  # type: ignore[union-attr]
        except Exception:
            free_b, total_b = 0, 0

    if is_hip:
        backend = "cuda-rocm"
    elif is_cuda:
        backend = "cuda-nvidia"
    else:
        # cuda available but neither hip nor cuda version? fallback to nvidia
        backend = "cuda-nvidia"

    # Optional: detect RX7600/gfx1102 via device name for observability, but backend stays cuda-rocm
    return BackendInfo(
        backend=backend,
        device_name=dev_name,
        total_bytes=int(total_b),
        free_bytes=int(free_b),
        is_hip=is_hip,
        is_cuda=is_cuda,
        bf16_supported=bf16_ok,
    )


@dataclasses.dataclass(frozen=True)
class VramBudget:
    """Dynamic VRAM budget per module."""

    total_bytes: int
    free_bytes: int
    budget_bytes: int
    reserved_bytes: int
    min_free_bytes: int


def compute_vram_budget(
    free_bytes: int,
    total_bytes: int,
    max_fraction: float = 0.80,
    reserve_mb: int = 1024,
    min_free_mb: int = 768,
) -> VramBudget:
    """Compute per-module GPU budget from observed free/total.

    - free/total は torch.cuda.mem_get_info() の生値（Windows desktop 分を含む）
    - reserve_mb: デスクトップ等が常時使用する分を必ず残す
    - min_free_mb: 確保後も最低残す free
    - max_fraction: プロセスが占有してよい total の上限（0.80）

    budget = min(free - reserve, free - min_free, free - total*(1-max_fraction))
    下限は 0。
    """
    if total_bytes <= 0 or free_bytes <= 0:
        return VramBudget(
            total_bytes=total_bytes,
            free_bytes=free_bytes,
            budget_bytes=0,
            reserved_bytes=reserve_mb * 1024 * 1024,
            min_free_bytes=min_free_mb * 1024 * 1024,
        )
    reserve_b = int(reserve_mb * 1024 * 1024)
    min_free_b = int(min_free_mb * 1024 * 1024)
    # max_fraction 制約: used = total - free, budget <= total*max_fraction - used
    max_frac_budget = int(free_bytes - total_bytes * (1.0 - max_fraction))
    candidates = [
        int(free_bytes - reserve_b),
        int(free_bytes - min_free_b),
        int(max_frac_budget),
    ]
    budget = int(min(candidates))
    if budget < 0:
        budget = 0
    return VramBudget(
        total_bytes=int(total_bytes),
        free_bytes=int(free_bytes),
        budget_bytes=int(budget),
        reserved_bytes=reserve_b,
        min_free_bytes=min_free_b,
    )


def format_bytes(n: int) -> str:
    """Human readable bytes."""
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(f) < 1024.0:
            return f"{f:.1f}{unit}"
        f /= 1024.0
    return f"{f:.1f}TB"


def is_oom_error(exc: BaseException) -> bool:
    """Check if exception is CUDA OOM."""
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or "memory" in msg and "allocate" in msg
