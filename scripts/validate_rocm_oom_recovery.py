"""Bounded real-ROCm OOM and post-OOM kernel/optimizer recovery probe."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.03)
    parser.add_argument("--block-mb", type=int, default=16)
    parser.add_argument("--max-probe-mb", type=int, default=1024)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    if not 0.005 <= args.fraction <= 0.25:
        raise ValueError("--fraction must be in [0.005, 0.25]")
    if args.block_mb < 1 or args.max_probe_mb < args.block_mb:
        raise ValueError("invalid bounded allocation sizes")

    import torch

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RuntimeError("this probe requires a real ROCm device")

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(args.fraction, 0)
    free_before, total = torch.cuda.mem_get_info(0)
    device_name = torch.cuda.get_device_name(0)
    blocks: list[torch.Tensor] = []
    allocated_probe_bytes = 0
    oom: BaseException | None = None
    block_bytes = args.block_mb * 1024 * 1024
    max_probe_bytes = args.max_probe_mb * 1024 * 1024
    try:
        while allocated_probe_bytes + block_bytes <= max_probe_bytes:
            block = torch.empty(block_bytes, dtype=torch.uint8, device="cuda:0")
            block.fill_(1)
            blocks.append(block)
            allocated_probe_bytes += block_bytes
            torch.cuda.synchronize()
    except BaseException as exc:  # torch OOM type differs between builds
        oom = exc
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if oom is None:
        report = {
            "status": "fail",
            "reason": "bounded allocation completed without a real OOM",
            "allocated_probe_bytes": allocated_probe_bytes,
        }
        _atomic_json(args.output, report)
        return 2
    if not isinstance(oom, torch.OutOfMemoryError) and "out of memory" not in str(oom).lower():
        raise oom

    free_after_cleanup, _ = torch.cuda.mem_get_info(0)
    parameter = torch.nn.Parameter(torch.ones((256, 256), device="cuda:0", dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    left = torch.randn((256, 256), device="cuda:0", dtype=torch.bfloat16)
    right = torch.randn((256, 256), device="cuda:0", dtype=torch.bfloat16)
    loss = (left @ right).float().square().mean() + parameter.square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    recovered = bool(torch.isfinite(loss).item() and torch.isfinite(parameter).all().item())
    peak = int(torch.cuda.max_memory_allocated(0))

    report = {
        "schema_version": 1,
        "status": "pass" if recovered else "fail",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "device": device_name,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "fraction": args.fraction,
        "block_mb": args.block_mb,
        "max_probe_mb": args.max_probe_mb,
        "total_bytes": int(total),
        "free_before_bytes": int(free_before),
        "allocated_before_oom_bytes": allocated_probe_bytes,
        "oom_type": type(oom).__name__,
        "oom_message": str(oom),
        "free_after_cleanup_bytes": int(free_after_cleanup),
        "post_oom_bf16_backward": recovered,
        "post_oom_adam_step": recovered,
        "peak_allocated_bytes": peak,
        "scientific_acceptance": False,
        "scope": "allocator and post-OOM runtime recovery; not calibration transaction equivalence",
    }
    _atomic_json(args.output, report)
    return 0 if recovered else 3


if __name__ == "__main__":
    sys.exit(main())
