"""Run the installed OpenTernary CLI against one explicit pretrained LLM.

This harness is intentionally separate from ``validate_cli_real.py``: it loads
real weights, can use substantial RAM/VRAM, and writes large artifacts.  Every
output directory must be new.  The frozen validation split is used for quality;
the test split is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def run_cli(root: Path, name: str, arguments: list[str], expected: tuple[int, ...] = (0,)) -> dict[str, Any]:
    command = [sys.executable, "-m", "openternary", *arguments, "--json"]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    (root / f"{name}.stdout.json").write_text(result.stdout, encoding="utf-8")
    (root / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    record = {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if result.returncode not in expected:
        raise RuntimeError(f"{name} exited {result.returncode}; see {name}.stderr.log")
    try:
        record["result"] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} did not emit one JSON document") from exc
    return record


def compare_snapshot_bytes(left: Path, right: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    def mapping(root: Path) -> dict[str, str]:
        indexes = list(root.glob("*.safetensors.index.json"))
        if len(indexes) != 1:
            raise ValueError(f"expected one safetensors index in {root}")
        return json.loads(indexes[0].read_text(encoding="utf-8"))["weight_map"]

    left_map, right_map = mapping(left), mapping(right)
    if set(left_map) != set(right_map):
        raise ValueError("restored snapshot tensor names differ")
    left_files: dict[str, Any] = {}
    right_files: dict[str, Any] = {}
    bytes_compared = 0
    try:
        for name in sorted(left_map):
            lf, rf = left_map[name], right_map[name]
            if lf not in left_files:
                left_files[lf] = safe_open(left / lf, framework="pt", device="cpu")
            if rf not in right_files:
                right_files[rf] = safe_open(right / rf, framework="pt", device="cpu")
            a, b = left_files[lf].get_tensor(name), right_files[rf].get_tensor(name)
            if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
                raise ValueError(f"restored snapshot metadata differs: {name}")
            raw_a = a.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            raw_b = b.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            if len(raw_a) != len(raw_b) or hashlib.sha256(raw_a).digest() != hashlib.sha256(raw_b).digest():
                raise ValueError(f"restored snapshot bytes differ: {name}")
            bytes_compared += len(raw_a)
    finally:
        left_files.clear()
        right_files.clear()
    return {"tensor_count": len(left_map), "bytes_compared": bytes_compared, "exact_match": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quality-data", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--skip-torchao", action="store_true")
    parser.add_argument("--require-wheel", action="store_true")
    args = parser.parse_args()
    model, quality_data, root = args.model.resolve(), args.quality_data.resolve(), args.root.resolve()
    if not model.is_dir() or not quality_data.is_file():
        parser.error("--model must be a local snapshot and --quality-data must be a file")
    if root.exists():
        parser.error("--root must not exist")
    if args.stride >= args.max_length:
        parser.error("--stride must be smaller than --max-length")
    spec = importlib.util.find_spec("openternary")
    package_path = Path(spec.origin).resolve() if spec and spec.origin else None
    if args.require_wheel and package_path and "site-packages" not in package_path.parts:
        parser.error(f"installed wheel required, got {package_path}")
    root.mkdir(parents=True)
    common = ["--device", args.device, "--dtype", args.dtype, "--seed", str(args.seed)]
    records: dict[str, Any] = {
        "schema_version": 1,
        "model": str(model),
        "quality_data": str(quality_data),
        "package": str(package_path),
        "test_split_opened": False,
        "steps": {},
    }
    try:
        records["steps"]["doctor"] = run_cli(root, "doctor", ["doctor", "--probe-runtime", "--device", args.device])
        records["steps"]["inspect"] = run_cli(
            root,
            "inspect",
            [
                "inspect",
                str(model),
                "--output",
                str(root / "inspect"),
                "--events-jsonl",
                str(root / "inspect.events.jsonl"),
            ],
        )
        records["steps"]["benchmark_baseline"] = run_cli(
            root,
            "benchmark-baseline",
            ["benchmark", str(model), *common, "--no-thinking", "--output", str(root / "benchmark-baseline")],
        )
        quality_args = [
            "--data",
            str(quality_data),
            "--split",
            "validation",
            "--max-length",
            str(args.max_length),
            "--stride",
            str(args.stride),
            *common,
        ]
        records["steps"]["quality_baseline"] = run_cli(
            root, "quality-baseline", ["quality", str(model), *quality_args, "--output", str(root / "quality-baseline")]
        )
        ternary_run = root / "quantize-ternary"
        ternary_artifact = ternary_run / "artifacts" / "snapshot"
        records["steps"]["quantize_ternary"] = run_cli(
            root,
            "quantize-ternary",
            [
                "quantize",
                str(model),
                "--backend",
                "ternary",
                "--scheme",
                "absmean",
                "--weight-dtype",
                "ternary",
                "--scale-granularity",
                "per_group",
                "--group-size",
                "128",
                "--device",
                "cpu",
                "--seed",
                str(args.seed),
                "--output",
                str(ternary_run),
                "--events-jsonl",
                str(root / "quantize-ternary.events.jsonl"),
            ],
        )
        records["steps"]["validate_ternary"] = run_cli(
            root, "validate-ternary", ["artifacts", "validate", str(ternary_artifact), "--level", "tensors"]
        )
        records["steps"]["benchmark_ternary"] = run_cli(
            root,
            "benchmark-ternary",
            ["benchmark", str(ternary_artifact), *common, "--no-thinking", "--output", str(root / "benchmark-ternary")],
        )
        records["steps"]["quality_ternary"] = run_cli(
            root,
            "quality-ternary",
            ["quality", str(ternary_artifact), *quality_args, "--output", str(root / "quality-ternary")],
        )
        records["steps"]["compare_ternary"] = run_cli(
            root,
            "compare-ternary",
            [
                "compare",
                str(root / "quality-baseline"),
                str(root / "quality-ternary"),
                "--output",
                str(root / "compare-ternary"),
            ],
        )
        packed, restored = root / "packed-ternary", root / "restored-safetensors"
        records["steps"]["pack"] = run_cli(
            root,
            "pack",
            [
                "export",
                str(ternary_artifact),
                "--format",
                "ternary-packed",
                "--output",
                str(packed),
                "--events-jsonl",
                str(root / "pack.events.jsonl"),
            ],
        )
        records["steps"]["validate_packed"] = run_cli(
            root, "validate-packed", ["artifacts", "validate", str(packed), "--level", "tensors"]
        )
        records["steps"]["restore"] = run_cli(
            root,
            "restore",
            [
                "export",
                str(packed),
                "--format",
                "safetensors",
                "--output",
                str(restored),
                "--events-jsonl",
                str(root / "restore.events.jsonl"),
            ],
        )
        records["steps"]["validate_restored"] = run_cli(
            root, "validate-restored", ["artifacts", "validate", str(restored), "--level", "tensors"]
        )
        records["packed_roundtrip"] = compare_snapshot_bytes(ternary_artifact, restored)
        records["steps"]["benchmark_restored"] = run_cli(
            root,
            "benchmark-restored",
            ["benchmark", str(restored), *common, "--no-thinking", "--output", str(root / "benchmark-restored")],
        )
        if not args.skip_torchao:
            torchao_run = root / "quantize-torchao"
            torchao_artifact = torchao_run / "artifacts" / "snapshot"
            records["steps"]["quantize_torchao"] = run_cli(
                root,
                "quantize-torchao",
                [
                    "quantize",
                    str(model),
                    "--backend",
                    "torchao",
                    "--scheme",
                    "int8-weight-only",
                    "--weight-dtype",
                    "int8",
                    "--scale-granularity",
                    "per_tensor",
                    "--device",
                    "cpu",
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(torchao_run),
                    "--events-jsonl",
                    str(root / "quantize-torchao.events.jsonl"),
                ],
            )
            records["steps"]["validate_torchao"] = run_cli(
                root, "validate-torchao", ["artifacts", "validate", str(torchao_artifact), "--level", "tensors"]
            )
            records["steps"]["benchmark_torchao"] = run_cli(
                root,
                "benchmark-torchao",
                [
                    "benchmark",
                    str(torchao_artifact),
                    *common,
                    "--no-thinking",
                    "--output",
                    str(root / "benchmark-torchao"),
                ],
            )
            records["steps"]["quality_torchao"] = run_cli(
                root,
                "quality-torchao",
                ["quality", str(torchao_artifact), *quality_args, "--output", str(root / "quality-torchao")],
            )
            records["steps"]["compare_torchao"] = run_cli(
                root,
                "compare-torchao",
                [
                    "compare",
                    str(root / "quality-baseline"),
                    str(root / "quality-torchao"),
                    "--output",
                    str(root / "compare-torchao"),
                ],
            )
        records["status"] = "completed"
    except BaseException as exc:
        records.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(root / "validation.json", records)


if __name__ == "__main__":
    main()
