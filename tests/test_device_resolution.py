"""Device resolution tests — CPU-only auto must not disk-offload."""

from __future__ import annotations

import sys
import types


def _install_fake_cuda_unavailable() -> types.ModuleType:
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    fake_torch.float16 = "fp16"  # type: ignore[attr-defined]
    fake_torch.float32 = "fp32"  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, synchronize=lambda: None, manual_seed_all=lambda s: None
    )  # type: ignore[attr-defined]

    class _InfCtx:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: object) -> None:
            return None

    fake_torch.inference_mode = lambda: _InfCtx()  # type: ignore[attr-defined]
    fake_torch.manual_seed = lambda seed: None  # type: ignore[attr-defined]
    fake_torch.Tensor = object  # type: ignore[attr-defined]
    sys.modules["torch"] = fake_torch
    return fake_torch


def test_device_auto_resolves_to_cpu_when_cuda_unavailable() -> None:
    orig = sys.modules.get("torch")
    try:
        _install_fake_cuda_unavailable()
        from openternary.benchmark.runner import _resolve_device_map

        m = _resolve_device_map("auto")
        # CPU-only環境では {"": "cpu"} を返し、 "auto" のままdisk offloadさせない
        assert m == {"": "cpu"}
        assert m != "auto"
    finally:
        if orig is not None:
            sys.modules["torch"] = orig
        else:
            sys.modules.pop("torch", None)


def test_device_auto_resolves_to_auto_when_cuda_available() -> None:
    orig = sys.modules.get("torch")
    try:
        fake = types.ModuleType("torch")
        fake.bfloat16 = "bf16"  # type: ignore[attr-defined]
        fake.float16 = "fp16"  # type: ignore[attr-defined]
        fake.float32 = "fp32"  # type: ignore[attr-defined]
        fake.cuda = types.SimpleNamespace(
            is_available=lambda: True, synchronize=lambda: None, manual_seed_all=lambda s: None
        )  # type: ignore[attr-defined]
        fake.inference_mode = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)  # type: ignore[attr-defined]
        fake.manual_seed = lambda seed: None  # type: ignore[attr-defined]
        sys.modules["torch"] = fake
        from openternary.benchmark.runner import _resolve_device_map

        assert _resolve_device_map("auto") == "auto"
        assert _resolve_device_map("cpu") == {"": "cpu"}
    finally:
        if orig is not None:
            sys.modules["torch"] = orig
        else:
            sys.modules.pop("torch", None)


def test_failed_benchmark_writes_metrics_status_failed() -> None:
    """Failed benchmark は metrics.json status=failed を残す（QR-02）."""
    import json
    import pathlib
    import shutil
    import uuid

    from typer.testing import CliRunner

    from openternary.cli.main import app

    runner = CliRunner()

    # snapshot は存在するが torch を欠落させる → ImportError path
    tmp = pathlib.Path.cwd() / f"test_failed_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    # 一時的な torch 欠落を演出: sys.modules から torch を退避し import 失敗させる
    # ここでは存在しないモデルIDで FileNotFound path を使えば確実に失敗する
    try:
        out = tmp / "out_failed"
        result = runner.invoke(app, ["benchmark", "nonexistent/model-id-xyz", "--output", str(out)])
        assert result.exit_code == 2
        # run_dir が作成され metrics.json が failed で残ること
        assert out.exists()
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["status"] == "failed"
        assert "error_type" in metrics
        assert "error_message" in metrics
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
