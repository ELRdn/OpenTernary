"""CLI quantize tests — fake small snapshots, no 10GB."""

from __future__ import annotations

import json
import pathlib
import shutil
import uuid

import pytest

torch = pytest.importorskip("torch")
import safetensors.torch  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from openternary.cli.main import app  # noqa: E402

runner = CliRunner()


def _make_src(tmp: pathlib.Path) -> pathlib.Path:
    src = tmp / "src"
    src.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1},
    }
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (src / "tokenizer.json").write_text(json.dumps({}), encoding="utf-8")
    tensors = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(4, 8, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(tensors, str(src / "model.safetensors"))
    return src


def test_cli_quantize_dry_run() -> None:
    tmp = pathlib.Path.cwd() / f"test_q_dry_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = _make_src(tmp)
        result = runner.invoke(app, ["quantize", str(src), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "dry-run" in result.output.lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_quantize_per_tensor_and_per_group() -> None:
    tmp = pathlib.Path.cwd() / f"test_q_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = _make_src(tmp)
        out = tmp / "out_q"
        result = runner.invoke(app, ["quantize", str(src), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "quantization.json").exists()
        assert (out / "artifacts" / "snapshot" / "model.safetensors.index.json").exists()
        data = json.loads((out / "quantization.json").read_text(encoding="utf-8"))
        assert data["scale_granularity"] == "per_tensor"

        # per_group
        out2 = tmp / "out_q2"
        result2 = runner.invoke(
            app,
            ["quantize", str(src), "--output", str(out2), "--scale-granularity", "per_group", "--group-size", "2"],
        )
        assert result2.exit_code == 0, result2.output
        data2 = json.loads((out2 / "quantization.json").read_text(encoding="utf-8"))
        assert data2["scale_granularity"] == "per_group"
        assert data2["group_size"] == 2
        # content fingerprints should differ between per_tensor and per_group
        assert data["content_fingerprint"] != data2["content_fingerprint"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_quantize_cache_miss_exit_2() -> None:
    result = runner.invoke(app, ["quantize", "nonexistent/model-id-xyz-9999"])
    assert result.exit_code == 2
