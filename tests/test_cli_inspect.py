"""CLI inspect integration tests — uses fake snapshots, no 10GB needed."""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import uuid

from typer.testing import CliRunner

from openternary.cli.main import app

runner = CliRunner()


def _write_fake_snapshot(base: pathlib.Path) -> pathlib.Path:
    snapshot = base / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "transformers_version": "5.6.2",
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "vocab_size": 262144,
        },
        "vision_config": {"hidden_size": 768, "num_hidden_layers": 16},
        "audio_config": {"hidden_size": 1024, "num_hidden_layers": 12},
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header: dict[str, object] = {}
    header["model.layers.0.self_attn.q_proj.weight"] = {
        "dtype": "BF16",
        "shape": [1536, 1536],
        "data_offsets": [0, 4718592],
    }
    header["model.embed_tokens.weight"] = {
        "dtype": "BF16",
        "shape": [262144, 1536],
        "data_offsets": [4718592, 810000000],
    }
    # Use small payload for test speed — override offsets to small
    header = {
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [8, 4], "data_offsets": [32, 96]},
    }
    header_json = json.dumps(header).encode("utf-8")
    path = snapshot / "model.safetensors"
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * 96)
    return snapshot


def test_cli_inspect_with_local_snapshot() -> None:
    tmp = pathlib.Path.cwd() / f"test_cli_inspect_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_fake_snapshot(tmp)
        out = tmp / "out_inspect"
        result = runner.invoke(app, ["inspect", str(snapshot), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "inspection.json").exists()
        data = json.loads((out / "inspection.json").read_text(encoding="utf-8"))
        assert "inspection_fingerprint" in data
        assert data["summary"]["total_tensors"] == 2
        assert isinstance(data["memory_estimate"]["ternary"], dict)
        assert "packed_weight_bits" in data["memory_estimate"]["ternary"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_inspect_dry_run_no_output() -> None:
    tmp = pathlib.Path.cwd() / f"test_cli_dry_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_fake_snapshot(tmp)
        result = runner.invoke(app, ["inspect", str(snapshot), "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()
        # no runs created by dry-run
        assert not (tmp / "out_inspect").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_inspect_output_unified_exact_path() -> None:
    """--output指定時はそのpathそのままに生成される."""
    tmp = pathlib.Path.cwd() / f"test_cli_out_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_fake_snapshot(tmp)
        out = tmp / "runs" / "inspect-smoke"
        result = runner.invoke(app, ["inspect", str(snapshot), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "inspection.json").exists()
        # not nested date dir
        assert out.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_inspect_cache_miss_exit_2() -> None:
    result = runner.invoke(app, ["inspect", "nonexistent/model-id-xyz-9999"])
    assert result.exit_code == 2
    assert "not found" in result.output.lower() or "snapshot" in result.output.lower()
