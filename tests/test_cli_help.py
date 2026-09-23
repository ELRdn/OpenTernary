"""CLI help and exit code tests — Phase 0 acceptance criteria."""

import json
import os
import pathlib
import struct
import subprocess
import sys

import pytest
from click import unstyle
from typer.testing import CliRunner

from openternary.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize("arguments", [["--help"], ["quantize", "--help"]])
def test_help_with_windows_legacy_output_encoding(arguments: list[str]) -> None:
    """Non-UTF-8 Windows output must not turn help into an encoding exception."""
    result = subprocess.run(
        [sys.executable, "-m", "openternary", *arguments],
        env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("cp1252", errors="replace")
    assert b"Usage:" in result.stdout


def test_main_help_shows_six_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["inspect", "quantize", "benchmark", "compare", "calibrate", "export"]:
        assert cmd in result.output, f"missing command {cmd}"


def test_inspect_help_shows_common_options() -> None:
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    for opt in ["--config", "--output", "--seed", "--device", "--dtype", "--dry-run", "--load-weights"]:
        assert opt in unstyle(result.output), f"missing option {opt}"


def test_quantize_help_shows_group_size() -> None:
    result = runner.invoke(app, ["quantize", "--help"])
    assert result.exit_code == 0
    assert "--group-size" in unstyle(result.output)


def test_dry_run_exit_0_no_side_effect() -> None:
    result = runner.invoke(app, ["inspect", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "No run directory was created" in result.output


def test_dry_run_with_config_and_group_size() -> None:
    result = runner.invoke(app, ["quantize", "--dry-run", "--group-size", "64", "--config", "configs/gemma4-e2b.yaml"])
    assert result.exit_code == 0
    assert "group_size: 64" in result.output


def test_inspect_normal_creates_run_exit_0(tmp_path: pathlib.Path) -> None:
    """The normal path is hermetic and must not depend on a populated HF cache."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "transformers_version": "5.6.2",
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 8,
        },
        "vision_config": {"hidden_size": 4, "num_hidden_layers": 1},
        "audio_config": {"hidden_size": 4, "num_hidden_layers": 1},
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [0, 32],
        },
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [8, 4],
            "data_offsets": [32, 96],
        },
    }
    header_json = json.dumps(header).encode("utf-8")
    with (snapshot / "model.safetensors").open("wb") as file:
        file.write(struct.pack("<Q", len(header_json)))
        file.write(header_json)
        file.write(b"\x00" * 96)

    out = tmp_path / "run_cli_test"
    result = runner.invoke(app, ["inspect", str(snapshot), "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert (out / "inspection.json").exists()
    assert (out / "metrics.json").exists()
    inspection = json.loads((out / "inspection.json").read_text(encoding="utf-8"))
    assert inspection["summary"]["total_tensors"] == 2


def test_quantize_now_implemented_dry_run() -> None:
    result = runner.invoke(app, ["quantize", "--dry-run"])
    assert result.exit_code == 0
    assert "quantize: dry-run" in result.output.lower()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "openternary" in result.output.lower()


def test_cache_info() -> None:
    result = runner.invoke(app, ["cache-info"])
    assert result.exit_code == 0
    assert "Cache info" in result.output
