"""Config loader tests."""

import pathlib

import pytest

from openternary.config.loader import dump_config_yaml, load_config


def test_load_defaults() -> None:
    cfg = load_config()
    assert cfg.model.id == "google/gemma-4-E2B-it-qat-q4_0-unquantized"
    assert cfg.quantization.group_size == 128
    assert cfg.seed == 42


def test_load_yaml_config() -> None:
    cfg = load_config(config_path=pathlib.Path("configs/gemma4-e2b.yaml"))
    assert cfg.model.id == "google/gemma-4-E2B-it-qat-q4_0-unquantized"
    assert cfg.quantization.group_size == 128
    assert cfg.benchmark.suite == "smoke"
    assert cfg.benchmark.thinking is False
    assert cfg.benchmark.generation.max_new_tokens == 64


def test_cli_overrides_take_precedence() -> None:
    cfg = load_config(
        config_path=pathlib.Path("configs/gemma4-e2b.yaml"),
        cli_overrides={"seed": 123, "quantization.group_size": 64, "model.id": "custom/model"},
    )
    assert cfg.seed == 123
    assert cfg.quantization.group_size == 64
    assert cfg.model.id == "custom/model"


def test_invalid_group_size_raises() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        load_config(cli_overrides={"quantization.group_size": -1})


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(config_path=pathlib.Path("nonexistent.yaml"))


def test_dump_yaml_roundtrip() -> None:
    cfg = load_config(cli_overrides={"seed": 99})
    yaml_str = dump_config_yaml(cfg)
    assert "seed: 99" in yaml_str
    # reload from yaml string via temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_str)
        f.flush()
        cfg2 = load_config(config_path=pathlib.Path(f.name))
    assert cfg2.seed == 99
    pathlib.Path(f.name).unlink()
