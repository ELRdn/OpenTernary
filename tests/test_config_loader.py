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


def test_soft_to_hard_calibration_config_roundtrip(tmp_path: pathlib.Path) -> None:
    config_path = tmp_path / "soft-to-hard.yaml"
    config_path.write_text(
        """calibration:
  enabled: true
  method: recon-soft-to-hard
  temperature_schedule: cosine
  temperature_start: 1.0
  temperature_end: 0.05
  hard_fraction: 0.1
  zero_logit_bias: 0.25
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path=config_path)
    assert cfg.calibration.method == "recon-soft-to-hard"
    assert cfg.calibration.temperature_schedule == "cosine"
    assert cfg.calibration.temperature_start == 1.0
    assert cfg.calibration.temperature_end == 0.05
    assert cfg.calibration.hard_fraction == 0.1
    assert cfg.calibration.zero_logit_bias == 0.25


def test_soft_to_hard_rejects_threshold_optimizer_combination() -> None:
    with pytest.raises(ValueError, match="threshold_enabled"):
        load_config(
            cli_overrides={
                "calibration.method": "recon-soft-to-hard",
                "calibration.threshold_enabled": True,
            }
        )


def test_soft_to_hard_rejects_increasing_temperature() -> None:
    with pytest.raises(ValueError, match="temperature_start"):
        load_config(
            cli_overrides={
                "calibration.method": "recon-soft-to-hard",
                "calibration.temperature_start": 0.05,
                "calibration.temperature_end": 1.0,
            }
        )
