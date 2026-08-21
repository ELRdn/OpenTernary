"""CLI help and exit code tests — Phase 0 acceptance criteria."""

from typer.testing import CliRunner

from openternary.cli.main import app

runner = CliRunner()


def test_main_help_shows_six_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["inspect", "quantize", "benchmark", "compare", "calibrate", "export"]:
        assert cmd in result.output, f"missing command {cmd}"


def test_inspect_help_shows_common_options() -> None:
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    for opt in ["--config", "--output", "--seed", "--device", "--dtype", "--dry-run", "--load-weights"]:
        assert opt in result.output, f"missing option {opt}"


def test_quantize_help_shows_group_size() -> None:
    result = runner.invoke(app, ["quantize", "--help"])
    assert result.exit_code == 0
    assert "--group-size" in result.output


def test_dry_run_exit_0_no_side_effect() -> None:
    result = runner.invoke(app, ["inspect", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "No run directory was created" in result.output


def test_dry_run_with_config_and_group_size() -> None:
    result = runner.invoke(app, ["quantize", "--dry-run", "--group-size", "64", "--config", "configs/gemma4-e2b.yaml"])
    assert result.exit_code == 0
    assert "group_size: 64" in result.output


def test_inspect_normal_creates_run_exit_0() -> None:
    # inspect is now implemented (header-only, cache-only). Should exit 0 and create inspection.json.
    import pathlib
    import shutil
    import uuid

    tmp_base = pathlib.Path.cwd() / f"test_cli_{uuid.uuid4().hex[:6]}"
    tmp_base.mkdir(parents=True, exist_ok=True)
    try:
        out = str(tmp_base / "run_cli_test")
        result = runner.invoke(app, ["inspect", "--output", out])
        assert result.exit_code == 0, result.output
        assert pathlib.Path(out).exists()
        assert (pathlib.Path(out) / "inspection.json").exists()
        assert (pathlib.Path(out) / "metrics.json").exists()
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)


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
