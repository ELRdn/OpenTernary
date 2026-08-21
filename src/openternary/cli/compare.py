"""compare subcommand — Phase 0 stub."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.main import _handle_command, _resolve_cli_overrides

app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def compare_cmd(
    ctx: typer.Context,
    baseline: Annotated[str | None, typer.Argument(help="Baseline run directory")] = None,
    quantized: Annotated[str | None, typer.Argument(help="Quantized run directory")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Compare benchmark runs (Phase 0 stub)."""
    if ctx.invoked_subcommand is not None:
        return
    overrides = _resolve_cli_overrides(seed, device, dtype, output)
    # baseline/quantized は将来利用
    _handle_command("compare", config, overrides, dry_run, verbose)
