"""benchmark subcommand — delegates to main.benchmark (Phase 1b real)."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.main import benchmark as main_benchmark

app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def benchmark_cmd(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    suite: Annotated[str | None, typer.Option("--suite", help="Benchmark suite: smoke")] = None,
    thinking: Annotated[bool | None, typer.Option("--thinking/--no-thinking", help="Gemma 4 thinking mode")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Benchmark model — delegates to main benchmark command."""
    if ctx.invoked_subcommand is not None:
        return
    main_benchmark(
        model=model,
        config=config,
        output=output,
        seed=seed,
        device=device,
        dtype=dtype,
        suite=suite,
        thinking=thinking,
        dry_run=dry_run,
        verbose=verbose,
    )
