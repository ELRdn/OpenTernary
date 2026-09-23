"""OpenTernary CLI entry point — Phase 0 foundation."""

from __future__ import annotations

import pathlib

import typer
from rich.console import Console

app = typer.Typer(
    name="openternary",
    help="OpenTernary — quantization CLI (model validation tracked separately)",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def _resolve_cli_overrides(
    seed: int | None,
    device: str | None,
    dtype: str | None,
    output: str | None,
    group_size: int | None = None,
    scale_granularity: str | None = None,
    grouping_scheme: str | None = None,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if seed is not None:
        overrides["seed"] = seed
    if device is not None:
        overrides["device"] = device
    if dtype is not None:
        overrides["dtype"] = dtype
    if output is not None:
        overrides["output"] = output
    if group_size is not None:
        overrides["quantization.group_size"] = group_size
    if scale_granularity is not None:
        overrides["quantization.scale_granularity"] = scale_granularity
    if grouping_scheme is not None:
        overrides["quantization.grouping_scheme"] = grouping_scheme
    return overrides


def _write_failed_metrics(run_dir: pathlib.Path, run_id: str, error_type: str, error_message: str) -> None:
    """失敗runの metrics.json を QR-02 Failure Visibility に従い更新."""
    try:
        from openternary.experiment.metadata import write_json

        write_json(
            run_dir / "metrics.json",
            {
                "status": "failed",
                "run_id": run_id,
                "error_type": error_type,
                "error_message": error_message,
            },
        )
    except Exception:
        pass
