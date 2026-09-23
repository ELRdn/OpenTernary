"""Atomic export of snapshots and packed artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from openternary.cli.common import app
from openternary.cli.product import display
from openternary.cli.protocol import Command
from openternary.services.artifacts import export_artifact
from openternary.services.errors import CapabilityError


@app.command("export", cls=Command)
def export(
    run: Annotated[Path, typer.Argument(help="Run or artifact directory")],
    format: Annotated[str, typer.Option("--format")] = "safetensors",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    converter: Annotated[
        Path | None, typer.Option("--converter", help="Explicit local llama.cpp converter for GGUF")
    ] = None,
    output_dtype: Annotated[str, typer.Option("--output-dtype")] = "f16",
    converter_python: Annotated[Path | None, typer.Option("--converter-python")] = None,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=0.001)] = 3600,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Export without modifying the source. Existing outputs are rejected."""
    if format == "gguf" and converter is None:
        raise CapabilityError("GGUF requires an explicitly selected local --converter; runtime validation is pending")
    if format not in {"safetensors", "ternary-packed", "torchao", "gguf"}:
        from openternary.services.plugins import discover_plugins

        if not any(
            row["kind"] == "exporters" and row["name"] == format and row["status"] == "discovered"
            for row in discover_plugins()
        ):
            raise CapabilityError(f"unsupported export format: {format}")
    destination = output or run.with_name(f"{run.name}-{format}")
    if dry_run:
        display({"status": "planned", "source": str(run), "destination": str(destination), "format": format})
        return
    if format == "gguf":
        from openternary.export.gguf import export_gguf

        assert converter is not None
        display(
            export_gguf(
                run,
                destination,
                converter,
                output_dtype=output_dtype,
                converter_python=converter_python,
                timeout_seconds=timeout_seconds,
            )
        )
        return
    display(export_artifact(run, destination, format))
