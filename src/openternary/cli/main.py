"""Public entry point; commands delegate to reusable library services."""

from __future__ import annotations

from importlib import import_module
from typing import Annotated

import typer

from openternary import __version__
from openternary.cli.common import app, console


def _version_callback(value: bool) -> None:
    if value:
        from openternary.utils.git import get_git_commit

        commit = get_git_commit()
        console.print(f"openternary {__version__} (commit: {commit})")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", help="Show version and exit", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """OpenTernary CLI."""
    pass


for _module in (
    "inspect",
    "product",  # plan, doctor, plugins, backends, artifacts
    "quantize",
    "calibrate",
    "export",
    "benchmark",
    "quality",
    "evaluate",
    "compare",
    "optimize",
    "cache_info",
    "cache",
    "setup",
    "update",
):
    import_module(f"openternary.cli.{_module}")

# Help shows panels in order of first appearance: workflow, evaluate, environment.
HELP_ORDER = (
    "inspect",
    "plan",
    "quantize",
    "calibrate",
    "export",
    "benchmark",
    "quality",
    "evaluate",
    "compare",
    "optimize",
    "doctor",
    "setup",
    "update",
    "plugins",
    "cache-info",
)
app.registered_commands.sort(key=lambda c: HELP_ORDER.index(c.name) if c.name in HELP_ORDER else len(HELP_ORDER))
