"""Public entry point; commands delegate to reusable library services."""

from __future__ import annotations

import os
import signal
import sys
from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer

from openternary import __version__
from openternary.cli.common import app, console, err_console
from openternary.ui.i18n import t


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


def interactive() -> bool:
    """A person at a terminal (not a pipe, CI, or a dumb terminal) and the TUI not switched off."""
    if os.environ.get("OPENTERNARY_NO_TUI") or os.environ.get("CI") or os.environ.get("TERM") == "dumb":
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


@app.command("ui", help=t("help.ui"), rich_help_panel=t("panel.environment"))
def ui(
    runs: Annotated[Path, typer.Option("--runs", help="Folder that holds run directories")] = Path("runs"),
) -> None:
    """Open the full-screen interface (needs an interactive terminal)."""
    if not interactive():
        err_console.print(t("tui.needs_terminal"))
        raise typer.Exit(2)
    from openternary.ui.app import run as run_tui

    run_tui(runs)


def run() -> None:
    """Console entry point: no arguments at a terminal opens the TUI; everything else is the CLI."""
    from openternary.services.jobs import BREAK_ENV

    if os.environ.get(BREAK_ENV) and hasattr(signal, "SIGBREAK"):  # Windows job cancel -> exit 130
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    if len(sys.argv) == 1 and interactive():
        from openternary.ui.app import run as run_tui

        run_tui()
        return
    app()


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
    "ui",
    "doctor",
    "setup",
    "update",
    "plugins",
    "cache-info",
)
app.registered_commands.sort(key=lambda c: HELP_ORDER.index(c.name) if c.name in HELP_ORDER else len(HELP_ORDER))
