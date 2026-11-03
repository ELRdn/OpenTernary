from __future__ import annotations

import typer

from openternary.cli.common import app, console
from openternary.cli.protocol import Command
from openternary.ui.i18n import t


@app.command("cache-info", cls=Command, help=t("help.cache_info"), rich_help_panel=t("panel.environment"))
def cache_info() -> None:
    """Show cache configuration."""
    import os

    console.print("[bold]Cache info[/bold]")
    console.print(f"UV_CACHE_DIR: {os.environ.get('UV_CACHE_DIR', '(default: %LOCALAPPDATA%/uv/cache)')}")
    console.print(f"HF_HOME: {os.environ.get('HF_HOME', '(default: ~/.cache/huggingface)')}")
    raise typer.Exit(0)
