"""Public entry point; commands delegate to reusable library services."""

from __future__ import annotations

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


from openternary.cli import (  # noqa: E402, F401
    benchmark,
    cache,  # noqa: E402, F401
    cache_info,
    calibrate,
    compare,
    evaluate,  # noqa: E402, F401
    export,
    inspect,
    optimize,  # noqa: E402, F401
    product,  # noqa: E402, F401
    quality,
    quantize,
)
