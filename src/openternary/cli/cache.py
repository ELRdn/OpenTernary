"""Explicit cache inspection and deletion, separate from shared Hugging Face caches."""

from pathlib import Path
from typing import Annotated

import typer

from openternary.cli.common import app
from openternary.cli.product import display
from openternary.cli.protocol import Command

cache = typer.Typer(help="Manage explicitly selected OpenTernary artifact roots", no_args_is_help=True)
app.add_typer(cache, name="cache")


@cache.command("list", cls=Command)
def cache_list(root: Path) -> None:
    from openternary.services.cache import list_artifacts

    display({"artifacts": list_artifacts(root)})


@cache.command("remove", cls=Command)
def cache_remove(
    target: Path,
    root: Annotated[Path, typer.Option("--root")],
    execute: Annotated[bool, typer.Option("--execute", help="Perform the named deletion")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    from openternary.services.cache import remove_artifact

    if execute and dry_run:
        raise ValueError("--execute and --dry-run cannot be combined")
    display(remove_artifact(root, target, execute=execute))
