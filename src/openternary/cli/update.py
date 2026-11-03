"""`openternary update`: check PyPI, upgrade, and restore the PyTorch build chosen by `setup`."""

from __future__ import annotations

import subprocess
import sys
from typing import Annotated, Any

import typer

from openternary.cli.common import app, console, explain
from openternary.cli.protocol import Command
from openternary.ui.i18n import t


@app.command("update", cls=Command, help=t("help.update"), rich_help_panel=t("panel.environment"))
def update(
    check_only: Annotated[bool, typer.Option("--check", help="Only check for a newer release")] = False,
    pre: Annotated[bool, typer.Option("--pre", help="Consider pre-releases (default: if installed is one)")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Upgrade without asking")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would run; change nothing")] = False,
) -> None:
    """Check for a newer OpenTernary on PyPI and upgrade this installation."""
    from openternary.services import torch_setup as ts
    from openternary.services import update as upd
    from openternary.services.reporting import milestone, publish
    from openternary.ui.render import update_view

    milestone("update.check")
    status = upd.check(force=True, include_pre=True if pre else None)
    publish(status=status.__dict__)
    if status.error:
        update_view(console, status, None, None)
        explain(t("update.check_failed"))
        raise ConnectionError(status.error)
    profile = ts.read_profile()
    mode = upd.install_mode()
    plan = upd.plan_upgrade(mode, profile is not None, upd.find_uv())
    publish(install_mode=mode, plan=plan.__dict__)
    if not status.update_available or check_only:
        update_view(console, status, None, None)
        return
    saved = upd.save_profile_copy(profile) if plan.restore else None
    restore = _restore_command(saved) if saved else None
    publish(restore_command=restore, saved_profile=str(saved) if saved else None)
    update_view(console, status, plan, restore)
    if dry_run or not plan.automatic or not plan.commands:
        if not plan.commands:
            explain(t("update.no_uv"))
            raise FileNotFoundError("uv was not found on PATH")
        return
    if not yes:
        if not sys.stdin.isatty() or not console.is_terminal:
            explain(t("setup.needs_yes"))
            raise ValueError("upgrading needs confirmation: rerun with --yes")
        if not typer.confirm(t("update.confirm", version=status.latest), default=False):
            raise typer.Abort()
    for command in plan.commands:
        milestone("update.run", command=command[1:3])
        if _run(command) != 0:
            explain(t("update.failed"))
            raise RuntimeError(f"{' '.join(command[:3])} failed")
    if restore:
        milestone("update.restore")
        if _run(restore) != 0:
            explain(t("update.restore_failed", command=" ".join(restore)))
            raise RuntimeError("restoring the recorded PyTorch build failed")
    upd.check(force=True)  # refresh the cache with the new version
    console.print(t("update.done", version=status.latest))


def _restore_command(saved: Any) -> list[str]:
    return [sys.executable, "-m", "openternary", "setup", "--restore", "--profile", str(saved), "--yes"]


def _run(command: list[str]) -> int:
    try:
        sink: Any = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        sink = None
    return subprocess.run(command, stdout=sink, stderr=sink, check=False).returncode
