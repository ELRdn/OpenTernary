"""The Textual TUI. Imported lazily: plain commands and every ``--json`` path never load Textual."""

from __future__ import annotations

from pathlib import Path


def run(runs_root: Path | None = None) -> None:
    from openternary.ui.app.app import OpenTernaryApp

    OpenTernaryApp(runs_root=runs_root).run()
