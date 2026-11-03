"""Animated splash: the OpenTernary symbol turning beside the wordmark.

Run ``python -m openternary.ui.splash`` (``--ccw`` for counter-clockwise) to preview in a real
terminal. The default turn is clockwise; ``OPENTERNARY_SPIN=ccw`` changes it. Motion is skipped
when stdout is not a TTY or when ``OPENTERNARY_REDUCED_MOTION`` / ``NO_MOTION`` is set; the
first frame is then printed once.
"""

from __future__ import annotations

import os
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from openternary import __version__
from openternary.ui import brand

FPS = 12  # 48 frames -> one calm turn every 4 seconds


def reduced_motion(console: Console) -> bool:
    flags = (os.environ.get("OPENTERNARY_REDUCED_MOTION"), os.environ.get("NO_MOTION"))
    return not console.is_terminal or any(v not in (None, "", "0") for v in flags)


def clockwise_default() -> bool:
    spin = os.environ.get("OPENTERNARY_SPIN", "").lower()
    return brand.SPIN_CLOCKWISE if spin not in ("cw", "ccw") else spin == "cw"


def frame(console: Console, index: int, clockwise: bool = False) -> Group:
    width = console.size.width
    cols = 30 if width >= 100 else 24 if width >= 70 else 16
    info = Table.grid()
    info.add_row(Text(f"OpenTernary {__version__}", style=f"bold {brand.ACCENT}"))
    info.add_row(Text("ternary LLM toolkit · 三値LLMツールキット", style=brand.MUTED))
    info.add_row(Text(""))
    for trit, label in ((1, "+1"), (0, " 0"), (-1, "-1")):
        row = Text()
        row.append("━━ ", style=brand.TRIT[trit])
        row.append(label, style=brand.TEXT)
        info.add_row(row)
    body = Table.grid(padding=(0, 3))
    body.add_column()
    body.add_column(vertical="middle")
    body.add_row(brand.mark(cols, index, clockwise), info)
    return Group(brand.banner(width), Text(""), body)


def play(seconds: float = 8.0, console: Console | None = None, clockwise: bool | None = None) -> None:
    console = console or Console()
    cw = clockwise_default() if clockwise is None else clockwise
    if reduced_motion(console):
        console.print(frame(console, 0, cw))
        return
    n = brand.mark_frame_count()
    with Live(frame(console, 0, cw), console=console, refresh_per_second=FPS, transient=False) as live:
        start = time.monotonic()
        i = 0
        while time.monotonic() - start < seconds:
            i += 1
            live.update(frame(console, i % n, cw))
            time.sleep(1 / FPS)


if __name__ == "__main__":
    play(clockwise=False if "--ccw" in sys.argv[1:] else True if "--cw" in sys.argv[1:] else None)
