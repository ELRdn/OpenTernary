"""Render the draft home-screen mock (sample data only) to SVG for design review.

usage: python scripts/render_brand_preview.py OUT.svg [WIDTH]
"""

import sys

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.terminal_theme import TerminalTheme
from rich.text import Text

from openternary.ui import brand as b

W = int(sys.argv[2]) if len(sys.argv) > 2 else 120
c = Console(record=True, width=W, force_terminal=True, color_system="truecolor")


def kv(k, v, vs=b.CHAMPAGNE):
    t = Text()
    t.append(f"{k:<10}", style=b.BRONZE)
    t.append(v, style=vs)
    return t


info = Table.grid(padding=(0, 0))
info.add_row(Text("Environment  環境", style=f"bold {b.GOLD}"))
info.add_row(kv("python", "3.12.8 · uv 0.12.5"))
info.add_row(kv("torch", "2.13.0+rocm6.4  ·  RX 9070 XT"))
info.add_row(kv("backends", "ternary ● torchao ○ diffusers ○"))
info.add_row(Text(""))
info.add_row(Text("Recent runs  最近の実行", style=f"bold {b.GOLD}"))


def run(d, name, cmd, state, col):
    t = Text()
    t.append(f"{d}  ", style=b.ASH)
    t.append(f"{name:<22}", style=b.CHAMPAGNE)
    t.append(f"{cmd:<10}", style=b.BRONZE)
    t.append(state, style=col)
    return t


info.add_row(run("11-01", "gemma4-e2b", "calibrate", "◐ running 42%", b.GOLD))
info.add_row(run("10-31", "gemma4-e2b-g128", "quantize", "● completed · quality not run", b.CHAMPAGNE))
info.add_row(run("10-30", "baseline", "benchmark", "✔ accepted", "#9cc48b"))
info.add_row(run("10-29", "gemma4-e2b-thr", "calibrate", "✖ failed (exit 5)", b.COPPER))
info.add_row(Text(""))
keys = Text()
for k, v in [("n", "new run"), ("r", "runs"), ("m", "compare"), ("d", "doctor"), ("u", "update"), ("?", "help")]:
    keys.append(f" {k} ", style=f"bold {b.INK} on {b.GOLD}")
    keys.append(f" {v}  ", style=b.CHAMPAGNE)
info.add_row(keys)

body = Table.grid(padding=(0, 3))
body.add_column()
body.add_column()
mark = Group(b.mark(30), Text(""), Text("  ternary lab · 三値研究室", style=b.ASH))
body.add_row(mark, info)

title = Text()
title.append(" OpenTernary ", style=f"bold {b.GOLD}")
title.append("v0.2.0 ", style=b.CHAMPAGNE)
title.append("· main@544e993 ", style=b.ASH)
sub = Text()
sub.append(" ▲ v0.2.1 available — ", style=b.EMBER)
sub.append("openternary update ", style=f"bold {b.GOLD}")
c.print(b.banner(W))
c.print()
c.print(
    Panel(
        body,
        title=title,
        title_align="left",
        subtitle=sub,
        subtitle_align="left",
        border_style=b.BRONZE,
        padding=(1, 2),
    )
)
c.print(Text(" Welcome. Type a command or press ? — ようこそ。? でヘルプ", style=b.CHAMPAGNE))
theme = TerminalTheme((11, 11, 12), (240, 228, 204), [(0, 0, 0)] * 8, [(0, 0, 0)] * 8)
c.save_svg(sys.argv[1], title="openternary", theme=theme)
