import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.terminal_theme import TerminalTheme
import rainbow, braille as br
from hinted import hinted
from openternary.ui.brand import WORDMARK, WORDMARK_WIDTH
from palettes import hx

p = [x for x in json.load(open(str(__import__("pathlib").Path(__file__).resolve().parent / "palettes.json"))) if x["name"] == sys.argv[1]][0]
c, dark = p["colors"], p["mode"] == "dark"
Lw, Cw = (0.82, 0.15) if dark else (0.56, 0.17)

def word():
    out = Text(no_wrap=True)
    for i, line in enumerate(WORDMARK):
        for j, ch in enumerate(line):
            if ch == " ": out.append(ch); continue
            hue = rainbow.H0 + 360 * j / (WORDMARK_WIDTH - 1) * 0.92
            out.append(ch, Style(color=hx(Lw, Cw, hue), bold=True) if ch == "█" else Style(color=c["border"] if dark else c["muted"]))
        if i < 5: out.append("\n")
    return out

# rainbow mark, lightness tuned per mode
_orig = rainbow.colourise2
br.colourise2 = lambda lab: _orig(lab, L=0.74 if dark else 0.60, C=0.19 if dark else 0.17)
mark = br.braille(hinted(24)["D  kazaguruma"], 24)

W = 106
con = Console(record=True, width=W, force_terminal=True, color_system="truecolor")
def kv(k, v): t = Text(); t.append(f"{k:<10}", style=c["muted"]); t.append(v, style=c["text"]); return t
def run(d, n, cmd, sym, label, col):
    t = Text(); t.append(f"{d}  ", style=c["muted"]); t.append(f"{n:<17}", style=c["text"]); t.append(f"{cmd:<10}", style=c["muted"])
    t.append(f"{sym} {label}", style=col); return t
info = Table.grid()
info.add_row(Text("Environment  環境", style=f"bold {c['text']}"))
info.add_row(kv("python", "3.12.8 · uv 0.12.5")); info.add_row(kv("torch", "2.13.0+rocm · RX 9070 XT"))
info.add_row(kv("backends", "ternary ● torchao ○")); info.add_row(Text(""))
info.add_row(Text("Recent runs  最近の実行", style=f"bold {c['text']}"))
info.add_row(run("11-01", "gemma4-e2b", "calibrate", "◐", "running 42%", c["info"]))
info.add_row(run("10-31", "gemma4-e2b-g128", "quantize", "●", "completed · quality not run", c["text"]))
info.add_row(run("10-30", "baseline", "benchmark", "✔", "accepted", c["ok"]))
info.add_row(run("10-29", "gemma4-thr", "calibrate", "✖", "failed (exit 5)", c["err"]))
info.add_row(run("10-28", "search-g64", "optimize", "▲", "budget exhausted", c["warn"]))
info.add_row(Text(""))
keys = Text()
for k, v in [("n", "new"), ("r", "runs"), ("m", "compare"), ("u", "update"), ("?", "help")]:
    keys.append(f" {k} ", style=f"bold {c['bg']} on {c['accent']}"); keys.append(f" {v}  ", style=c["text"])
info.add_row(keys)
body = Table.grid(padding=(0, 3)); body.add_column(); body.add_column()
body.add_row(mark, info)
title = Text(); title.append(" OpenTernary ", style=f"bold {c['accent']}"); title.append("v0.2.0 · main@bba74c1 ", style=c["muted"])
sub = Text(); sub.append(" ▲ v0.2.1 available · ", style=c["warn"]); sub.append("openternary update ", style=f"bold {c['text']}")
con.print(word()); con.print()
con.print(Panel(body, title=title, title_align="left", subtitle=sub, subtitle_align="left", border_style=c["border"], style=f"on {c['surface']}", padding=(1, 2)))
sw = Text(f" {p['name']} {p['jp']} · {p['mode']}   ", style=f"bold {c['text']}")
for k in ("bg", "surface", "border", "muted", "text", "accent", "ok", "warn", "err", "info"):
    sw.append("  ", style=f"on {c[k]}"); sw.append(f" {k} ", style=c["muted"])
con.print(sw)
bg = tuple(int(c["bg"][i:i+2], 16) for i in (1, 3, 5)); fg = tuple(int(c["text"][i:i+2], 16) for i in (1, 3, 5))
con.save_svg(sys.argv[2], title=f"palette: {p['name']} {p['jp']}", theme=TerminalTheme(bg, fg, [(0,0,0)]*8, [(0,0,0)]*8))
