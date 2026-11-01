import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.terminal_theme import TerminalTheme
from final_sheet import labels, hint
from rainbow import colourise3
BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])
def braille(which, cols, S=8, hinted=True):
    n = cols * 2 * S
    lab = labels(which, n)
    if hinted: lab = hint(lab, n, cols)
    rgb, m = colourise3(lab)
    out = Text(no_wrap=True)
    for cy in range(cols // 2):
        for cx in range(cols):
            bits, acc, cnt = 0, np.zeros(3), 0
            for py in range(4):
                for px in range(2):
                    ys, xs = (cy * 4 + py) * S, (cx * 2 + px) * S
                    blk = m[ys:ys + S, xs:xs + S]
                    if blk.mean() > 0.45:
                        bits |= int(BITS[py, px]); acc += rgb[ys:ys + S, xs:xs + S][blk].mean(0); cnt += 1
            if bits:
                c = (acc / cnt * 255).astype(int); out.append(chr(0x2800 + bits), Style(color="#%02x%02x%02x" % tuple(c)))
            else:
                out.append(" ")
        out.append("\n")
    return out
c = Console(record=True, width=140, force_terminal=True, color_system="truecolor")
t = Table.grid(padding=(0, 4))
for _ in range(4): t.add_column()
t.add_row(braille("A", 30), braille("B", 30), braille("A", 16), braille("B", 16))
t.add_row("A  30 cols", "B  30 cols", "A  16 cols", "B  16 cols")
c.print("[#8a90a0]terminal Braille, optically sized (regions eroded so gaps >= ~1.2 dots)[/]"); c.print(t)
c.save_svg(sys.argv[1], title="A / B in the terminal", theme=TerminalTheme((14, 15, 19), (225, 228, 235), [(0, 0, 0)] * 8, [(0, 0, 0)] * 8))
