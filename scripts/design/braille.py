import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from rich.text import Text
from rich.style import Style
from gen import STYLES
from rainbow import colourise2
BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])

def braille(style_fn, cols, S=8):
    rows = cols // 2
    n = cols * 2 * S                      # square: dots = cols*2 wide, rows*4 = cols*2 tall
    lab = style_fn(n)
    rgb, m = colourise2(lab)
    out = Text(no_wrap=True)
    for cy in range(rows):
        for cx in range(cols):
            bits, acc, cnt = 0, np.zeros(3), 0
            for py in range(4):
                for px in range(2):
                    ys, xs = (cy * 4 + py) * S, (cx * 2 + px) * S
                    blk = m[ys:ys + S, xs:xs + S]
                    if blk.mean() > 0.45:
                        bits |= int(BITS[py, px])
                        acc += rgb[ys:ys + S, xs:xs + S][blk].mean(0); cnt += 1
            if bits:
                c = (acc / cnt * 255).astype(int)
                out.append(chr(0x2800 + bits), Style(color=f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"))
            else:
                out.append(" ")
        out.append("\n")
    return out
