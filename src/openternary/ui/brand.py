"""OpenTernary brand assets: palette, 3D ANSI-Shadow wordmark, and the mitsudomoe mark.

The mark is a procedural mitsudomoe (三つ巴): three commas for the three ternary
states {-1, 0, +1}. It is rendered with Braille dots so it scales to any cell grid
and needs no image support in the terminal. Draft for design review.
"""

from __future__ import annotations

import math
from functools import lru_cache

from rich.style import Style
from rich.text import Text

# --- palette (draft; derived from the lab mark: champagne gold, copper, ink) ---------
INK = "#0b0b0c"
CHAMPAGNE = "#f0e4cc"
GOLD = "#e2c28a"
BRONZE = "#8a7350"
COPPER = "#c65a3a"
EMBER = "#ffd9a0"
ASH = "#6e6a63"

# trit -> colour. +1 gold, 0 champagne, -1 copper (same order as the mark's commas)
TRIT = {1: GOLD, 0: CHAMPAGNE, -1: COPPER}

WORDMARK = (
    " ██████╗ ██████╗ ███████╗███╗   ██╗   ████████╗███████╗██████╗ ███╗   ██╗ █████╗ ██████╗ ██╗   ██╗",
    "██╔═══██╗██╔══██╗██╔════╝████╗  ██║   ╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝",
    "██║   ██║██████╔╝█████╗  ██╔██╗ ██║█████╗██║   █████╗  ██████╔╝██╔██╗ ██║███████║██████╔╝ ╚████╔╝ ",
    "██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════╝██║   ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║██╔══██╗  ╚██╔╝  ",
    "╚██████╔╝██║     ███████╗██║ ╚████║      ██║   ███████╗██║  ██║██║ ╚████║██║  ██║██║  ██║   ██║   ",
    " ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ",
)
WORDMARK_WIDTH = max(len(line) for line in WORDMARK)


def _hex(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _mix(a: str, b: str, t: float) -> str:
    (r1, g1, b1), (r2, g2, b2) = _hex(a), _hex(b)
    return "#{:02x}{:02x}{:02x}".format(*(round(x + (y - x) * t) for x, y in ((r1, r2), (g1, g2), (b1, b2))))


def _gradient(t: float) -> str:
    """champagne -> gold -> copper across the wordmark."""
    return _mix(CHAMPAGNE, GOLD, t * 2) if t < 0.5 else _mix(GOLD, COPPER, (t - 0.5) * 2)


def wordmark() -> Text:
    """Faces (█) carry the metallic gradient; strokes (╔═╝) are the extruded shadow."""
    out = Text(no_wrap=True, overflow="crop")
    for row, line in enumerate(WORDMARK):
        for col, ch in enumerate(line):
            if ch == " ":
                out.append(ch)
                continue
            t = col / (WORDMARK_WIDTH - 1)
            if ch == "█":
                out.append(ch, Style(color=_gradient(t), bold=True))
            else:
                out.append(ch, Style(color=_mix(BRONZE, COPPER, t) if row < 5 else BRONZE))
        if row < len(WORDMARK) - 1:
            out.append("\n")
    return out


def compact_wordmark() -> Text:
    """Narrow-terminal fallback (< ~100 columns)."""
    out = Text(no_wrap=True)
    out.append("◐ ", Style(color=COPPER))
    for i, ch in enumerate("OPEN-TERNARY"):
        out.append(ch, Style(color=_gradient(i / 11), bold=True))
    return out


# --- mitsudomoe ---------------------------------------------------------------------
_A, _H, _TIP, _SWEEP = 0.40, 0.25, 0.93, math.radians(205)
_ORBIT = (math.cos(math.radians(38)), math.sin(math.radians(38)))


def _sample(x: float, y: float) -> int | None:
    """Comma index 0..2, 3 = outer ring, 4 = orbit dot, None = empty. Domain [-1, 1]^2."""
    r = math.hypot(x, y)
    phi = math.atan2(y, x)
    for k in range(3):
        th = 2 * math.pi * k / 3 + math.pi / 2
        if (x - _A * math.cos(th)) ** 2 + (y - _A * math.sin(th)) ** 2 <= _H * _H:
            return k
        d = (th - phi) % (2 * math.pi)
        if d <= _SWEEP:
            t = d / _SWEEP
            outer = (_A + _H) + (_TIP - (_A + _H)) * math.sin(t * math.pi / 2)
            inner = _A + (_TIP - _A) * t**0.75
            if inner <= r <= outer:
                return k
    if (x - _ORBIT[0]) ** 2 + (y - _ORBIT[1]) ** 2 <= 0.07**2:
        return 4
    if 0.975 <= r <= 1.03:
        return 3
    return None


_BRAILLE = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))  # [row][col]
_COMMA_COLOUR = {0: TRIT[1], 1: TRIT[0], 2: TRIT[-1], 3: BRONZE, 4: EMBER}


@lru_cache(maxsize=8)
def _mark_cells(cols: int) -> tuple[tuple[tuple[str, int | None], ...], ...]:
    rows = cols // 2  # terminal cells are ~1:2, braille dots are then ~square
    dw, dh = cols * 2, rows * 4
    margin = 1.06
    grid: list[tuple[tuple[str, int | None], ...]] = []
    for cy in range(rows):
        line: list[tuple[str, int | None]] = []
        for cx in range(cols):
            bits, votes = 0, [0] * 5
            for py in range(4):
                for px in range(2):
                    x = ((cx * 2 + px + 0.5) / dw * 2 - 1) * margin
                    y = (1 - (cy * 4 + py + 0.5) / dh * 2) * margin
                    s = _sample(x, y)
                    if s is not None:
                        bits |= _BRAILLE[py][px]
                        votes[s] += 1
            if bits:
                # comma/orbit dots win ties over the ring so silhouettes stay crisp
                owner = max((i for i in range(5) if votes[i]), key=lambda i: votes[i] + (0 if i == 3 else 0.5))
                line.append((chr(0x2800 + bits), owner))
            else:
                line.append((" ", None))
        grid.append(tuple(line))
    return tuple(grid)


def mark(cols: int = 26) -> Text:
    """Mitsudomoe mark, `cols` cells wide and cols/2 rows tall."""
    out = Text(no_wrap=True)
    cells = _mark_cells(cols)
    for i, line in enumerate(cells):
        for ch, owner in line:
            out.append(ch, Style(color=_COMMA_COLOUR[owner]) if owner is not None else None)
        if i < len(cells) - 1:
            out.append("\n")
    return out


def banner(width: int) -> Text:
    """Hero banner sized to the terminal: full 3D wordmark when it fits."""
    return wordmark() if width >= WORDMARK_WIDTH + 2 else compact_wordmark()
