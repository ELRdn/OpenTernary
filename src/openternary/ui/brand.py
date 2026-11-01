"""OpenTernary brand: Yoru palette, 3D ANSI-Shadow wordmark, and the official symbol.

The symbol is the owner's ``docs/brand/openternary-v1.0.svg`` (three rotated tomoe for
+1 / 0 / -1 inside a spectrum ring). ``scripts/design/bake_mark.py`` rasterises it exactly and
bakes Braille rotation frames into ``assets/tomoe.otmk``; loading needs only the standard
library, so no numerical dependency is added to the core install.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from rich.style import Style
from rich.text import Text

# --- Yoru (夜) palette: WCAG AA for every foreground on BG, CVD-checked status colours ---
BG = "#090c1c"
SURFACE = "#121627"
BORDER = "#272c46"
TEXT = "#e9ebf3"
MUTED = "#949cc2"
ACCENT = "#b89ffe"
OK = "#40e5e5"
WARN = "#f0ba59"
ERR = "#f2725b"
INFO = "#8bc3fc"

PALETTE = {
    "bg": BG,
    "surface": SURFACE,
    "border": BORDER,
    "text": TEXT,
    "muted": MUTED,
    "accent": ACCENT,
    "ok": OK,
    "warn": WARN,
    "err": ERR,
    "info": INFO,
}

# Colours taken from the official SVG (docs/brand/openternary-v1.0.svg).
SPECTRUM = ("#ffa36b", "#ffd157", "#7aef9c", "#38d9d0", "#5e8dff", "#aa7eff", "#ef8dd0")  # ring
SPECTRUM_STOPS = (0.0, 0.18, 0.34, 0.53, 0.72, 0.9, 1.0)


def oklch(lightness: float, chroma: float, hue: float) -> str:
    """OKLCH -> sRGB hex, reducing chroma until the colour is inside the sRGB gamut."""
    h = math.radians(hue)
    for _ in range(40):
        a, b = chroma * math.cos(h), chroma * math.sin(h)
        l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
        m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
        s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
        lin = (
            4.0767416621 * l_**3 - 3.3077115913 * m_**3 + 0.2309699292 * s_**3,
            -1.2684380046 * l_**3 + 2.6097574011 * m_**3 - 0.3413193965 * s_**3,
            -0.0041960863 * l_**3 - 0.7034186147 * m_**3 + 1.7076147010 * s_**3,
        )
        if all(-1e-4 <= c <= 1 + 1e-4 for c in lin):
            break
        chroma *= 0.95
    rgb = (12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0.0) ** (1 / 2.4) - 0.055 for c in lin)
    return "#" + "".join(f"{round(min(max(c, 0.0), 1.0) * 255):02x}" for c in rgb)


# trit -> the middle stop of that tomoe's gradient in the SVG (+1 amber, 0 mint, -1 violet)
TRIT = {1: "#ffad55", 0: "#43efc5", -1: "#8168fa"}


def spectrum(t: float) -> str:
    """Colour of the ring gradient at position ``t`` in [0, 1] (linear between SVG stops)."""
    t = min(max(t, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(
        zip(SPECTRUM_STOPS, SPECTRUM, strict=True), zip(SPECTRUM_STOPS[1:], SPECTRUM[1:], strict=True), strict=False
    ):
        if t <= b:
            u = (t - a) / (b - a)
            rgb = (round(int(ca[i : i + 2], 16) * (1 - u) + int(cb[i : i + 2], 16) * u) for i in (1, 3, 5))
            return "#" + "".join(f"{c:02x}" for c in rgb)
    return SPECTRUM[-1]


WORDMARK = (
    " ██████╗ ██████╗ ███████╗███╗   ██╗   ████████╗███████╗██████╗ ███╗   ██╗ █████╗ ██████╗ ██╗   ██╗",
    "██╔═══██╗██╔══██╗██╔════╝████╗  ██║   ╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝",
    "██║   ██║██████╔╝█████╗  ██╔██╗ ██║█████╗██║   █████╗  ██████╔╝██╔██╗ ██║███████║██████╔╝ ╚████╔╝ ",
    "██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════╝██║   ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║██╔══██╗  ╚██╔╝  ",
    "╚██████╔╝██║     ███████╗██║ ╚████║      ██║   ███████╗██║  ██║██║ ╚████║██║  ██║██║  ██║   ██║   ",
    " ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ",
)
WORDMARK_WIDTH = max(len(line) for line in WORDMARK)


@lru_cache(maxsize=1)
def _wordmark_gradient() -> tuple[str, ...]:
    return tuple(spectrum(i / (WORDMARK_WIDTH - 1)) for i in range(WORDMARK_WIDTH))


def wordmark() -> Text:
    """Faces (█) carry the rainbow; strokes (╔═╝) are the extruded shadow in the border colour."""
    grad = _wordmark_gradient()
    out = Text(no_wrap=True, overflow="crop")
    for row, line in enumerate(WORDMARK):
        for col, ch in enumerate(line):
            if ch == " ":
                out.append(ch)
            elif ch == "█":
                out.append(ch, Style(color=grad[col], bold=True))
            else:
                out.append(ch, Style(color=MUTED if row < 5 else BORDER))
        if row < len(WORDMARK) - 1:
            out.append("\n")
    return out


def compact_wordmark() -> Text:
    """Narrow-terminal fallback (< ~100 columns)."""
    grad = _wordmark_gradient()
    out = Text(no_wrap=True)
    for i, ch in enumerate("OPEN-TERNARY"):
        out.append(ch, Style(color=grad[round(i / 11 * (WORDMARK_WIDTH - 1))], bold=True))
    return out


def banner(width: int) -> Text:
    """Hero banner sized to the terminal: full 3D wordmark when it fits."""
    return wordmark() if width >= WORDMARK_WIDTH + 2 else compact_wordmark()


# --- baked mark -------------------------------------------------------------------------
@dataclass(frozen=True)
class MarkAsset:
    frames: int
    palette: tuple[tuple[str, int], ...]  # (hex colour, band)
    sizes: dict[int, tuple[int, list[bytes]]]  # cols -> (rows, per-frame cell bytes)


@lru_cache(maxsize=1)
def load_mark() -> MarkAsset:
    raw = zlib.decompress(resources.files("openternary.ui").joinpath("assets/tomoe.otmk").read_bytes())
    if raw[:4] != b"OTMK" or raw[4] != 1:
        raise ValueError("unsupported mark asset")
    frames, n_sizes, n_pal = raw[5], raw[6], raw[7]
    pos = 8
    palette = []
    for _ in range(n_pal):
        r, g, b, band = struct.unpack_from("BBBB", raw, pos)
        palette.append((f"#{r:02x}{g:02x}{b:02x}", band))
        pos += 4
    sizes: dict[int, tuple[int, list[bytes]]] = {}
    for _ in range(n_sizes):
        cols, rows = raw[pos], raw[pos + 1]
        pos += 2
        step = rows * cols * 2
        sizes[cols] = (rows, [raw[pos + f * step : pos + (f + 1) * step] for f in range(frames)])
        pos += frames * step
    return MarkAsset(frames, tuple(palette), sizes)


MARK_SIZES = (16, 24, 30)


def _best_size(cols: int) -> int:
    fitting = [c for c in MARK_SIZES if c <= cols]
    return max(fitting) if fitting else min(MARK_SIZES)


@lru_cache(maxsize=512)
def mark(cols: int = 30, frame: int = 0, clockwise: bool = False) -> Text:
    """The symbol at the largest baked size <= ``cols``.

    Successive ``frame`` values turn it counter-clockwise (or clockwise when asked); frames are
    stored for counter-clockwise turns, so clockwise playback walks them backwards.
    """
    asset = load_mark()
    size = _best_size(cols)
    rows, frames = asset.sizes[size]
    cells = frames[(-frame if clockwise else frame) % asset.frames]
    out = Text(no_wrap=True)
    for cy in range(rows):
        for cx in range(size):
            bits, idx = cells[(cy * size + cx) * 2], cells[(cy * size + cx) * 2 + 1]
            if idx == 255:
                out.append(" ")
            else:
                out.append(chr(0x2800 + bits), Style(color=asset.palette[idx][0]))
        if cy < rows - 1:
            out.append("\n")
    return out


def mark_frame_count() -> int:
    return load_mark().frames
