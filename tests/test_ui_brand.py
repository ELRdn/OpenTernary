"""Brand assets: deterministic, degrade on narrow / colourless terminals, mark rotates CCW."""

from __future__ import annotations

import io
import math

from rich.console import Console

from openternary.ui import brand


def render(renderable: object, width: int, color: bool = True, no_color: bool = False) -> str:
    buf = io.StringIO()
    console = Console(
        file=buf, width=width, force_terminal=color, no_color=no_color, color_system="truecolor" if color else "auto"
    )
    console.print(renderable)
    return buf.getvalue()


def test_wordmark_is_rectangular_and_fits_100_columns() -> None:
    assert len(brand.WORDMARK) == 6
    assert brand.WORDMARK_WIDTH <= 100
    plain = brand.wordmark().plain.splitlines()
    assert [line.rstrip() for line in plain] == [line.rstrip() for line in brand.WORDMARK]


def test_banner_falls_back_to_compact_on_narrow_terminals() -> None:
    assert brand.banner(80).plain == "OPEN-TERNARY"
    assert brand.banner(120).plain.splitlines()[0] == brand.WORDMARK[0]


def test_oklch_matches_reference_values() -> None:
    assert brand.oklch(1.0, 0.0, 0.0) == "#ffffff"
    assert brand.oklch(0.0, 0.0, 0.0) == "#000000"
    # chroma is reduced into gamut instead of clipping channels
    assert brand.oklch(0.74, 0.40, 140.0).startswith("#")


def test_mark_sizes_are_square_braille() -> None:
    for cols in brand.MARK_SIZES:
        lines = brand.mark(cols).plain.splitlines()
        assert len(lines) == cols // 2
        assert all(len(line) == cols for line in lines)
        assert all(ch == " " or 0x2800 <= ord(ch) <= 0x28FF for line in lines for ch in line)
    assert brand.mark(200).plain == brand.mark(30).plain  # largest baked size
    assert brand.mark(10).plain == brand.mark(16).plain  # smallest baked size


def test_palette_keeps_tomoe_and_ring_bands() -> None:
    asset = brand.load_mark()
    assert {band for _, band in asset.palette} == {0, 1, 2, 3}  # +1, 0, -1, ring


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return int(hex_colour[1:3], 16), int(hex_colour[3:5], 16), int(hex_colour[5:7], 16)


def test_baked_colours_follow_the_official_svg() -> None:
    """Each tomoe's baked colours sit closest to its own SVG gradient colour."""
    asset = brand.load_mark()
    trits = (1, 0, -1)
    for band, trit in enumerate(trits):
        cols = [_rgb(c) for c, b in asset.palette if b == band]
        mean = tuple(sum(c[i] for c in cols) / len(cols) for i in range(3))
        dist = {t: sum((mean[i] - _rgb(brand.TRIT[t])[i]) ** 2 for i in range(3)) for t in trits}
        assert min(dist, key=dist.__getitem__) == trit, (band, mean)


def test_clockwise_walks_frames_backwards() -> None:
    n = brand.mark_frame_count()
    assert brand.mark(30, 1, clockwise=True).plain == brand.mark(30, n - 1).plain
    assert brand.mark(30, 0, clockwise=True).plain == brand.mark(30, 0).plain


def _band_centroid_angle(frame: int, band: int, cols: int = 30) -> float:
    asset = brand.load_mark()
    rows, frames = asset.sizes[cols]
    cells = frames[frame]
    xs, ys = [], []
    for cy in range(rows):
        for cx in range(cols):
            idx = cells[(cy * cols + cx) * 2 + 1]
            if idx != 255 and asset.palette[idx][1] == band:
                xs.append(cx - (cols - 1) / 2)
                ys.append((cy - (rows - 1) / 2) * 2)  # cells are ~1:2
    # screen y grows downwards, so flip it: a positive angle step is counter-clockwise
    return math.degrees(math.atan2(-sum(ys) / len(ys), sum(xs) / len(xs)))


def test_mark_rotates_counter_clockwise() -> None:
    n = brand.mark_frame_count()
    assert n == 48
    step = 360 / n
    # cell-level centroids jitter by a few degrees, so compare over 120 degrees
    for band in range(3):
        angles = [_band_centroid_angle(f, band) for f in range(0, 17, 4)]
        steps = [(b - a + 180) % 360 - 180 for a, b in zip(angles, angles[1:], strict=False)]
        assert all(d > 0 for d in steps), (band, steps)  # always counter-clockwise
        assert abs(sum(steps) - 16 * step) < 12, (band, sum(steps))


def test_mark_has_threefold_symmetry_in_shape() -> None:
    """A 120 degree turn reproduces the dot pattern (colours move to the next band)."""
    asset = brand.load_mark()
    rows, frames = asset.sizes[30]
    third = asset.frames // 3
    same = sum(frames[0][i] == frames[third][i] for i in range(0, len(frames[0]), 2))
    assert same / (rows * 30) > 0.9


def test_no_color_strips_colour_but_keeps_shape() -> None:
    out = render(brand.banner(120), 120, color=True, no_color=True)
    assert "\x1b[38;" not in out  # NO_COLOR forbids colour; bold is allowed by the spec
    assert out.count("\n") == len(brand.WORDMARK)


def test_piped_output_is_plain_text() -> None:
    out = render(brand.banner(120), 120, color=False)
    assert "\x1b[" not in out
    assert out.splitlines()[0].rstrip() == brand.WORDMARK[0].rstrip()


def test_trit_colours_are_distinct() -> None:
    assert len(set(brand.TRIT.values())) == 3


def test_splash_prints_static_frame_without_tty() -> None:
    from openternary.ui import splash

    buf = io.StringIO()
    console = Console(file=buf, width=120)  # not a terminal -> reduced motion
    assert splash.reduced_motion(console)
    splash.play(seconds=0.1, console=console)
    out = buf.getvalue()
    assert "\x1b[" not in out and "OpenTernary" in out
    assert out.count("\n") >= len(brand.WORDMARK) + 15
