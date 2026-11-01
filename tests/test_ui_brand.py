"""Brand assets render deterministically and degrade on narrow / colourless terminals."""

from __future__ import annotations

import io

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
    assert brand.banner(80).plain.endswith("OPEN-TERNARY")
    assert brand.banner(120).plain.splitlines()[0] == brand.WORDMARK[0]


def test_mark_dimensions_and_symmetry() -> None:
    for cols in (16, 26, 30):
        lines = brand.mark(cols).plain.splitlines()
        assert len(lines) == cols // 2
        assert all(len(line) == cols for line in lines)
        assert all(ch == " " or 0x2800 <= ord(ch) <= 0x28FF for line in lines for ch in line)


def test_mark_uses_all_three_trit_colours() -> None:
    styles = {str(span.style) for span in brand.mark(26).spans}
    for colour in brand.TRIT.values():
        assert any(colour in s for s in styles), colour


def test_no_color_strips_colour_but_keeps_shape() -> None:
    out = render(brand.banner(120), 120, color=True, no_color=True)
    assert "\x1b[38;" not in out  # NO_COLOR forbids colour; bold is allowed by the spec
    assert out.count("\n") == len(brand.WORDMARK)


def test_piped_output_is_plain_text() -> None:
    out = render(brand.banner(120), 120, color=False)
    assert "\x1b[" not in out
    assert out.splitlines()[0].rstrip() == brand.WORDMARK[0].rstrip()
