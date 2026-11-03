#!/usr/bin/env python3
"""Generate the geometrically symmetric OpenTernary mitsudomoe v1.0 SVGs.

The paths are authored mathematically, rather than traced from a raster image.
There are exactly three copies of one original Bezier silhouette, rotated by
120 degrees. No embedded images, fonts, JavaScript, or external dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
CENTER = 512.0
TAU = math.tau


def fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def point(x: float, y: float) -> str:
    return f"{fmt(x)},{fmt(y)}"


def local_spiral(t: float) -> tuple[float, float, float, float, float, float]:
    """Center, tangent, and left normal for a gently widening polar spiral."""
    # A round three-tomoe crest with original thick/comma-shaped heads.
    radius = 172.0 + 215.0 * t
    angle = -math.pi / 2 - math.radians(145.0) * t
    r_prime = 215.0
    a_prime = -math.radians(145.0)
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    dx = r_prime * math.cos(angle) - radius * math.sin(angle) * a_prime
    dy = r_prime * math.sin(angle) + radius * math.cos(angle) * a_prime
    mag = math.hypot(dx, dy)
    tx, ty = dx / mag, dy / mag
    return x, y, tx, ty, -ty, tx


def cubic_segments(points: list[tuple[float, float]]) -> str:
    """Convert sampled points to an editable, smooth cubic-Bezier outline."""
    chunks: list[str] = []
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        chunks.append(f"C{point(*c1)} {point(*c2)} {point(*p2)}")
    return " ".join(chunks)


def original_tomoe_path() -> str:
    samples = 40
    head_half_width = 108.0
    plus: list[tuple[float, float]] = []
    minus: list[tuple[float, float]] = []
    for i in range(samples + 1):
        t = i / samples
        x, y, _tx, _ty, nx, ny = local_spiral(t)
        half_width = head_half_width * (1.0 - t) ** 0.7
        plus.append((CENTER + x + half_width * nx, CENTER + y + half_width * ny))
        minus.append((CENTER + x - half_width * nx, CENTER + y - half_width * ny))

    # Rounded comma head; a half-circle behind the spiral's starting tangent.
    x, y, tx, ty, nx, ny = local_spiral(0.0)
    cap: list[tuple[float, float]] = []
    for i in range(1, 19):
        a = math.pi - i * math.pi / 18
        cap.append(
            (
                CENTER + x + head_half_width * (math.cos(a) * nx - math.sin(a) * tx),
                CENTER + y + head_half_width * (math.cos(a) * ny - math.sin(a) * ty),
            )
        )

    reverse_minus = list(reversed(minus))
    return " ".join(
        (
            "M" + point(*plus[0]),
            cubic_segments(plus),
            "L" + point(*reverse_minus[0]),
            cubic_segments(reverse_minus),
            cubic_segments([reverse_minus[-1], *cap]),
            "Z",
        )
    )


GRADIENTS = '''
    <linearGradient id="spectrum" x1="8%" y1="7%" x2="92%" y2="96%">
      <stop offset="0" stop-color="#FFA36B"/>
      <stop offset=".18" stop-color="#FFD157"/>
      <stop offset=".34" stop-color="#7AEF9C"/>
      <stop offset=".53" stop-color="#38D9D0"/>
      <stop offset=".72" stop-color="#5E8DFF"/>
      <stop offset=".9" stop-color="#AA7EFF"/>
      <stop offset="1" stop-color="#EF8DD0"/>
    </linearGradient>
    <linearGradient id="positive" x1="14%" y1="8%" x2="89%" y2="94%">
      <stop offset="0" stop-color="#FFD669"/>
      <stop offset=".44" stop-color="#FFAD55"/>
      <stop offset="1" stop-color="#FF6D8A"/>
    </linearGradient>
    <linearGradient id="zero" x1="9%" y1="4%" x2="93%" y2="92%">
      <stop offset="0" stop-color="#ADFB7F"/>
      <stop offset=".42" stop-color="#43EFC5"/>
      <stop offset="1" stop-color="#3A9FFF"/>
    </linearGradient>
    <linearGradient id="negative" x1="11%" y1="7%" x2="88%" y2="93%">
      <stop offset="0" stop-color="#75A1FF"/>
      <stop offset=".52" stop-color="#8168FA"/>
      <stop offset="1" stop-color="#E38DE8"/>
    </linearGradient>'''


def make_svg(mono: bool = False) -> str:
    path = original_tomoe_path()
    gradients = "" if mono else GRADIENTS
    ring_fill = "currentColor" if mono else "url(#spectrum)"
    colors = ("currentColor",) * 3 if mono else (
        "url(#positive)", "url(#zero)", "url(#negative)"
    )
    swirls = "\n".join(
        f'    <use xlink:href="#tomoe" transform="rotate({angle} 512 512)" fill="{color}"/>'
        for angle, color in zip((0, 120, 240), colors)
    )
    description = (
        "Monochrome OpenTernary logo: an original symmetrical three-tomoe icon."
        if mono else
        "Official OpenTernary logo v1.0: three mathematically rotated tomoe shapes "
        "inside a circular rainbow ring represent ternary values +1, 0 and -1."
    )
    default_color = ' color="#12182B"' if mono else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg"
  xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1024 1024"
  width="1024" height="1024" role="img" aria-labelledby="title desc"{default_color}>
  <title id="title">OpenTernary symbol</title>
  <desc id="desc">{description}</desc>
  <defs>{gradients}
    <path id="tomoe" d="{path}"/>
  </defs>
  <circle cx="512" cy="512" r="416" fill="none"
    stroke="{ring_fill}" stroke-width="54"/>
{swirls}
</svg>
'''


if __name__ == "__main__":
    for name, is_mono in (("openternary-v1.0.svg", False),
                           ("openternary-v1.0-mono.svg", True)):
        target = HERE / name
        target.write_text(make_svg(mono=is_mono), encoding="utf-8")
        print(target)
