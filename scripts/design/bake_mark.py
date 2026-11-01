"""Bake the official OpenTernary symbol (docs/brand/openternary-v1.0.svg) into a package asset.

Design-time only: needs numpy, scipy, pillow and Node + Playwright (Chromium) for exact SVG
rasterisation. Output: src/openternary/ui/assets/tomoe.otmk

The SVG is rendered twice: as authored (colours) and with flat per-element fills (labels:
+1 / 0 / -1 tomoe and the ring). Frames are rigid rotations of both, eroded per region so the
gaps survive Braille resolution, then encoded per terminal cell.

Format (zlib-compressed):
  b"OTMK" | u8 version=1 | u8 frames | u8 n_sizes | u8 n_palette
  palette: n_palette x (u8 r, u8 g, u8 b, u8 band)   band: 0=+1, 1=0, 2=-1, 3=ring
  per size: u8 cols | u8 rows | frames x rows x cols x (u8 braille_bits, u8 palette_index; 255 = empty)
Frame f is the symbol rotated counter-clockwise (on screen) by f * 360 / frames degrees.
"""

from __future__ import annotations

import os
import pathlib
import struct
import subprocess
import tempfile
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[2]
SVG = ROOT / "docs" / "brand" / "openternary-v1.0.svg"
OUT = ROOT / "src" / "openternary" / "ui" / "assets" / "tomoe.otmk"
FRAMES, SIZES, S = 48, (16, 24, 30), 8
BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])
PER_BAND = 16
VIEW, CROP = 1024, 452  # SVG viewBox; half-size of the square kept around the centre

SHOT_JS = """
const { chromium } = require('playwright');
(async () => {
  const [svg, png] = process.argv.slice(2);
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1024, height: 1024 } });
  await p.goto('file://' + svg);
  await p.screenshot({ path: png, omitBackground: true, clip: { x: 0, y: 0, width: 1024, height: 1024 } });
  await b.close();
})();
"""


def rasterise(svg_text: str, tmp: pathlib.Path, name: str) -> np.ndarray:
    svg, png, js = tmp / f"{name}.svg", tmp / f"{name}.png", tmp / "shot.js"
    svg.write_text(svg_text, encoding="utf-8")
    js.write_text(SHOT_JS, encoding="utf-8")
    env = {**os.environ, "NODE_PATH": subprocess.check_output(["npm", "root", "-g"], text=True).strip()}
    subprocess.run(["node", str(js), str(svg), str(png)], check=True, env=env)
    return np.asarray(Image.open(png).convert("RGBA"))


def sources() -> tuple[np.ndarray, np.ndarray]:
    src = SVG.read_text(encoding="utf-8")
    flat = src.replace('stroke="url(#spectrum)"', 'stroke="#ffffff"')
    for grad, colour in (("positive", "#ff0000"), ("zero", "#00ff00"), ("negative", "#0000ff")):
        flat = flat.replace(f'fill="url(#{grad})"', f'fill="{colour}"')
    with tempfile.TemporaryDirectory() as d:
        colour_px = rasterise(src, pathlib.Path(d), "colour")
        label_px = rasterise(flat, pathlib.Path(d), "label")
    c0, c1 = VIEW // 2 - CROP, VIEW // 2 + CROP
    colour_px, label_px = colour_px[c0:c1, c0:c1], label_px[c0:c1, c0:c1]
    lab = np.full(label_px.shape[:2], -1, np.int8)
    on = label_px[..., 3] >= 128
    rgb = label_px[..., :3].astype(int)
    lab[on & (rgb[..., 0] > 128) & (rgb[..., 1] > 128) & (rgb[..., 2] > 128)] = 3
    lab[on & (lab < 0) & (rgb[..., 0] >= rgb[..., 1]) & (rgb[..., 0] >= rgb[..., 2])] = 0
    lab[on & (lab < 0) & (rgb[..., 1] >= rgb[..., 2])] = 1
    lab[on & (lab < 0)] = 2
    return lab, colour_px[..., :3] / 255


def encode(lab: np.ndarray, rgb: np.ndarray, cols: int) -> np.ndarray:
    rows = cols // 2
    cells = np.zeros((rows, cols, 5))  # bits, r, g, b, band
    for cy in range(rows):
        for cx in range(cols):
            bits, acc, votes = 0, [], np.zeros(4)
            for py in range(4):
                for px in range(2):
                    ys, xs = (cy * 4 + py) * S, (cx * 2 + px) * S
                    blk = lab[ys : ys + S, xs : xs + S]
                    on = blk >= 0
                    if on.mean() > 0.45:
                        bits |= int(BITS[py, px])
                        acc.append(rgb[ys : ys + S, xs : xs + S][on].mean(0))
                        votes += np.bincount(blk[on], minlength=4)[:4]
            if bits:
                cells[cy, cx] = (bits, *np.mean(acc, 0), int(votes.argmax()))
    return cells


def main() -> None:
    lab_hr, rgb_hr = sources()
    baked = {}
    for cols in SIZES:
        n = cols * 2 * S
        lab0 = np.asarray(Image.fromarray(lab_hr.astype(np.int16)).resize((n, n), Image.NEAREST)).astype(np.int8)
        rgb0 = np.asarray(Image.fromarray((rgb_hr * 255).astype(np.uint8)).resize((n, n), Image.LANCZOS)) / 255
        e = max(0, int(round(0.6 * S - 0.5)))
        frames = []
        for f in range(FRAMES):
            ang = 360 * f / FRAMES  # positive = counter-clockwise on screen
            lab = ndi.rotate(lab0, ang, reshape=False, order=0, cval=-1, mode="constant")
            rgb = ndi.rotate(rgb0, ang, axes=(1, 0), reshape=False, order=1, mode="nearest")
            hinted = np.full_like(lab, -1)
            for k in range(4):
                hinted[ndi.binary_erosion(lab == k, iterations=e)] = k
            frames.append(encode(hinted, rgb, cols))
        baked[cols] = np.stack(frames)
    palette, lut = [], {}
    for band in range(4):
        cols_rgb = np.concatenate([b[(b[..., 0] > 0) & (b[..., 4] == band)][:, 1:4] for b in baked.values()])
        img = Image.fromarray((cols_rgb[None] * 255).clip(0, 255).astype(np.uint8))
        q = img.quantize(colors=PER_BAND, method=Image.Quantize.MEDIANCUT)
        pal = np.array(q.getpalette()[: PER_BAND * 3]).reshape(-1, 3)
        lut[band] = (len(palette), pal)
        palette += [(*map(int, p), band) for p in pal]
    out = bytearray(b"OTMK" + struct.pack("BBBB", 1, FRAMES, len(SIZES), len(palette)))
    for p in palette:
        out += struct.pack("BBBB", *p)
    for cols, b in baked.items():
        rows = cols // 2
        out += struct.pack("BB", cols, rows)
        for f in range(FRAMES):
            for cy in range(rows):
                for cx in range(cols):
                    bits, r, g, bl, band = b[f, cy, cx]
                    if bits == 0:
                        out += bytes((0, 255))
                        continue
                    base, pal = lut[int(band)]
                    idx = base + int(np.argmin(((pal - np.array([r, g, bl]) * 255) ** 2).sum(1)))
                    out += bytes((int(bits), idx))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(zlib.compress(bytes(out), 9))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, raw {len(out)})")


if __name__ == "__main__":
    main()
