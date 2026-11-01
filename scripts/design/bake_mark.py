"""Bake the chosen mark (style A, rainbow, CCW rotation frames) into a package asset.

Design-time only (numpy/scipy/pillow). Output: src/openternary/ui/assets/tomoe_a.otmk

Format (zlib-compressed):
  b"OTMK" | u8 version=1 | u8 frames | u8 n_sizes | u8 n_palette
  palette: n_palette x (u8 r, u8 g, u8 b, u8 band)
  per size: u8 cols | u8 rows | frames x rows x cols x (u8 braille_bits, u8 palette_index; 255 = empty)
Frame f is the mark rotated counter-clockwise (on screen) by f * 360 / frames degrees.
"""

from __future__ import annotations

import pathlib
import struct
import sys
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fit"))
sys.path.insert(0, str(HERE))
from final_sheet import labels  # noqa: E402
from rainbow import colourise3  # noqa: E402

FRAMES, SIZES, S = 48, (16, 24, 30), 8
OUT = HERE.parents[1] / "src" / "openternary" / "ui" / "assets" / "tomoe_a.otmk"
BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])
PER_BAND = 21


def encode(lab: np.ndarray, rgb: np.ndarray, cols: int) -> np.ndarray:
    rows = cols // 2
    cells = np.zeros((rows, cols, 5))  # bits, r, g, b, band
    for cy in range(rows):
        for cx in range(cols):
            bits, acc, votes = 0, [], np.zeros(3)
            for py in range(4):
                for px in range(2):
                    ys, xs = (cy * 4 + py) * S, (cx * 2 + px) * S
                    blk = lab[ys : ys + S, xs : xs + S]
                    on = blk >= 0
                    if on.mean() > 0.45:
                        bits |= int(BITS[py, px])
                        acc.append(rgb[ys : ys + S, xs : xs + S][on].mean(0))
                        votes += np.bincount(blk[on], minlength=3)[:3]
            if bits:
                cells[cy, cx] = (bits, *np.mean(acc, 0), int(votes.argmax()))
    return cells


def main() -> None:
    hr = 960
    lab_hr = labels("A", hr)
    rgb_hr, _ = colourise3(lab_hr)
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
            for k in range(3):
                hinted[ndi.binary_erosion(lab == k, iterations=e)] = k
            frames.append(encode(hinted, rgb, cols))
        baked[cols] = np.stack(frames)
    # per-band palette quantisation (keeps band identity in the palette)
    palette, lut = [], {}
    for band in range(3):
        cols_rgb = np.concatenate([b[(b[..., 0] > 0) & (b[..., 4] == band)][:, 1:4] for b in baked.values()])
        img = Image.fromarray((cols_rgb[None] * 255).clip(0, 255).astype(np.uint8))
        q = img.quantize(colors=PER_BAND, method=Image.Quantize.MEDIANCUT)
        pal = np.array(q.getpalette()[: PER_BAND * 3]).reshape(-1, 3)
        base = len(palette)
        palette += [(*map(int, p), band) for p in pal]
        lut[band] = (base, pal)
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
