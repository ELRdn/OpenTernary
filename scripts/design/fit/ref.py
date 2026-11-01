"""Extract the two reference silhouettes as unit-circle-normalised binary masks."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import os
SRC = os.environ.get("TOMOE_REFERENCE", "reference.jpg")  # not committed: third-party image

def masks(n=200):
    g = np.asarray(Image.open(SRC).convert("L"), float) / 255
    ink = g < 0.5
    h, w = ink.shape
    out = {}
    for name, (x0, x1) in {"A": (0, w // 2), "B": (w // 2, w)}.items():
        sub = ink[:, x0:x1].copy()
        lab, nl = ndi.label(sub)
        sizes = ndi.sum(sub, lab, range(1, nl + 1))
        keep = np.isin(lab, 1 + np.nonzero(sizes > 30)[0])  # drop specks / neighbours' slivers
        ys, xs = np.nonzero(keep)
        # circle fit by bounding extent (the marks are inscribed in a circle)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        rad = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2
        # resample to n x n over [-1, 1]
        c = (np.arange(n) + 0.5) / n * 2 - 1
        X, Y = np.meshgrid(c, c)
        gy, gx = cy + Y * rad, x0 + cx + X * rad
        val = ndi.map_coordinates(g, [gy, gx], order=1, mode="constant", cval=1.0)
        m = val < 0.5
        m &= np.hypot(X, Y) <= 1.02
        out[name] = m
        out[name + "_meta"] = (cx + x0, cy, rad)
    return out

if __name__ == "__main__":
    m = masks(240)
    Image.fromarray(np.hstack([m["A"], m["B"]]).astype(np.uint8) * 255).save("ref_masks.png")
    print(m["A_meta"], m["B_meta"])
