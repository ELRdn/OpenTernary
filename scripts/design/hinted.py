"""Optical sizing: widen gaps to >= ~1.5 braille dots for the target cell size."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import gen

def hinted(cols):
    dot = 2 / (cols * 2)
    g = max(0.032, 1.6 * dot)
    def a(n): return gen._a(n, C=0.235, R=0.44, W=max(0.072, 1.3 * dot), halo=max(0.038, 1.2 * dot), open_deg=24)
    def b(n):  # fewer turns + explicit gap so the bands separate at low resolution
        return gen.style_b(n, A=0.22, H=0.17, rim=0.80, sweep=np.radians(300))
    def c(n): return gen.voronoi_fill(n, A=0.37, H=0.27, rim=0.84, sweep=np.radians(215), gap=g, centre_gap=0.06, centre_r=0.25)
    def d(n): return gen.voronoi_fill(n, A=0.40, H=0.22, rim=0.84, sweep=np.radians(205), gap=g, centre_gap=0.22, centre_r=0.40)
    return {"A  kumihimo": a, "B  uzu": b, "C  big heads": c, "D  kazaguruma": d}
