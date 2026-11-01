import numpy as np
TAU = 2 * np.pi

def grid(n):
    c = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(c, c)          # image coords: y down (matches reference masks)
    return X, Y, np.hypot(X, Y), np.arctan2(Y, X)

def render_b(n, P, labels=False):
    A, H, R1, sweep, p, q, wend, th0, s, ri, ro, Ah, dh, Hh = P[:14]
    X, Y, r, phi = grid(n)
    lab = np.full((n, n), -1, np.int8)
    for k in range(3):
        th = th0 + TAU * k / 3
        m = np.hypot(X - Ah * np.cos(th + s * dh), Y - Ah * np.sin(th + s * dh)) <= Hh
        d0 = (s * (th - phi)) % TAU
        for w in range(3):
            d = d0 + w * TAU
            t = np.clip(d / sweep, 0, 1)
            rc = A + (R1 - A) * t**p
            half = (H - wend) * (1 - t) ** q + wend
            m |= (d <= sweep) & (np.abs(r - rc) <= half)
        lab[m & (lab < 0)] = k
    lab[(r >= ri) & (r <= ro) & (lab < 0)] = 3
    return lab if labels else lab >= 0

BOUNDS = [(0.12, 0.40), (0.10, 0.26), (0.70, 0.97), (np.radians(250), np.radians(560)),
          (0.5, 1.6), (0.2, 2.0), (0.01, 0.10), (-np.pi, np.pi), (-1, 1), (0.80, 0.97), (0.95, 1.02), (0.15, 0.45), (-0.8, 0.8), (0.10, 0.26)]
