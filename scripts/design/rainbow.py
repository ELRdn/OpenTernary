import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from gen import grid, HEADS, TAU

def oklch_to_srgb(L, C, h_deg):
    h = np.radians(h_deg)
    a, b = C * np.cos(h), C * np.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    lin = np.stack([r, g, bb], -1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    return srgb

def in_gamut(L, C, h):
    s = oklch_to_srgb(L, C, h)
    return np.all((s >= -1e-4) & (s <= 1 + 1e-4), axis=-1)

def oklch(L, C, h):
    """Gamut-map by reducing chroma (keeps lightness and hue)."""
    h = np.asarray(h, float); Cc = np.full_like(h, C)
    for _ in range(30):
        bad = ~in_gamut(L, Cc, h)
        if not bad.any(): break
        Cc = np.where(bad, Cc * 0.95, Cc)
    return np.clip(oklch_to_srgb(L, Cc, h), 0, 1)

H0 = 25.0  # hue at the head of comma 0 (warm red)

def colourise(lab, L=0.78, C=0.17, sweep_deg=230, lead_deg=30):
    n = lab.shape[0]
    x, y, r, phi = grid(n)
    rgb = np.zeros((n, n, 3))
    for k, th in enumerate(HEADS):
        d = (th + np.radians(lead_deg) - phi) % TAU
        t = np.clip(d / np.radians(sweep_deg + lead_deg), 0, 1)
        hue = H0 + 120 * k + 120 * t
        m = lab == k
        rgb[m] = oklch(L, C, hue[m])
    ring = lab == 3
    rgb[ring] = oklch(0.80, 0.10, (np.degrees(phi[ring]) * -1 + 90 + H0) % 360) * 0.55 + 0.45 * np.array([0.42, 0.43, 0.47])
    return rgb, lab >= 0

from scipy.ndimage import distance_transform_edt as edt, zoom
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def geodesic_t(mask, coarse=200):
    """0 at the fattest point (the head), 1 at the farthest tail pixel, measured inside the shape."""
    n = mask.shape[0]
    f = n // coarse
    m = mask[: coarse * f, : coarse * f].reshape(coarse, f, coarse, f).mean(axis=(1, 3)) > 0.3
    idx = -np.ones(m.shape, int); ys, xs = np.nonzero(m); idx[ys, xs] = np.arange(len(ys))
    rows, cols, w = [], [], []
    for dy, dx, c in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < coarse) & (x2 >= 0) & (x2 < coarse)
        ok[ok] &= m[y2[ok], x2[ok]]
        rows += list(idx[ys[ok], xs[ok]]); cols += list(idx[y2[ok], x2[ok]]); w += [c] * int(ok.sum())
    g = coo_matrix((w, (rows, cols)), shape=(len(ys),) * 2).tocsr()
    src = int(np.argmax(edt(m)[ys, xs]))
    d = dijkstra(g, directed=False, indices=src)
    d[~np.isfinite(d)] = 0
    full = np.zeros(m.shape); full[ys, xs] = d / max(d.max(), 1e-9)
    # dilate values a little so upsampled edges are covered, then upsample
    t = zoom(full, f, order=1)
    out = np.zeros(mask.shape); out[: t.shape[0], : t.shape[1]] = t
    return out

def colourise2(lab, L=0.74, C=0.19, h0=H0):
    n = lab.shape[0]
    rgb = np.zeros((n, n, 3))
    for k in range(3):
        m = lab == k
        t = geodesic_t(m)
        rgb[m] = oklch(L, C, h0 + 120 * k + 120 * np.clip(t[m], 0, 1))
    ring = lab == 3
    rgb[ring] = np.array([0.36, 0.37, 0.42])
    return rgb, lab >= 0
