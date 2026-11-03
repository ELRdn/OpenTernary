"""Vectorised mitsudomoe generator. label(style, n) -> int8 array: -1 bg, 0..2 commas, 3 ring."""
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
TAU = 2 * np.pi
HEADS = [TAU * k / 3 + np.pi / 2 for k in range(3)]

def grid(n):
    c = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(c, -c)
    return x, y, np.hypot(x, y), np.arctan2(y, x)

def comma_seed(x, y, r, phi, th, A, H, rim, sweep, p=0.8):
    head = np.hypot(x - A * np.cos(th), y - A * np.sin(th)) <= H
    d = (th - phi) % TAU
    t = d / sweep
    inner = A + (rim - A) * np.clip(t, 0, 1) ** p
    tail = (d <= sweep) & (r >= inner) & (r <= rim)
    return head | tail

def ring(r, a=0.90, b=0.965):
    return (r >= a) & (r <= b)

def partition(n, A, H, rim, sweep, gap, centre_gap=0.0, centre_r=0.35, p=0.8):
    x, y, r, phi = grid(n)
    px = 2 / n
    dist = np.stack([edt(~comma_seed(x, y, r, phi, th, A, H, rim, sweep, p)) * px for th in HEADS])
    order = np.sort(dist, axis=0)
    lab = np.argmin(dist, axis=0).astype(np.int8)
    g = gap + centre_gap * np.clip(1 - r / centre_r, 0, 1) ** 1.5
    lab[(order[1] - order[0]) < g] = -1
    lab[r > rim] = -1
    lab[ring(r)] = 3
    return lab

def style_c(n):  # big heads, uniform hairlines
    return partition(n, A=0.42, H=0.30, rim=0.845, sweep=np.radians(230), gap=0.030)

def style_d(n):  # pinwheel opening at the centre
    return partition(n, A=0.45, H=0.24, rim=0.845, sweep=np.radians(220), gap=0.030, centre_gap=0.20, centre_r=0.42)

def style_b(n, A=0.21, H=0.165, rim=0.835, sweep=np.radians(340)):
    x, y, r, phi = grid(n)
    lab = np.full((n, n), -1, np.int8)
    b = (rim - A) / sweep
    for k, th in enumerate(HEADS):
        m = np.hypot(x - A * np.cos(th), y - A * np.sin(th)) <= H
        d0 = (th - phi) % TAU
        for k2 in range(2):
            d = d0 + k2 * TAU
            t = np.clip(d / sweep, 0, 1)
            w = H * (1 - t) ** 0.55 + 0.01
            m |= (d <= sweep) & (np.abs(r - (A + b * d)) <= w)
        lab[m & (lab < 0)] = k
    lab[ring(r)] = 3
    return lab

def style_a(n, C=0.27, R=0.40, W=0.085, halo=0.04, open_deg=38):
    x, y, r, phi = grid(n)
    lab = np.full((n, n), -1, np.int8)
    sd, inside = [], []
    for th in HEADS:
        cx, cy = C * np.cos(th), C * np.sin(th)
        rho = np.hypot(x - cx, y - cy)
        ang = np.arctan2(y - cy, x - cx)
        # opening on the outer side of each loop, shifted forward so it reads as a comma
        gap_c = th - np.radians(35)
        dg = np.abs((ang - gap_c + np.pi) % TAU - np.pi)
        band = np.abs(rho - R) - W
        e1 = (cx + R * np.cos(gap_c + np.radians(open_deg)), cy + R * np.sin(gap_c + np.radians(open_deg)))
        e2 = (cx + R * np.cos(gap_c - np.radians(open_deg)), cy + R * np.sin(gap_c - np.radians(open_deg)))
        knob = np.hypot(x - e1[0], y - e1[1]) - W * 1.45   # head: round knob
        cap = np.hypot(x - e2[0], y - e2[1]) - W           # tail: plain round cap
        band = np.where(dg < np.radians(open_deg), np.inf, band)
        s = np.minimum(np.minimum(band, knob), cap)
        sd.append(s)
    sd = np.stack(sd)
    inner = r < 0.27
    def over(i, j):  # i drawn over j at this pixel
        if j == (i + 1) % 3:
            return inner
        return ~inner
    for k in range(3):
        mine = sd[k] <= 0
        for o in range(3):
            if o == k:
                continue
            under = (sd[o] <= halo) & over(o, k)  # o passes over k here (with halo)
            mine &= ~under
        lab[mine] = k
    m = ring(r, 0.90, 0.96)
    for th in HEADS:
        m2 = m & (((th - np.radians(20) - phi) % TAU) < np.radians(85))
        lab[m2] = 3
    return lab

STYLES = {"A  kumihimo / interlace": style_a, "B  uzu / spiral": style_b,
          "C  big heads + hairline": style_c, "D  kazaguruma / pinwheel": style_d}

# ---- v2: spiral-sector partition for filled styles ---------------------------------
def spiral_fill(n, A, H, rim, S, gap, centre_gap=0.0, centre_r=0.35, q=1.0):
    """Sectors bounded by spirals psi = phi + S*r^q; head disks override; gaps by distance to other labels."""
    x, y, r, phi = grid(n)
    psi = phi + S * r**q
    lab = np.full((n, n), -1, np.int8)
    for k, th in enumerate(HEADS):
        base = th + S * A**q
        dd = np.abs((psi - base - np.radians(60) + np.pi) % TAU - np.pi)  # sector trails the head
        lab[(dd <= np.radians(60)) & (r <= rim)] = k
    for k, th in enumerate(HEADS):
        lab[np.hypot(x - A * np.cos(th), y - A * np.sin(th)) <= H] = k
    px = 2 / n
    out = np.full_like(lab, -1)
    g = gap + centre_gap * np.clip(1 - r / centre_r, 0, 1) ** 1.5
    for k in range(3):
        d = edt(lab == k) * px
        out[(lab == k) & (d >= g / 2)] = k
    out[ring(r)] = 3
    return out

def style_c(n):
    return spiral_fill(n, A=0.40, H=0.30, rim=0.855, S=-2.6, gap=0.035)

def style_d(n):
    return spiral_fill(n, A=0.47, H=0.235, rim=0.855, S=-2.2, gap=0.035, centre_gap=0.22, centre_r=0.40)

_a = style_a
def style_a(n):
    return _a(n, C=0.235, R=0.44, W=0.072, halo=0.038, open_deg=24)

STYLES = {"A  kumihimo / interlace": style_a, "B  uzu / spiral": style_b,
          "C  big heads + hairline": style_c, "D  kazaguruma / pinwheel": style_d}

# ---- v3: Voronoi fill around the original (good-looking) separated commas ----------
def tomoe_seed(x, y, r, phi, th, A, H, tip, sweep):
    head = np.hypot(x - A * np.cos(th), y - A * np.sin(th)) <= H
    d = (th - phi) % TAU
    t = np.clip(d / sweep, 0, 1)
    outer = (A + H) + (tip - (A + H)) * np.sin(t * np.pi / 2)
    inner = A + (tip - A) * t**0.75
    return head | ((d <= sweep) & (r >= inner) & (r <= outer))

def voronoi_fill(n, A, H, rim, sweep, gap, centre_gap=0.0, centre_r=0.35):
    x, y, r, phi = grid(n)
    px = 2 / n
    dist = np.stack([edt(~tomoe_seed(x, y, r, phi, th, A, H, rim, sweep)) * px for th in HEADS])
    lab = np.argmin(dist, axis=0).astype(np.int8)
    lab[r > rim] = -1
    out = np.full_like(lab, -1)
    g = gap + centre_gap * np.clip(1 - r / centre_r, 0, 1) ** 1.5
    for k in range(3):
        d = edt(lab == k) * px
        out[(lab == k) & (d >= g / 2)] = k
    out[ring(r)] = 3
    return out

def style_c(n):
    return voronoi_fill(n, A=0.37, H=0.27, rim=0.86, sweep=np.radians(215), gap=0.032, centre_gap=0.06, centre_r=0.25)

def style_d(n):
    return voronoi_fill(n, A=0.40, H=0.22, rim=0.86, sweep=np.radians(205), gap=0.032, centre_gap=0.20, centre_r=0.38)

STYLES = {"A  kumihimo / interlace": style_a, "B  uzu / spiral": style_b,
          "C  big heads + hairline": style_c, "D  kazaguruma / pinwheel": style_d}
