import numpy as np
TAU = 2 * np.pi
M = 40  # samples per band path

def grid(n):
    c = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(c, c)
    return X, Y

def band_path(P, k):
    rk, ak, kr, W, L1, g1, L2, Ra0, Ra1, b0, D, halo, order = P
    rot = TAU * k / 3
    K = np.array([rk * np.cos(ak + rot), rk * np.sin(ak + rot)])
    b = b0 + rot
    S = np.array([Ra0 * np.cos(b), Ra0 * np.sin(b)])
    sd = np.sign(D) if D != 0 else 1.0
    tan = sd * np.array([-np.sin(b), np.cos(b)])          # arc direction at its start
    P1 = K + L1 * np.array([np.cos(g1 + rot), np.sin(g1 + rot)])
    P2 = S - L2 * tan
    t = np.linspace(0, 1, M)[:, None]
    bez = (1 - t) ** 3 * K + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * S
    u = np.linspace(0, 1, M)
    ang = b + D * u
    rad = Ra0 + (Ra1 - Ra0) * u
    arc = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    pts = np.vstack([bez, arc[1:]])
    half = np.concatenate([np.full(M, W), W * (1 - u[1:]) ** 0.9])
    return K, pts, half

def seg_dist(X, Y, pts, half):
    """min over segments of (distance - half width); also arclength param of the winner."""
    A, B = pts[:-1], pts[1:]
    AB = B - A
    L2 = (AB**2).sum(1) + 1e-12
    px, py = X[..., None], Y[..., None]
    t = np.clip(((px - A[:, 0]) * AB[:, 0] + (py - A[:, 1]) * AB[:, 1]) / L2, 0, 1)
    cx, cy = A[:, 0] + t * AB[:, 0], A[:, 1] + t * AB[:, 1]
    d = np.hypot(px - cx, py - cy)
    hw = half[:-1] + t * (half[1:] - half[:-1])
    sdist = d - hw
    i = np.argmin(sdist, -1)
    seglen = np.sqrt(L2); cum = np.concatenate([[0], np.cumsum(seglen)])
    s = (cum[i] + np.take_along_axis(t, i[..., None], -1)[..., 0] * seglen[i]) / cum[-1]
    return np.take_along_axis(sdist, i[..., None], -1)[..., 0], s

def render_a(n, P, labels=False):
    X, Y = grid(n)
    kr, halo, order = P[2], P[11], P[12]
    sd, ss = [], []
    for k in range(3):
        K, pts, half = band_path(P, k)
        d, s = seg_dist(X, Y, pts, half)
        dk = np.hypot(X - K[0], Y - K[1]) - kr
        s = np.where(dk < d, 0.0, s)
        sd.append(np.minimum(d, dk)); ss.append(s)
    sd, ss = np.stack(sd), np.stack(ss)
    key = ss * (1 if order >= 0 else -1)           # smaller key = drawn on top
    lab = np.full(X.shape, -1, np.int8)
    inside = sd <= 0
    for k in range(3):
        mine = inside[k].copy()
        for j in range(3):
            if j == k:
                continue
            over = key[j] < key[k]
            mine &= ~((sd[j] <= halo) & over)      # j passes over k (plus halo gap)
        lab[mine] = k
    return lab if labels else lab >= 0

BOUNDS = [(0.50, 0.78), (np.radians(-110), np.radians(-55)), (0.10, 0.22), (0.05, 0.14),
          (0.15, 1.3), (-np.pi, np.pi), (0.05, 1.2), (0.55, 0.95), (0.75, 1.0),
          (-np.pi, np.pi), (np.radians(-300), np.radians(300)), (0.0, 0.07), (-1, 1)]
