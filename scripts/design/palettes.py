import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from rainbow import oklch

def hx(L, C, h):
    c = oklch(L, C, np.array([h], float))[0]
    return "#{:02x}{:02x}{:02x}".format(*(int(round(v * 255)) for v in c))

def lum(h):
    c = np.array([int(h[i:i+2], 16) for i in (1, 3, 5)]) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(c @ [0.2126, 0.7152, 0.0722])

def contrast(a, b):
    la, lb = sorted([lum(a), lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)

# Machado et al. 2009, severity 1.0 (linear RGB)
CVD = {
 "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
 "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
 "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}
def lin(h):
    c = np.array([int(h[i:i+2], 16) for i in (1, 3, 5)]) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def oklab(rgb_lin):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]])
    return M2 @ np.cbrt(M1 @ np.clip(rgb_lin, 0, 1))
def de(a, b, kind=None):
    la, lb = lin(a), lin(b)
    if kind: la, lb = CVD[kind] @ la, CVD[kind] @ lb
    return float(np.linalg.norm(oklab(la) - oklab(lb)) * 100)

def build(name, jp, mode, bgLCh, surf_dL, border_dL, textL, mutedL, statusL, statusC, accent):
    L, C, h = bgLCh
    sign = 1 if mode == "dark" else -1
    p = {
        "bg": hx(L, C, h), "surface": hx(L + sign * surf_dL, C, h), "border": hx(L + sign * border_dL, C * 1.4, h),
        "text": hx(textL, min(C, 0.012), h), "muted": hx(mutedL, C * 1.6, h), "accent": hx(*accent),
        "ok": hx(statusL + (0.04 if sign > 0 else -0.08), statusC, 195), "warn": hx(statusL + sign * 0.02, statusC, 80), "err": hx(statusL + (-0.10 if sign > 0 else 0.0), statusC * 1.25, 32),
        "info": hx(statusL, statusC, 250),
    }
    return {"name": name, "jp": jp, "mode": mode, "colors": p}

P = [
    build("Sumi", "墨", "dark", (0.17, 0.006, 260), 0.035, 0.12, 0.93, 0.66, 0.80, 0.13, (0.82, 0.10, 250)),
    build("Yoru", "夜", "dark", (0.16, 0.035, 275), 0.045, 0.14, 0.94, 0.70, 0.80, 0.13, (0.76, 0.15, 295)),
    build("Shiro", "白", "light", (0.975, 0.006, 90), -0.02, 0.12, 0.22, 0.50, 0.52, 0.15, (0.50, 0.17, 280)),
]
for p in P:
    c = p["colors"]; bg = c["bg"]
    p["checks"] = {
        "text/bg": round(contrast(c["text"], bg), 2), "muted/bg": round(contrast(c["muted"], bg), 2),
        **{f"{k}/bg": round(contrast(c[k], bg), 2) for k in ("ok", "warn", "err", "info", "accent")},
        "ok-vs-err dE (normal/protan/deutan/tritan)": [round(de(c["ok"], c["err"], k), 1) for k in (None, "protan", "deutan", "tritan")],
    }
json.dump(P, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
for p in P:
    print(p["name"], p["mode"], p["colors"]); print("   ", p["checks"])
