import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import minimize
from ref import masks
import model_a, model_b
which, N = sys.argv[1], int(sys.argv[2])
mod = model_a if which == "A" else model_b
render = mod.render_a if which == "A" else mod.render_b
ref = masks(N)[which]
P0 = json.load(open(fstr(__import__("pathlib").Path(__file__).resolve().parent) + "/params_{which}.json"))
fixed = {12: P0[12]} if which == "A" else {8: P0[8]}
free = [i for i in range(len(P0)) if i not in fixed]
def full(x):
    P = list(P0)
    for i, v in zip(free, x): P[i] = v
    return P
def loss(x):
    m = render(N, full(x)); return -(m & ref).sum() / (m | ref).sum()
r = minimize(loss, [P0[i] for i in free], method="Nelder-Mead", options={"maxiter": int(sys.argv[3]), "xatol": 1e-4, "fatol": 1e-6, "adaptive": True})
P = full(r.x)
ref2 = masks(240)[which]; m2 = render(240, P)
print(f"IoU@{N}", -r.fun, "IoU@240", (m2 & ref2).sum() / (m2 | ref2).sum())
json.dump([float(v) for v in P], open(fstr(__import__("pathlib").Path(__file__).resolve().parent) + "/params_{which}.json", "w"))
from PIL import Image
viz = np.zeros((240, 240, 3), np.uint8)
viz[ref2 & m2] = (235, 235, 235); viz[ref2 & ~m2] = (230, 60, 60); viz[~ref2 & m2] = (60, 160, 255)
Image.fromarray(viz).save(f"diff_{which}.png")
