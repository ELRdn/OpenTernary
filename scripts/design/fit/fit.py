import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import differential_evolution
from ref import masks
import model_b
which = sys.argv[1]
N = 120 if sys.argv[1] == "B" else 64
ref = masks(N)[which]
mod = {"B": model_b}[which] if which == "B" else __import__("model_a")
render, bounds = (mod.render_b, mod.BOUNDS) if which == "B" else (mod.render_a, mod.BOUNDS)

def iou(P):
    P = list(P)
    if which == "B": P[8] = 1 if P[8] >= 0 else -1
    m = render(N, P)
    return -(m & ref).sum() / max((m | ref).sum(), 1)

import multiprocessing as mp
pool = mp.get_context('fork').Pool(4)
res = differential_evolution(iou, bounds, maxiter=int(sys.argv[2]) if len(sys.argv) > 2 else 120,
                             popsize=24 if which == "B" else 16, tol=1e-6, seed=7, polish=True, workers=pool.map, updating="deferred")
pool.close()
P = list(res.x)
if which == "B": P[8] = 1 if P[8] >= 0 else -1
# report IoU at higher resolution too
ref2 = masks(240)[which]; m2 = render(240, P)
print("IoU@120", -res.fun, "IoU@240", (m2 & ref2).sum() / (m2 | ref2).sum())
json.dump([float(v) for v in P], open(fstr(__import__("pathlib").Path(__file__).resolve().parent) + "/params_{which}.json", "w"))
from PIL import Image
viz = np.zeros((240, 240, 3), np.uint8)
viz[ref2 & m2] = (235, 235, 235); viz[ref2 & ~m2] = (230, 60, 60); viz[~ref2 & m2] = (60, 160, 255)
Image.fromarray(viz).save(f"diff_{which}.png")
