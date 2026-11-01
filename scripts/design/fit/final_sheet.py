import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import model_a, model_b, ref
from rainbow import colourise3 as colourise2

PA = json.load(open(str(__import__("pathlib").Path(__file__).resolve().parent) + "/params_A.json")); PB = json.load(open(str(__import__("pathlib").Path(__file__).resolve().parent) + "/params_B.json"))

def labels(which, n, step=40):
    if which == "B":
        return model_b.render_b(n, PB, labels=True)
    c = (np.arange(n) + 0.5) / n * 2 - 1; X, Y = np.meshgrid(c, c)
    out = np.empty((n, n), np.int8); orig = model_a.grid
    for i in range(0, n, step):
        model_a.grid = lambda _n, i=i: (X[i:i + step], Y[i:i + step])
        out[i:i + step] = model_a.render_a(n, PA, labels=True)
    model_a.grid = orig
    return out

def hint(lab, n, cols):
    """Optical sizing: erode every region so gaps are >= ~1.5 braille dots."""
    dot_px = n / (cols * 2)
    e = max(0, int(round(0.6 * dot_px - 0.5)))
    if e == 0: return lab
    out = np.full_like(lab, -1)
    st = ndi.generate_binary_structure(2, 1)
    for k in range(4):
        out[ndi.binary_erosion(lab == k, st, iterations=e)] = k
    return out

if __name__ == "__main__":
    N, S, PAD = 300, 3, 26
    BG = np.array([14, 15, 19]) / 255
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    sheet = Image.new("RGB", (3 * (N + PAD) + PAD, 2 * (N + PAD + 24) + PAD), (14, 15, 19))
    dr = ImageDraw.Draw(sheet)
    R = ref.masks(N)
    for row, w in enumerate("AB"):
        y0 = PAD + row * (N + PAD + 24)
        refimg = Image.fromarray(np.where(R[w], 0, 255).astype(np.uint8)).convert("RGB")
        lab = labels(w, N * S)
        sil = Image.fromarray(np.where(lab >= 0, 0, 255).astype(np.uint8)).resize((N, N), Image.LANCZOS).convert("RGB")
        rgb, m = colourise2(lab)
        col = Image.fromarray((np.where(m[..., None], rgb, BG) * 255).astype(np.uint8)).resize((N, N), Image.LANCZOS)
        for i, (im, t) in enumerate([(refimg, f"{w}: reference (83 px source)"), (sil, f"{w}: geometric rebuild"), (col, f"{w}: rainbow, 3 colours")]):
            x0 = PAD + i * (N + PAD); sheet.paste(im, (x0, y0)); dr.text((x0 + 4, y0 + N + 4), t, fill=(220, 220, 225), font=font)
    sheet.save(sys.argv[1])
