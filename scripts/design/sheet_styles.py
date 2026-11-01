import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gen import STYLES
from rainbow import colourise2 as colourise
N, SS, PAD = 300, 3, 30
BG = np.array([0.055, 0.06, 0.075])
sheet = Image.new("RGB", (len(STYLES) * (N + PAD) + PAD, N + 2 * PAD + 26), tuple(int(v*255) for v in BG))
dr = ImageDraw.Draw(sheet)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)

for i, (name, f) in enumerate(STYLES.items()):
    lab = f(N * SS)
    rgb, m = colourise(lab)
    img = np.where(m[..., None], rgb, BG)
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((N, N), Image.LANCZOS)
    x0 = PAD + i * (N + PAD)
    sheet.paste(im, (x0, PAD)); dr.text((x0 + 8, N + PAD + 4), name, fill=(220, 220, 220), font=font)
sheet.save(sys.argv[1])
