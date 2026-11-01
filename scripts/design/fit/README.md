# Geometric rebuild of the chosen mitsudomoe (A: interlace, B: spiral)

The marks are parametric models (`model_a.py`: Bezier necks + tapering outer arcs with
over/under weave and halo gaps; `model_b.py`: spiral tails + offset heads + outer ring).
Parameters were fitted to the owner's reference image by maximising IoU
(differential evolution, then Nelder-Mead). The reference image is **not** committed.

| | IoU vs reference (240 px) | reference self-consistency (threshold .35 vs .65) |
|---|---|---|
| A | 0.859 | 0.875 |
| B | 0.898 | 0.866 |

The fits sit at the reference's own edge-uncertainty floor (83 px source), so further
fidelity needs a higher-resolution / vector source.

- Render without the reference: `python final_sheet.py OUT.png` needs `TOMOE_REFERENCE`
  only for the reference column; `term_sheet.py OUT.svg` renders terminal Braille from
  `params_*.json` alone.
- Refit: `TOMOE_REFERENCE=ref.jpg python fit.py A 200 && python polish.py A 120 700`
