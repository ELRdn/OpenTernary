# Brand design tooling (draft)

Generates the mitsudomoe mark candidates, the perceptual rainbow, and the UI palette
candidates used in `docs/TUI_DESIGN.md`. Design-time only: needs `numpy scipy pillow`
(not runtime dependencies). The final mark will be baked into `src/openternary/ui/brand.py`.

| script | output |
|---|---|
| `sheet_styles.py OUT.png` | 4 styles, high resolution, rainbow |
| `sheet_terminal_raw.py OUT.svg` | same styles as terminal Braille (no hinting) |
| `sheet_terminal.py OUT.svg` | Braille with optical sizing (gaps >= 1.6 dots) |
| `palettes.py palettes.json` | 3 UI palettes + WCAG contrast + CVD (Machado 2009) checks |
| `palette_mock.py NAME OUT.svg` | home-screen mock in palette NAME (Sumi / Yoru / Shiro) |
