"""Status vocabulary -> glyph + colour (Yoru palette). Only glyphs are coloured; labels keep the
terminal's own foreground so output stays readable on light and dark backgrounds, and meaning
never depends on colour alone (glyph + words are always shown).

Honesty rule from docs/CLI.md: ``completed`` (execution) is not ``accepted`` (quality), so it is
never drawn in the success colour.
"""

from __future__ import annotations

from rich.text import Text

from openternary.ui import brand

OK, WARN, ERR, INFO, MUTED = "ok", "warn", "err", "info", "muted"
COLOURS = {OK: brand.OK, WARN: brand.WARN, ERR: brand.ERR, INFO: brand.INFO, MUTED: brand.MUTED}

# value -> (glyph, tone)
STATUS: dict[str, tuple[str, str]] = {
    "accepted": ("✔", OK),
    "passed": ("✔", OK),
    "pass": ("✔", OK),
    "implemented": ("✔", OK),
    "installed": ("●", OK),
    "file_integrity": ("●", INFO),
    "completed": ("●", INFO),
    "running": ("◐", INFO),
    "planned": ("○", INFO),
    "experimental": ("▲", WARN),
    "unverified": ("◐", WARN),
    "implemented_model_unverified": ("◐", WARN),
    "model_validation_pending": ("◐", WARN),
    "not_installed": ("○", MUTED),
    "not_run": ("○", MUTED),
    "not_probed": ("○", MUTED),
    "unsupported": ("–", MUTED),
    "unknown": ("?", MUTED),
    "budget_exhausted": ("▲", WARN),
    "no_feasible_candidate": ("▲", WARN),
    "failed": ("✖", ERR),
    "fail": ("✖", ERR),
    "interrupted": ("✖", ERR),
    "rejected": ("✖", ERR),
}


def badge(value: object) -> Text:
    """Glyph (coloured) + the original value (uncoloured, underscores as spaces)."""
    if value is None:
        return Text("—", style="dim")
    key = str(value)
    glyph, tone = STATUS.get(key, ("·", MUTED))
    out = Text()
    out.append(glyph + " ", style=COLOURS[tone])
    out.append(key.replace("_", " "))
    return out
