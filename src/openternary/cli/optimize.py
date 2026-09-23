"""Explicit, bounded search-space execution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from openternary.cli.common import app
from openternary.cli.product import display
from openternary.cli.protocol import Command
from openternary.config.loader import load_config
from openternary.config.schema import QuantizationConfig
from openternary.services.errors import AcceptanceError, BudgetError
from openternary.services.optimize import QualityProfile, SearchBudget, run_search
from openternary.services.planning import build_plan


def parse_bytes(text: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB|TiB)?", text)
    if not match:
        raise ValueError("size must use B, KiB, MiB, GiB or TiB")
    result = int(float(match[1]) * 1024 ** {None: 0, "B": 0, "KiB": 1, "MiB": 2, "GiB": 3, "TiB": 4}[match[2]])
    if result <= 0:
        raise ValueError("size must be positive")
    return result


@app.command("optimize", cls=Command)
def optimize(
    model: str,
    search_space: Annotated[Path, typer.Option("--search-space", help="JSON list of quantization configurations")],
    quality_profile: Annotated[Path, typer.Option("--quality-profile", help="Version-one validation profile JSON")],
    target_vram: Annotated[str, typer.Option("--target-vram")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    objective: Annotated[str, typer.Option()] = "speed",
    max_candidates: Annotated[int, typer.Option(min=1, max=256)] = 4,
    wall_seconds: Annotated[float, typer.Option(min=0.001)] = 3600,
    gpu_seconds: Annotated[float, typer.Option(min=0.001)] = 3600,
    max_artifact_bytes: Annotated[str, typer.Option()] = "20GiB",
    resume: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Search only explicit candidates; never select on the test split."""
    cfg = load_config(config, {"model.id": str(Path(model).resolve())})
    profile = QualityProfile.model_validate_json(quality_profile.read_text(encoding="utf-8"))
    for key in ("dataset", "baseline"):
        value = Path(getattr(profile, key))
        setattr(
            profile,
            key,
            str((quality_profile.parent / value).resolve() if not value.is_absolute() else value.resolve()),
        )
    raw = json.loads(search_space.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("search space must be a non-empty list")
    candidates = [QuantizationConfig.model_validate(item) for item in raw]
    budget = SearchBudget(
        max_candidates=max_candidates,
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
        artifact_bytes=parse_bytes(max_artifact_bytes),
    )
    vram = parse_bytes(target_vram)
    if objective not in {"speed", "size"}:
        raise ValueError("objective must be speed or size")
    if dry_run:
        plans = []
        for candidate in candidates:
            candidate_config = cfg.model_copy(deep=True)
            candidate_config.quantization = candidate
            plans.append(build_plan(candidate_config))
        display(
            {
                "status": "planned",
                "candidates": plans,
                "profile": profile.model_dump(),
                "budget": budget.model_dump(),
                "target_vram_bytes": vram,
                "objective": objective,
            }
        )
        return
    if output is None:
        raise ValueError("optimize requires --output")
    result = run_search(cfg, profile, candidates, output, budget, target_vram=vram, objective=objective, resume=resume)
    display(result)
    if result["status"] == "budget_exhausted":
        raise BudgetError("search stopped at the configured budget; no best artifact was published")
    if result["status"] == "no_feasible_candidate":
        raise AcceptanceError("no candidate met the measured resource and quality constraints")
