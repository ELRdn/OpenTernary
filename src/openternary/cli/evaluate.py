"""Profile-driven diffusion evaluator and explicitly selected evaluator plugins."""

from pathlib import Path
from typing import Annotated

import typer

from openternary.cli.common import app
from openternary.cli.product import display
from openternary.cli.protocol import Command
from openternary.config.loader import load_config
from openternary.experiment.metadata import write_json
from openternary.experiment.run import create_run
from openternary.services.diffusion import DiffusionProfile, evaluate_diffusion


@app.command("evaluate", cls=Command)
def evaluate(
    model: Path,
    profile: Annotated[Path, typer.Option("--profile")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o")] = None,
    component_artifact: Annotated[Path | None, typer.Option("--component-artifact")] = None,
    evaluator: Annotated[str, typer.Option("--evaluator")] = "diffusion",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Evaluate from a fixed profile; image proxies never imply human acceptance."""
    import json

    cfg = load_config(config, {"model.id": str(model), "output": output})
    raw_profile = json.loads(profile.read_text(encoding="utf-8"))
    plugin = None
    if evaluator == "diffusion":
        parsed = DiffusionProfile.model_validate(raw_profile)
        if bool(parsed.component) != bool(component_artifact):
            raise ValueError("profile.component and --component-artifact must be supplied together")
        validated_profile = parsed.model_dump()
    else:
        from openternary.services.plugins import load_plugin

        plugin = load_plugin("evaluators", evaluator)
        validated_profile = plugin.validate(raw_profile)
    if dry_run:
        display({"status": "planned", "profile": validated_profile, "model_validation": "not_run"})
        return
    run, run_id = create_run(cfg)
    if plugin:
        report = plugin.evaluate(cfg, model, validated_profile, run)
        write_json(run / "evaluation.json", report)
    else:
        report = evaluate_diffusion(cfg, model, parsed, run, component_artifact)
    write_json(
        run / "metrics.json",
        {"status": "completed", "run_id": run_id, "evaluation": report, "resources": report.get("resources", {})},
    )
    display(report)
