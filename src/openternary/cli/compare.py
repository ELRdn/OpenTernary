from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.common import _resolve_cli_overrides, _write_failed_metrics, app, console
from openternary.cli.protocol import Command


@app.command("compare", cls=Command)
def compare(
    baseline: Annotated[str | None, typer.Argument(help="Baseline run directory")] = None,
    quantized: Annotated[str | None, typer.Argument(help="Quantized run directory")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    candidate: Annotated[
        list[pathlib.Path] | None, typer.Option("--candidate", help="Additional candidate (repeatable)")
    ] = None,
    require_acceptance: Annotated[bool, typer.Option("--require-acceptance")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Compare benchmark runs — behavioral delta (observational)."""
    from openternary.config.loader import dump_config_yaml, load_config
    from openternary.experiment.metadata import write_json
    from openternary.experiment.run import create_run
    from openternary.services.evaluation import compare_many
    from openternary.utils.logging import setup_logging
    from openternary.utils.seed import seed_everything

    overrides = _resolve_cli_overrides(seed, device, dtype, output)
    try:
        cfg = load_config(config_path=config, cli_overrides=overrides)
    except FileNotFoundError as e:
        console.print(f"[red]Config file not found:[/red] {e}")
        raise typer.Exit(2) from e
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(2) from e

    if dry_run:
        console.print("[bold cyan]compare: dry-run — resolved config[/bold cyan]")
        console.print(dump_config_yaml(cfg))
        console.print("[dim]No run directory was created (--dry-run).[/dim]")
        raise typer.Exit(0)

    if baseline is None or quantized is None:
        console.print("[red]compare requires two run directories: baseline and quantized[/red]")
        console.print("[dim]Usage: openternary compare <baseline_dir> <quantized_dir>[/dim]")
        raise typer.Exit(2)

    run_dir, run_id = create_run(cfg)
    logger = setup_logging(run_dir, verbose=verbose)
    logger.info(f"compare — run_id={run_id} baseline={baseline} quantized={quantized}")
    seed_everything(cfg.seed)

    import pathlib as _pl

    try:
        multiple = compare_many(_pl.Path(baseline), [_pl.Path(quantized), *(candidate or [])])
        result = multiple["comparisons"][0] if not candidate else multiple
        from openternary.services.reporting import publish

        publish(quality_acceptance=multiple["quality_acceptance"], comparison=multiple)
    except (ValueError, OSError) as exc:
        _write_failed_metrics(run_dir, run_id, type(exc).__name__, str(exc))
        console.print(f"[red]Compare failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    write_json(run_dir / "compare.json", result)
    if isinstance(result.get("quality_gate"), dict):
        write_json(run_dir / "acceptance.json", result["quality_gate"])
    write_json(run_dir / "metrics.json", {"status": "completed", "run_id": run_id, "compare": result})
    write_json(run_dir / "model.json", {"id": cfg.model.id, "revision": cfg.model.revision})

    # Rich table (behavioral, not quality)
    try:
        from rich.table import Table

        table = Table(title="Compare — behavioral delta (observational)", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Protocol match", str(result.get("protocol_match")))
        if result.get("dataset_match") is not None:
            table.add_row("Dataset match", str(result.get("dataset_match")))
        if result.get("quality_report_schema_match") is not None:
            table.add_row("Quality schema match", str(result.get("quality_report_schema_match")))
        if result.get("model_revision_match") is not None:
            table.add_row("Model revision match", str(result.get("model_revision_match")))
        if result.get("actual_dtype_match") is not None:
            table.add_row("Actual dtype match", str(result.get("actual_dtype_match")))
        if result.get("actual_device_match") is not None:
            table.add_row("Actual device match", str(result.get("actual_device_match")))
        table.add_row("Result observation", str(result.get("result_observation")))
        table.add_row("Result same?", str(result.get("result_match")))
        table.add_row("Content match", str(result.get("content_match")))
        quality_gate = result.get("quality_gate")
        if isinstance(quality_gate, dict):
            table.add_row("Quality accepted", str(quality_gate.get("accepted")))
            table.add_row("Quality composite", str(quality_gate.get("composite_score")))
        table.add_row(
            "Baseline result FP",
            str(result.get("baseline_result_fingerprint", "?"))[:24] + "..."
            if result.get("baseline_result_fingerprint")
            else "?",
        )
        table.add_row(
            "Quantized result FP",
            str(result.get("quantized_result_fingerprint", "?"))[:24] + "..."
            if result.get("quantized_result_fingerprint")
            else "?",
        )
        console.print(table)
        if result.get("token_diffs"):
            t2 = Table(title="Per-prompt token diff", show_header=True)
            t2.add_column("Prompt", style="cyan")
            t2.add_column("Same?", style="white")
            for d in result["token_diffs"]:
                t2.add_row(str(d.get("id")), str(d.get("same")))
            console.print(t2)
        console.print("[dim]Result fingerprint equality is observational, NOT pass/fail.[/dim]")
    except Exception:
        pass

    if require_acceptance and not multiple["quality_acceptance"]["accepted"]:
        from openternary.services.errors import AcceptanceError

        raise AcceptanceError("required quality acceptance failed or evidence is incomplete")
    typer.echo(f"Compare completed: {run_dir}")
    raise typer.Exit(0)
