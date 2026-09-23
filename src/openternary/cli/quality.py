from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.common import _resolve_cli_overrides, _write_failed_metrics, app, console
from openternary.cli.protocol import Command


@app.command("quality", cls=Command)
def quality(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    data: Annotated[pathlib.Path | None, typer.Option("--data", help="Frozen quality dataset JSON")] = None,
    split: Annotated[str, typer.Option("--split", help="Evaluation split: validation/test")] = "validation",
    max_length: Annotated[int, typer.Option("--max-length", min=2, help="Perplexity context length")] = 512,
    stride: Annotated[int, typer.Option("--stride", min=1, help="Perplexity sliding-window stride")] = 256,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate inputs without loading the model")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Evaluate frozen general/Japanese PPL and instruction cases."""
    from openternary.config.loader import dump_config_yaml, load_config
    from openternary.experiment.metadata import write_json
    from openternary.experiment.run import create_run
    from openternary.utils.logging import setup_logging
    from openternary.utils.seed import seed_everything

    overrides = _resolve_cli_overrides(seed, device, dtype, output)
    if model is not None:
        overrides["model.id"] = model
    try:
        cfg = load_config(config_path=config, cli_overrides=overrides)
    except FileNotFoundError as exc:
        console.print(f"[red]Config file not found:[/red] {exc}")
        raise typer.Exit(2) from exc
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(2) from exc

    if data is None:
        console.print("[red]quality requires --data with a frozen dataset JSON[/red]")
        raise typer.Exit(2)
    if not data.is_file():
        console.print(f"[red]Quality dataset not found:[/red] {data}")
        raise typer.Exit(2)
    if split not in {"validation", "test"}:
        console.print("[red]--split must be validation or test[/red]")
        raise typer.Exit(2)
    if stride >= max_length:
        console.print("[red]--stride must be smaller than --max-length[/red]")
        raise typer.Exit(2)

    if dry_run:
        try:
            from openternary.benchmark.quality_runner import validate_quality_dataset

            dataset_audit = validate_quality_dataset(data)
        except (FileNotFoundError, OSError, ValueError) as exc:
            console.print(f"[red]Quality dataset validation failed:[/red] {exc}")
            raise typer.Exit(2) from exc
        console.print("[bold cyan]quality: dry-run — resolved config[/bold cyan]")
        console.print(dump_config_yaml(cfg))
        console.print(f"[dim]Data: {data} | split={split} | max_length={max_length} | stride={stride}[/dim]")
        console.print(f"[dim]Dataset fingerprint: {dataset_audit['dataset_fingerprint']}[/dim]")
        console.print("[dim]No run directory was created (--dry-run).[/dim]")
        raise typer.Exit(0)

    run_dir, run_id = create_run(cfg)
    logger = setup_logging(run_dir, verbose=verbose)
    logger.info(f"quality — run_id={run_id} run_dir={run_dir} split={split} data={data}")
    seed_everything(cfg.seed)
    try:
        from openternary.services.evaluation import quality as run_quality_benchmark

        result = run_quality_benchmark(
            cfg,
            data,
            split=split,
            max_length=max_length,
            stride=stride,
        )
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        _write_failed_metrics(run_dir, run_id, type(exc).__name__, str(exc))
        console.print(f"[red]Quality evaluation failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("quality evaluation failed")
        _write_failed_metrics(run_dir, run_id, type(exc).__name__, str(exc))
        console.print(f"[red]Quality evaluation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    write_json(run_dir / "quality.json", result)
    write_json(
        run_dir / "metrics.json",
        {
            "status": "completed",
            "run_id": run_id,
            "quality_summary": result["summary"],
            "resources": result.get("resources", {}),
            "dataset_fingerprint": result["dataset_fingerprint"],
            "scientific_acceptance": False,
        },
    )
    logger.info(f"quality written to {run_dir / 'quality.json'}")
    typer.echo(f"Quality evaluation completed: {run_dir}")
    typer.echo(f"quality.json saved to {run_dir / 'quality.json'}")
    raise typer.Exit(0)
