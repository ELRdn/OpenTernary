from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.common import _resolve_cli_overrides, _write_failed_metrics, app, console
from openternary.cli.protocol import Command


@app.command("inspect", cls=Command)
def inspect(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    load_weights: Annotated[
        bool,
        typer.Option(
            "--load-weights", help="Read actual weight payloads for distribution stats; header-only by default"
        ),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Inspect model architecture and quantizable modules (header-only by default)."""
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
    except FileNotFoundError as e:
        console.print(f"[red]Config file not found:[/red] {e}")
        raise typer.Exit(2) from e
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(2) from e

    if dry_run:
        console.print("[bold cyan]inspect: dry-run — resolved config[/bold cyan]")
        console.print(dump_config_yaml(cfg))
        console.print("[dim]No run directory was created (--dry-run).[/dim]")
        raise typer.Exit(0)

    run_dir, run_id = create_run(cfg)
    logger = setup_logging(run_dir, verbose=verbose)
    logger.info(f"inspect — run_id={run_id} run_dir={run_dir}")
    seed_info = seed_everything(cfg.seed)
    logger.info(f"seed: {seed_info}")

    # Run inspection
    try:
        from openternary.inspect.engine import run_inspection
        from openternary.utils.hf_cache import resolve_snapshot

        # Resolve snapshot for logging (actual inspection also resolves, but we capture path)
        snapshot_path = resolve_snapshot(cfg.model.id, cfg.model.revision)
        result = run_inspection(cfg, snapshot_path=snapshot_path, load_weights=load_weights)
    except FileNotFoundError as e:
        _write_failed_metrics(run_dir, run_id, "FileNotFoundError", str(e))
        console.print(f"[red]Snapshot not found:[/red] {e}")
        raise typer.Exit(2) from e
    except ImportError as e:
        _write_failed_metrics(run_dir, run_id, "ImportError", str(e))
        console.print(f"[red]Missing dependency:[/red] {e}")
        raise typer.Exit(2) from e
    except ValueError as e:
        _write_failed_metrics(run_dir, run_id, "ValueError", str(e))
        console.print(f"[red]Inspection error:[/red] {e}")
        raise typer.Exit(2) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("inspect failed")
        _write_failed_metrics(run_dir, run_id, type(e).__name__, str(e))
        console.print(f"[red]Inspect failed:[/red] {e}")
        raise typer.Exit(1) from e

    # Write inspection.json
    write_json(run_dir / "inspection.json", result)
    # Update model.json and metrics.json
    write_json(
        run_dir / "model.json",
        {
            "id": cfg.model.id,
            "revision": cfg.model.revision,
            "adapter": result.get("model", {}).get("adapter", "unknown"),
        },
    )
    write_json(
        run_dir / "metrics.json",
        {
            "status": "completed",
            "run_id": run_id,
            "inspection_fingerprint": result.get("inspection_fingerprint"),
            "summary": result.get("summary"),
        },
    )
    logger.info(f"inspection written to {run_dir / 'inspection.json'}")

    # Rich summary table
    try:
        from rich.table import Table

        summary = result.get("summary", {})
        arch = result.get("architecture", {})
        mem = result.get("memory_estimate", {})
        text = arch.get("text", {}) if isinstance(arch, dict) else {}

        table = Table(title=f"Inspection — {cfg.model.id}", show_header=True)
        table.add_column("Section", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Revision", str(cfg.model.revision or "unknown"))
        table.add_row("Adapter", str(result.get("model", {}).get("adapter", "unknown")))
        table.add_row("Architecture", str(arch.get("architecture", "unknown")))
        table.add_row("Text layers", str(text.get("num_hidden_layers", "?")))
        table.add_row("Hidden / Intermediate", f"{text.get('hidden_size', '?')} / {text.get('intermediate_size', '?')}")
        table.add_row("Vocab", str(text.get("vocab_size", "?")))
        table.add_row("Total tensors", str(summary.get("total_tensors", "?")))
        table.add_row(
            "Quantizable", f"{summary.get('quantizable_tensors', '?')} ({summary.get('quantizable_ratio', '?')})"
        )
        table.add_row(
            "Total params",
            f"{summary.get('total_params', '?'):,}"
            if isinstance(summary.get("total_params"), int)
            else str(summary.get("total_params", "?")),
        )
        table.add_row(
            "Quantizable params",
            f"{summary.get('quantizable_params', '?'):,}"
            if isinstance(summary.get("quantizable_params"), int)
            else str(summary.get("quantizable_params", "?")),
        )
        table.add_row("BF16 GB", str(mem.get("bf16_GB", "?")))
        table.add_row("FP32 GB", str(mem.get("fp32_GB", "?")))
        table.add_row("Fingerprint", str(result.get("inspection_fingerprint", "?"))[:24] + "...")
        if result.get("warnings"):
            table.add_row("Warnings", "; ".join(result["warnings"][:3]))
        console.print(table)
    except Exception:
        pass

    typer.echo(f"Inspection completed: {run_dir}")
    typer.echo(f"inspection.json saved to {run_dir / 'inspection.json'}")
    raise typer.Exit(0)
