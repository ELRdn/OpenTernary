from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.common import _resolve_cli_overrides, _write_failed_metrics, app, console
from openternary.cli.protocol import Command


@app.command("benchmark", cls=Command)
def benchmark(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    suite: Annotated[str | None, typer.Option("--suite", help="Benchmark suite: smoke")] = None,
    thinking: Annotated[bool | None, typer.Option("--thinking/--no-thinking", help="Gemma 4 thinking mode")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Benchmark model — Phase 1b BF16 Smoke Inference Baseline."""
    from openternary.config.loader import dump_config_yaml, load_config
    from openternary.experiment.metadata import write_json
    from openternary.experiment.run import create_run
    from openternary.utils.logging import setup_logging
    from openternary.utils.seed import seed_everything

    overrides = _resolve_cli_overrides(seed, device, dtype, output)
    if model is not None:
        overrides["model.id"] = model
    if suite is not None:
        overrides["benchmark.suite"] = suite
    if thinking is not None:
        overrides["benchmark.thinking"] = thinking

    try:
        cfg = load_config(config_path=config, cli_overrides=overrides)
    except FileNotFoundError as e:
        console.print(f"[red]Config file not found:[/red] {e}")
        raise typer.Exit(2) from e
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(2) from e

    if dry_run:
        console.print("[bold cyan]benchmark: dry-run — resolved config[/bold cyan]")
        console.print(dump_config_yaml(cfg))
        console.print("[dim]No run directory was created (--dry-run).[/dim]")
        raise typer.Exit(0)

    run_dir, run_id = create_run(cfg)
    logger = setup_logging(run_dir, verbose=verbose)
    logger.info(f"benchmark — run_id={run_id} run_dir={run_dir} suite={cfg.benchmark.suite}")
    seed_info = seed_everything(cfg.seed)
    logger.info(f"seed: {seed_info}")

    try:
        from openternary.benchmark.metrics import to_metrics
        from openternary.services.evaluation import benchmark as run_benchmark
        from openternary.utils.hf_cache import resolve_snapshot

        snapshot_path = resolve_snapshot(cfg.model.id, cfg.model.revision)
        result = run_benchmark(cfg, snapshot_path=snapshot_path, suite=cfg.benchmark.suite)
    except FileNotFoundError as e:
        _write_failed_metrics(run_dir, run_id, "FileNotFoundError", str(e))
        console.print(f"[red]Snapshot not found:[/red] {e}")
        raise typer.Exit(2) from e
    except ImportError as e:
        _write_failed_metrics(run_dir, run_id, "ImportError", str(e))
        console.print(f"[red]Missing dependency:[/red] {e}")
        console.print("[dim]Install with: uv sync --extra ml[/dim]")
        raise typer.Exit(2) from e
    except ValueError as e:
        _write_failed_metrics(run_dir, run_id, "ValueError", str(e))
        console.print(f"[red]Benchmark configuration error:[/red] {e}")
        raise typer.Exit(2) from e
    except RuntimeError as e:
        # BF16 exact violation 等の loud error
        _write_failed_metrics(run_dir, run_id, "RuntimeError", str(e))
        console.print(f"[red]Benchmark failed:[/red] {e}")
        raise typer.Exit(2) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("benchmark failed")
        _write_failed_metrics(run_dir, run_id, type(e).__name__, str(e))
        console.print(f"[red]Benchmark failed:[/red] {e}")
        raise typer.Exit(1) from e

    # benchmark.json / metrics.json / model.json
    write_json(run_dir / "benchmark.json", result)
    write_json(run_dir / "model.json", {"id": cfg.model.id, "revision": cfg.model.revision, "adapter": "gemma4"})
    write_json(run_dir / "metrics.json", to_metrics(result, run_id))
    logger.info(f"benchmark written to {run_dir / 'benchmark.json'}")

    # Rich summary
    try:
        from rich.table import Table

        suite_info = result.get("suite", {})
        summary = result.get("summary", {})
        table = Table(title=f"Benchmark — {cfg.model.id} [{suite_info.get('name', '?')}]", show_header=True)
        table.add_column("Section", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Suite", f"{suite_info.get('name', '?')} v{suite_info.get('version', '?')}")
        table.add_row("Protocol FP", str(suite_info.get("fingerprint", "?"))[:24] + "...")
        table.add_row("Result FP", str(result.get("result_fingerprint", "?"))[:24] + "...")
        table.add_row("Thinking", str(result.get("thinking")))
        table.add_row(
            "Dtype", f"{result.get('dtype', {}).get('requested', '?')} → {result.get('dtype', {}).get('actual', '?')}"
        )
        table.add_row("Model load ms", str(result.get("model_load_ms", "?")))
        table.add_row("Warmup", str(result.get("warmup", {}).get("executed", "?")))
        table.add_row("Prompts", str(summary.get("num_prompts", "?")))
        table.add_row("Avg latency ms", str(summary.get("avg_generation_latency_ms", "?")))
        table.add_row("Tokens/s (output)", str(summary.get("output_tokens_per_second", "?")))
        table.add_row("Total output tokens", str(summary.get("total_output_tokens", "?")))
        console.print(table)
    except Exception:
        pass

    typer.echo(f"Benchmark completed: {run_dir}")
    typer.echo(f"benchmark.json saved to {run_dir / 'benchmark.json'}")
    raise typer.Exit(0)
