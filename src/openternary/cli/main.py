"""OpenTernary CLI entry point — Phase 0 foundation."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer
from rich.console import Console

from openternary import __version__

app = typer.Typer(
    name="openternary",
    help="OpenTernary — ternary LLM research toolkit (Phase 0: CLI foundation)",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        from openternary.utils.git import get_git_commit

        commit = get_git_commit()
        console.print(f"openternary {__version__} (commit: {commit})")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", help="Show version and exit", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """OpenTernary CLI."""
    pass


def _resolve_cli_overrides(
    seed: int | None,
    device: str | None,
    dtype: str | None,
    output: str | None,
    group_size: int | None = None,
    scale_granularity: str | None = None,
    grouping_scheme: str | None = None,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if seed is not None:
        overrides["seed"] = seed
    if device is not None:
        overrides["device"] = device
    if dtype is not None:
        overrides["dtype"] = dtype
    if output is not None:
        overrides["output"] = output
    if group_size is not None:
        overrides["quantization.group_size"] = group_size
    if scale_granularity is not None:
        overrides["quantization.scale_granularity"] = scale_granularity
    if grouping_scheme is not None:
        overrides["quantization.grouping_scheme"] = grouping_scheme
    return overrides


def _write_failed_metrics(run_dir: pathlib.Path, run_id: str, error_type: str, error_message: str) -> None:
    """失敗runの metrics.json を QR-02 Failure Visibility に従い更新."""
    try:
        from openternary.experiment.metadata import write_json

        write_json(
            run_dir / "metrics.json",
            {
                "status": "failed",
                "run_id": run_id,
                "error_type": error_type,
                "error_message": error_message,
            },
        )
    except Exception:
        pass


def _handle_command(
    name: str,
    config: pathlib.Path | None,
    overrides: dict[str, object],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Phase 0 共通処理: config解決 → dry-run 分岐 → 未実装なら exit 1."""
    from openternary.config.loader import dump_config_yaml, load_config
    from openternary.experiment.run import create_run
    from openternary.utils.logging import setup_logging
    from openternary.utils.seed import seed_everything

    try:
        cfg = load_config(config_path=config, cli_overrides=overrides)
    except FileNotFoundError as e:
        console.print(f"[red]Config file not found:[/red] {e}")
        raise typer.Exit(2) from e
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(2) from e

    if dry_run:
        console.print(f"[bold cyan]{name}: dry-run — resolved config[/bold cyan]")
        console.print(dump_config_yaml(cfg))
        console.print("[dim]No run directory was created (--dry-run).[/dim]")
        raise typer.Exit(0)

    run_dir, run_id = create_run(cfg)
    logger = setup_logging(run_dir, verbose=verbose)
    logger.info(f"{name} — run_id={run_id} run_dir={run_dir}")
    seed_info = seed_everything(cfg.seed)
    logger.info(f"seed: {seed_info}")

    # Use typer.echo to ensure CliRunner captures output (Rich Console at import time bypasses capture)
    typer.echo(f"{name} is not implemented in Phase 0 (planned for Phase 1/2).")
    typer.echo(f"Run directory created: {run_dir} (reproducibility bundle)")
    typer.echo("Use --dry-run to preview config without creating a run.")
    typer.echo(f"environment.json / config.yaml saved to {run_dir}")
    raise typer.Exit(1)


# --- Commands (Phase 0 stubs) ---


@app.command("inspect")
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
    write_json(run_dir / "model.json", {"id": cfg.model.id, "revision": cfg.model.revision, "adapter": "gemma4"})
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
        table.add_row("Adapter", "gemma4")
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


@app.command("quantize")
def quantize(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Dtype: bf16/fp16/fp32 (quantize preserves original dtype; benchmarkで制御)"),
    ] = None,
    group_size: Annotated[int | None, typer.Option("--group-size", help="Group size for ternary quantization")] = None,
    scale_granularity: Annotated[
        str | None, typer.Option("--scale-granularity", help="Scale granularity: per_tensor or per_group")
    ] = None,
    grouping_scheme: Annotated[
        str | None, typer.Option("--grouping-scheme", help="Grouping scheme: last-dim-rowwise-v1")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Quantize model to ternary fake-quant sharded snapshot (Phase 3)."""
    from openternary.config.loader import dump_config_yaml, load_config
    from openternary.experiment.metadata import write_json
    from openternary.experiment.run import create_run
    from openternary.utils.logging import setup_logging
    from openternary.utils.seed import seed_everything

    overrides = _resolve_cli_overrides(seed, device, dtype, output, group_size, scale_granularity, grouping_scheme)
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

    if dtype is not None:
        console.print(
            "[yellow]Note: --dtype is ignored for quantize (original dtype preserved; use benchmark --dtype).[/yellow]"
        )

    if dry_run:
        console.print("[bold cyan]quantize: dry-run — resolved config[/bold cyan]")
        console.print(dump_config_yaml(cfg))
        # also preview planned conversion
        q = cfg.quantization
        console.print(
            f"[dim]Planned: scale_granularity={q.scale_granularity} group_size={q.group_size} grouping_scheme={q.grouping_scheme} method={q.method}[/dim]"
        )
        console.print("[dim]No run directory was created (--dry-run).[/dim]")
        raise typer.Exit(0)

    run_dir, run_id = create_run(cfg)
    logger = setup_logging(run_dir, verbose=verbose)
    logger.info(f"quantize — run_id={run_id} run_dir={run_dir}")
    seed_info = seed_everything(cfg.seed)
    logger.info(f"seed: {seed_info}")

    try:
        from openternary.quant.fake_quant import convert_snapshot
        from openternary.utils.hf_cache import resolve_snapshot

        snapshot_path = resolve_snapshot(cfg.model.id, cfg.model.revision)
        logger.info(f"source snapshot: {snapshot_path}")
        # dst is artifacts/snapshot inside run_dir
        dst_snapshot = run_dir / "artifacts" / "snapshot"
        report = convert_snapshot(snapshot_path, dst_snapshot, cfg)
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
        console.print(f"[red]Quantize error:[/red] {e}")
        raise typer.Exit(2) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("quantize failed")
        _write_failed_metrics(run_dir, run_id, type(e).__name__, str(e))
        console.print(f"[red]Quantize failed:[/red] {e}")
        raise typer.Exit(1) from e

    # write quantization report already in dst_snapshot/quantization.json — also copy to run root for convenience
    try:
        import json as _json

        qpath = dst_snapshot / "quantization.json"
        if qpath.exists():
            data = _json.loads(qpath.read_text(encoding="utf-8"))
            write_json(run_dir / "quantization.json", data)
        else:
            write_json(run_dir / "quantization.json", report.quantization_json)
    except Exception:
        write_json(run_dir / "quantization.json", report.quantization_json)

    write_json(run_dir / "model.json", {"id": cfg.model.id, "revision": cfg.model.revision, "adapter": "gemma4"})
    write_json(
        run_dir / "metrics.json",
        {
            "status": "completed",
            "run_id": run_id,
            "quantization": report.quantization_json,
            "content_fingerprint": report.content_fingerprint,
            "file_hashes": report.file_hashes,
            "fake_quant_artifact": report.quantization_json.get("fake_quant_artifact"),
            "packed_estimate": report.quantization_json.get("packed_estimate"),
        },
    )
    logger.info(f"quantize completed: {dst_snapshot} shards={report.shard_count}")

    # Rich summary
    try:
        from rich.table import Table

        qj = report.quantization_json
        fq = qj.get("fake_quant_artifact", {})
        pe = qj.get("packed_estimate", {})
        table = Table(title=f"Quantize — {cfg.model.id} [{qj.get('scale_granularity')}]", show_header=True)
        table.add_column("Section", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Scale granularity", str(qj.get("scale_granularity")))
        table.add_row("Group size / scheme", f"{qj.get('group_size')} / {qj.get('grouping_scheme')}")
        table.add_row("Quantizable tensors", str(qj.get("num_quantizable_tensors")))
        table.add_row("Total groups", str(qj.get("total_groups")))
        table.add_row("Shards", str(fq.get("shard_count")))
        table.add_row("Tensor payload", str(fq.get("tensor_payload_bytes")))
        table.add_row("Shard file bytes", str(fq.get("shard_file_bytes")))
        table.add_row("Packed estimate bytes", str(pe.get("estimated_total_bytes", "?")))
        table.add_row("Content FP", str(report.content_fingerprint)[:24] + "...")
        console.print(table)
    except Exception:
        pass

    typer.echo(f"Quantize completed: {run_dir}")
    typer.echo(f"fake-quant snapshot: {dst_snapshot}")
    raise typer.Exit(0)


@app.command("benchmark")
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
        from openternary.benchmark.runner import run_benchmark
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


@app.command("compare")
def compare(
    baseline: Annotated[str | None, typer.Argument(help="Baseline run directory")] = None,
    quantized: Annotated[str | None, typer.Argument(help="Quantized run directory")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Compare benchmark runs — behavioral Δ (observational)."""
    from openternary.config.loader import dump_config_yaml, load_config
    from openternary.experiment.compare import compare_runs
    from openternary.experiment.metadata import write_json
    from openternary.experiment.run import create_run
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

    result = compare_runs(_pl.Path(baseline), _pl.Path(quantized))
    write_json(run_dir / "compare.json", result)
    write_json(run_dir / "metrics.json", {"status": "completed", "run_id": run_id, "compare": result})
    write_json(run_dir / "model.json", {"id": cfg.model.id, "revision": cfg.model.revision})

    # Rich table (behavioral, not quality)
    try:
        from rich.table import Table

        table = Table(title="Compare — behavioral Δ (observational)", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Protocol match", str(result.get("protocol_match")))
        table.add_row("Result observation", str(result.get("result_observation")))
        table.add_row("Result same?", str(result.get("result_match")))
        table.add_row("Content match", str(result.get("content_match")))
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

    typer.echo(f"Compare completed: {run_dir}")
    raise typer.Exit(0)


@app.command("calibrate")
def calibrate(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Calibrate / reconstruct quantized model (Phase 0 stub)."""
    overrides = _resolve_cli_overrides(seed, device, dtype, output)
    if model is not None:
        overrides["model.id"] = model
    _handle_command("calibrate", config, overrides, dry_run, verbose)


@app.command("export")
def export(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    format: Annotated[str | None, typer.Option("--format", help="Export format: fake-quant/gguf")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
) -> None:
    """Export model (Phase 0 stub)."""
    overrides = _resolve_cli_overrides(seed, device, dtype, output)
    if model is not None:
        overrides["model.id"] = model
    if format is not None:
        overrides["export.format"] = format  # type: ignore[assignment]
    _handle_command("export", config, overrides, dry_run, verbose)


@app.command("cache-info")
def cache_info() -> None:
    """Show cache configuration."""
    import os

    console.print("[bold]Cache info[/bold]")
    console.print(f"UV_CACHE_DIR: {os.environ.get('UV_CACHE_DIR', '(default: %LOCALAPPDATA%/uv/cache)')}")
    console.print(f"HF_HOME: {os.environ.get('HF_HOME', '(default: ~/.cache/huggingface)')}")
    raise typer.Exit(0)


__all__ = ["app", "_handle_command", "_resolve_cli_overrides"]
