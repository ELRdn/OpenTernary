from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.common import _resolve_cli_overrides, _write_failed_metrics, app, console
from openternary.cli.protocol import Command


@app.command("quantize", cls=Command)
def quantize(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Dtype: bf16/fp16/fp32 (quantize preserves source dtype; used by benchmark)"),
    ] = None,
    group_size: Annotated[int | None, typer.Option("--group-size", help="Group size for ternary quantization")] = None,
    scale_granularity: Annotated[
        str | None, typer.Option("--scale-granularity", help="Scale granularity: per_tensor or per_group")
    ] = None,
    grouping_scheme: Annotated[
        str | None, typer.Option("--grouping-scheme", help="Grouping scheme: last-dim-rowwise-v1")
    ] = None,
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    scheme: Annotated[str | None, typer.Option("--scheme")] = None,
    weight_dtype: Annotated[str | None, typer.Option("--weight-dtype")] = None,
    component: Annotated[str | None, typer.Option("--component")] = None,
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
    overrides.update(
        {
            "quantization.backend": backend,
            "quantization.scheme": scheme,
            "quantization.weight_dtype": weight_dtype,
            "model.component": component,
        }
    )
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
        raise ValueError("quantize preserves source dtype; use --weight-dtype for the quantized representation")
    from openternary.backends import validate_backend
    from openternary.services.planning import build_plan
    from openternary.services.reporting import publish

    validate_backend(cfg)
    planned = build_plan(cfg)

    if dry_run:
        publish(**planned)
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
        from openternary.services.execution import convert_artifact
        from openternary.utils.hf_cache import resolve_snapshot

        snapshot_path = resolve_snapshot(cfg.model.id, cfg.model.revision)
        logger.info(f"source snapshot: {snapshot_path}")
        # dst is artifacts/snapshot inside run_dir
        dst_snapshot = run_dir / "artifacts" / "snapshot"
        report = convert_artifact(snapshot_path, dst_snapshot, cfg)
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

    write_json(
        run_dir / "model.json", {"id": cfg.model.id, "revision": cfg.model.revision, "adapter": planned["adapter"]}
    )
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
    publish(artifact=str(dst_snapshot), artifact_validation="file_integrity", model_reload="not_run")

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
    typer.echo(f"{cfg.quantization.backend} artifact: {dst_snapshot}")
    raise typer.Exit(0)
