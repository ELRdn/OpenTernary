from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from openternary.cli.common import _resolve_cli_overrides, _write_failed_metrics, app, console
from openternary.cli.protocol import Command


@app.command("calibrate", cls=Command)
def calibrate(
    model: Annotated[str | None, typer.Argument(help="Model id or path")] = None,
    config: Annotated[pathlib.Path | None, typer.Option("--config", "-c", help="Path to YAML config")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Run output directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    device: Annotated[str | None, typer.Option("--device", help="Device: auto/cpu/cuda")] = None,
    dtype: Annotated[str | None, typer.Option("--dtype", help="Dtype: bf16/fp16/fp32")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show config and exit")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose")] = False,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from latest checkpoint")] = False,
    resume_from: Annotated[
        pathlib.Path | None, typer.Option("--resume-from", help="Resume from specific checkpoint")
    ] = None,
    materialize_only: Annotated[
        bool, typer.Option("--materialize-only", help="Skip training, materialize from latest/step checkpoint only")
    ] = False,
    init_from: Annotated[
        pathlib.Path | None,
        typer.Option("--init-from", help="Warm start from previous calibration run (Phase 4.1 etc)"),
    ] = None,
    finalize: Annotated[
        bool,
        typer.Option(
            "--finalize",
            help="Finalization-only recovery: regenerate calibration.json/metrics.json without re-running optimization",
        ),
    ] = False,
    preflight: Annotated[
        bool,
        typer.Option("--preflight", help="Validate source, canonical targets, environment, and resource budget only"),
    ] = False,
) -> None:
    """Calibrate — layer-local recon-scale-threshold (Phase 4.2)."""
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

    if not cfg.calibration.enabled:
        if "enabled" in cfg.calibration.model_fields_set:
            console.print("[red]calibrate conflicts with calibration.enabled=false[/red]")
            raise typer.Exit(2)
        cfg.calibration.enabled = True

    # Window validation (Phase 4.1 only per-layer)
    if cfg.calibration.window != "per-layer":
        console.print(f"[red]window={cfg.calibration.window} is not supported in Phase 4.1 (only per-layer)[/red]")
        raise typer.Exit(2)

    if preflight:
        if any((dry_run, resume, resume_from is not None, materialize_only, init_from is not None, finalize)):
            console.print("[red]--preflight cannot be combined with execution/recovery options[/red]")
            raise typer.Exit(2)
        if not cfg.output:
            console.print("[red]--preflight requires --output[/red]")
            raise typer.Exit(2)
        try:
            from openternary.calibration.preflight import build_preflight_report, write_preflight_report
            from openternary.utils.hf_cache import resolve_snapshot

            snapshot = resolve_snapshot(cfg.model.id, cfg.model.revision)
            output_path = pathlib.Path(cfg.output)
            report = build_preflight_report(cfg, snapshot, output_path)
            report_path = write_preflight_report(report, output_path)
        except (FileNotFoundError, FileExistsError, ValueError) as e:
            console.print(f"[red]Preflight failed:[/red] {e}")
            raise typer.Exit(2) from e
        if report["status"] != "pass":
            console.print(f"[red]Preflight gates failed:[/red] {report['checks']}")
            console.print(f"[dim]Report: {report_path}[/dim]")
            raise typer.Exit(2)
        console.print(
            f"[bold green]preflight PASS:[/bold green] targets={report['target_inventory']['count']} report={report_path}"
        )
        raise typer.Exit(0)

    # --- Finalization-only recovery (Phase 4.2) ---
    if finalize:
        if materialize_only:
            console.print("[red]--finalize cannot be used with --materialize-only[/red]")
            raise typer.Exit(2)
        if init_from is not None:
            console.print("[red]--finalize cannot be used with --init-from[/red]")
            raise typer.Exit(2)
        # finalize は既存 run 直下へ原子的に書き込むため --output が必須で存在している必要がある
        run_dir_final = pathlib.Path(cfg.output) if cfg.output else None
        if run_dir_final is None or not run_dir_final.exists():
            console.print(
                "[red]--finalize requires existing --output directory (e.g. runs/gemma4-e2b-g128-threshold-real-rocm)[/red]"
            )
            raise typer.Exit(2)
        # --resume / --resume-from は finalize では source checkpoint 指定として扱う（新規 -001 を作らない）
        ckpt_arg: pathlib.Path | None = None
        if resume_from is not None:
            ckpt_arg = pathlib.Path(resume_from)
            if not ckpt_arg.exists():
                console.print(f"[red]checkpoint not found for --finalize: {ckpt_arg}[/red]")
                raise typer.Exit(2)
        # --resume 単体は latest checkpoint を指すため None で OK
        try:
            from openternary.calibration.recovery import finalize_run
            from openternary.experiment.run import run_lock

            if dry_run:
                console.print("finalize: dry-run — checkpoint integrity not probed")
                raise typer.Exit(0)
            with run_lock(run_dir_final):
                result_final = finalize_run(run_dir_final, checkpoint_path=ckpt_arg, dry_run=False)
        except FileNotFoundError as e:
            console.print(f"[red]Finalize failed (file not found):[/red] {e}")
            raise typer.Exit(2) from e
        except ImportError as e:
            console.print(f"[red]Finalize failed (missing dependency):[/red] {e}")
            raise typer.Exit(2) from e
        except ValueError as e:
            console.print(f"[red]Finalize failed (integrity gate):[/red] {e}")
            raise typer.Exit(2) from e
        except Exception as e:  # noqa: BLE001
            # IntegrityError など
            try:
                from openternary.calibration.recovery import IntegrityError

                if isinstance(e, IntegrityError):
                    console.print(f"[red]Integrity gate FAILED — fail closed, no files written:[/red] {e}")
                    raise typer.Exit(2) from e
            except ImportError:
                pass
            console.print(f"[red]Finalize failed:[/red] {e}")
            raise typer.Exit(1) from e
        if dry_run:
            console.print("[bold cyan]finalize: dry-run — integrity gate PASS[/bold cyan]")
            console.print(
                f"[dim]Checkpoint: {result_final.get('report', {})}[/dim]"
                if isinstance(result_final, dict)
                else "[dim]gate passed[/dim]"
            )
            console.print(f"[dim]No files were written (--dry-run). Run dir: {run_dir_final}[/dim]")
        else:
            console.print(f"[bold green]finalize done:[/bold green] {run_dir_final}")
            console.print(
                f"[dim]checkpoint={result_final.get('checkpoint')} state_files={result_final.get('state_files')} tensor_entries={result_final.get('snapshot_tensor_entries')}[/dim]"
            )
            console.print(f"[dim]calibration.json: {result_final.get('calibration_json')}[/dim]")
            console.print(f"[dim]metrics.json: {result_final.get('metrics_json')} (status=completed)[/dim]")
        raise typer.Exit(0)

    if dry_run:
        console.print("[bold cyan]calibrate: dry-run — resolved config[/bold cyan]")
        console.print(dump_config_yaml(cfg))
        console.print("Target modules: unknown (inspection required)")
        console.print("Dataset probe: not performed; use explicit preflight for data checks")
        console.print("Teacher RAM / activation cache: unknown (model-dependent)")
        console.print("No run directory was created (--dry-run).")
        raise typer.Exit(0)

    # Handle --materialize-only (skip training, materialize from checkpoint)
    if materialize_only:
        if not cfg.output or not pathlib.Path(cfg.output).exists():
            console.print("[red]--materialize-only requires existing --output directory[/red]")
            raise typer.Exit(2)
        run_dir = pathlib.Path(cfg.output)  # type: ignore[no-redef]
        ckpt_dir = run_dir / "artifacts" / "checkpoint"
        has_ckpt = False
        if resume_from is not None:
            has_ckpt = pathlib.Path(resume_from).exists()
        else:
            if ckpt_dir.exists() and any(ckpt_dir.glob("step_*.pt")):
                has_ckpt = True
        if not has_ckpt:
            console.print(f"[red]no checkpoint found for --materialize-only in {run_dir}[/red]")
            raise typer.Exit(2)
        try:
            import json as _json

            env_path = run_dir / "environment.json"
            if env_path.exists():
                env_data = _json.loads(env_path.read_text(encoding="utf-8"))
                run_id = str(env_data.get("run_id", "materialize-only"))  # type: ignore[no-redef]
            else:
                run_id = "materialize-only"  # type: ignore[no-redef]
        except Exception:
            run_id = "materialize-only"  # type: ignore[no-redef]
        logger = setup_logging(run_dir, verbose=verbose)
        logger.info(f"calibrate --materialize-only — run_id={run_id} run_dir={run_dir} method={cfg.calibration.method}")
        seed_info = seed_everything(cfg.seed)
        logger.info(f"seed: {seed_info}")
        try:
            from openternary.calibration.runner import run_calibration
            from openternary.utils.hf_cache import resolve_snapshot

            teacher_snap = resolve_snapshot(cfg.model.id, cfg.model.revision)
            logger.info(f"teacher snapshot: {teacher_snap}")

            result = run_calibration(cfg, teacher_snap, run_dir, resume_from=resume_from, materialize_only=True)
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
            console.print(f"[red]Calibration error:[/red] {e}")
            raise typer.Exit(2) from e
        except Exception as e:  # noqa: BLE001
            logger.exception("calibrate materialize-only failed")
            _write_failed_metrics(run_dir, run_id, type(e).__name__, str(e))
            console.print(f"[red]Calibrate failed:[/red] {e}")
            raise typer.Exit(1) from e
        # success summary already handled below, but we duplicate minimal
        console.print(f"[bold green]materialize-only done:[/bold green] {result.get('calibrated_snapshot')}")
        raise typer.Exit(0)

    # Handle --init-from warm start
    if init_from is not None:
        if resume or resume_from is not None:
            console.print("[red]--init-from cannot be used with --resume/--resume-from[/red]")
            raise typer.Exit(2)
        if not pathlib.Path(init_from).exists():
            console.print(f"[red]--init-from path not found: {init_from}[/red]")
            raise typer.Exit(2)
        # store for runner and provenance
        cfg.calibration.init_from = str(init_from)  # type: ignore[attr-defined]

    # Handle --resume / --resume-from reuse (no new -001) — Phase 4.2 fix:
    # 通常の新規 run は create_run が -001 衝突回避を行うが、resume 系は既存 output を再利用する。
    # また checkpoint の source run directory から calibration_state を解決できるよう runner 側でフォールバックする。
    run_dir: pathlib.Path  # type: ignore[no-redef]
    run_id: str  # type: ignore[no-redef]
    _is_resume_requested = bool(resume or resume_from is not None)
    if _is_resume_requested and cfg.output and pathlib.Path(cfg.output).exists():
        run_dir = pathlib.Path(cfg.output)
        # --resume-from が別 run の checkpoint を指す場合でも、output 直下の checkpoint と同様に扱う。
        # 存在チェックは resume_from が指すファイルを優先し、無ければ run_dir/artifacts/checkpoint 内の最新を探索。
        ckpt_dir = run_dir / "artifacts" / "checkpoint"
        has_resumable = False
        if resume_from is not None:
            # resume_from が相対でも絶対でも存在すれば OK。無い場合は run_dir からの相対解決も試す
            p = pathlib.Path(resume_from)
            if p.exists():
                has_resumable = True
            else:
                alt = run_dir / p
                has_resumable = alt.exists()
                # さらに source run が checkpoint の親から推定できる場合、source の checkpoint も許容
                # ここでは有無のみで判定し、無い場合は runner がエラーにする
        else:
            if ckpt_dir.exists() and (any(ckpt_dir.glob("step_*.pt")) or any(ckpt_dir.glob("module_*.pt"))):
                has_resumable = True
        if not has_resumable:
            # resume_from が別 run を指す場合は run_dir 側に checkpoint が無くても許容する
            if resume_from is not None and pathlib.Path(resume_from).exists():
                has_resumable = True
            else:
                console.print(f"[red]no resumable checkpoint found in {run_dir}[/red]")
                raise typer.Exit(2)
        # read existing run_id if available
        try:
            import json as _json

            env_path = run_dir / "environment.json"
            if env_path.exists():
                env_data = _json.loads(env_path.read_text(encoding="utf-8"))
                run_id = str(env_data.get("run_id", "resume"))
            else:
                run_id = "resume"
        except Exception:
            run_id = "resume"
    elif _is_resume_requested:
        # resume requested but no existing output to resume from
        console.print(
            "[red]no resumable checkpoint found: --resume/--resume-from requires existing --output directory[/red]"
        )
        raise typer.Exit(2)
    else:
        run_dir, run_id = create_run(cfg)
    logger = setup_logging(run_dir, verbose=verbose)
    logger.info(f"calibrate — run_id={run_id} run_dir={run_dir} method={cfg.calibration.method}")
    seed_info = seed_everything(cfg.seed)
    logger.info(f"seed: {seed_info}")

    try:
        from openternary.calibration.runner import run_calibration
        from openternary.utils.hf_cache import resolve_snapshot

        teacher_snap = resolve_snapshot(cfg.model.id, cfg.model.revision)
        logger.info(f"teacher snapshot: {teacher_snap}")

        result = run_calibration(
            cfg, teacher_snap, run_dir, resume=resume, resume_from=resume_from, init_from=init_from
        )
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
        console.print(f"[red]Calibration error:[/red] {e}")
        raise typer.Exit(2) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("calibrate failed")
        _write_failed_metrics(run_dir, run_id, type(e).__name__, str(e))
        console.print(f"[red]Calibrate failed:[/red] {e}")
        raise typer.Exit(1) from e

    # Write model/metrics already done in runner, but ensure run-level
    write_json(run_dir / "model.json", {"id": cfg.model.id, "revision": cfg.model.revision, "adapter": "gemma4"})
    # Rich summary
    try:
        import json as _json

        from rich.table import Table

        calib_path = (
            pathlib.Path(result["calibration_json"]) if "calibration_json" in result else run_dir / "calibration.json"
        )
        data = {}
        if calib_path.exists():
            data = _json.loads(calib_path.read_text(encoding="utf-8"))
        table = Table(title=f"Calibrate — {cfg.model.id} [{cfg.calibration.method}]", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Dataset", str(data.get("dataset", cfg.calibration.dataset)))
        table.add_row("Steps", str(data.get("steps", cfg.calibration.steps)))
        table.add_row(
            "Initial loss",
            f"{data.get('initial_loss', '?'):.6f}"
            if isinstance(data.get("initial_loss"), (int, float))
            else str(data.get("initial_loss", "?")),
        )
        table.add_row(
            "Final loss",
            f"{data.get('final_loss', '?'):.6f}"
            if isinstance(data.get("final_loss"), (int, float))
            else str(data.get("final_loss", "?")),
        )
        table.add_row(
            "Held-out before",
            f"{data.get('heldout_loss_before', '?'):.6f}"
            if isinstance(data.get("heldout_loss_before"), (int, float))
            else str(data.get("heldout_loss_before", "?")),
        )
        table.add_row(
            "Held-out after",
            f"{data.get('heldout_loss_after', '?'):.6f}"
            if isinstance(data.get("heldout_loss_after"), (int, float))
            else str(data.get("heldout_loss_after", "?")),
        )
        table.add_row(
            "Scale delta",
            f"{data.get('mean_abs_scale_delta', '?'):.6f}"
            if isinstance(data.get("mean_abs_scale_delta"), (int, float))
            else str(data.get("mean_abs_scale_delta", "?")),
        )
        table.add_row("Contamination", str(data.get("contamination_train_smoke", "?")))
        console.print(table)
    except Exception:
        pass

    typer.echo(f"Calibrate completed: {run_dir}")
    raise typer.Exit(0)
