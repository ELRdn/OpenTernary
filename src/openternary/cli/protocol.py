"""One JSON result on stdout; human output on stderr; progress in an explicit stream."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
from typer.core import TyperCommand, TyperOption

from openternary.services import errors
from openternary.services.reporting import Operation, current


def error_code(exc: BaseException) -> int:
    for kind, code in (
        (errors.CapabilityError, 3),
        (ImportError, 4),
        (errors.HardwareError, 5),
        (errors.AcceptanceError, 6),
        (errors.BudgetError, 7),
        (KeyboardInterrupt, 130),
        (typer.Abort, 130),
        (typer.BadParameter.__mro__[1], 2),
        (ValueError, 2),
        (FileNotFoundError, 2),
        (FileExistsError, 2),
    ):
        if isinstance(exc, kind):
            return code
    if isinstance(exc, typer.Exit):
        return int(exc.exit_code)
    return 1


def envelope(operation: Operation, code: int, error: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": operation.command,
        "exit_code": code,
        "execution": "planned"
        if operation.result.get("status") == "planned" and code == 0
        else "completed"
        if code == 0 or code == 6
        else "interrupted"
        if code == 130
        else "failed",
        "artifact_validation": operation.result.get("artifact_validation", "not_run"),
        "quality_acceptance": operation.result.get("quality_acceptance", "not_run"),
        "result": operation.result,
        "error": error,
    }


def record_failure(operation: Operation, code: int, error: str | None) -> None:
    """Preserve a failed attempt even if initial run metadata could not be written."""
    if not code or code == 6 or not operation.result.get("run_dir"):
        return
    from openternary.experiment.metadata import write_json

    metrics = Path(operation.result["run_dir"]) / "metrics.json"
    if not metrics.parent.is_dir():
        return
    previous = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
    failed = {
        "status": "interrupted" if code == 130 else "failed",
        "run_id": operation.result.get("run_id"),
        "error_message": error,
        "exit_code": code,
    }
    if previous.get("status") in {None, "not_run"}:
        write_json(metrics, failed)
        operation.result["metrics"] = failed
    else:
        write_json(metrics.parent / "last_failure.json", failed)


class Command(TyperCommand):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params.extend(
            [
                TyperOption(param_decls=["--json", "json_output"], is_flag=True, help="Emit one versioned JSON result"),
                TyperOption(param_decls=["--quiet"], is_flag=True, help="Suppress progress, preserving result/error"),
                TyperOption(param_decls=["--events-jsonl"], help="Write progress events to a new file"),
            ]
        )

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except Exception as exc:
            if "--json" in args and hasattr(exc, "format_message"):
                typer.echo(
                    json.dumps(envelope(Operation(self.name or "unknown"), 2, exc.format_message()), ensure_ascii=False)
                )
                raise typer.Exit(2) from exc
            raise

    def invoke(self, ctx: Any) -> Any:
        json_output = ctx.params.pop("json_output", False)
        quiet = ctx.params.pop("quiet", False)
        events = ctx.params.pop("events_jsonl", None)
        operation = Operation(self.name or "unknown")
        token = current.set(operation)
        code, error = 0, None
        try:
            with contextlib.ExitStack() as stack:
                if events is not None:
                    if ctx.params.get("dry_run"):
                        raise ValueError("--events-jsonl cannot be used with --dry-run (no writes)")
                    operation.stream = stack.enter_context(Path(events).open("x", encoding="utf-8"))
                if quiet:
                    sink = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
                    stack.enter_context(contextlib.redirect_stdout(sink))
                    stack.enter_context(contextlib.redirect_stderr(sink))
                elif json_output:
                    stack.enter_context(contextlib.redirect_stdout(sys.stderr))
                operation.event("started")
                if (
                    self.name == "calibrate"
                    and not ctx.params.get("dry_run")
                    and any(ctx.params.get(k) for k in ("resume", "resume_from", "materialize_only", "finalize"))
                ):
                    from openternary.config.loader import load_config
                    from openternary.experiment.run import run_lock

                    config = load_config(ctx.params.get("config"), {"output": ctx.params.get("output")})
                    if config.output:
                        stack.enter_context(run_lock(Path(config.output)))
                if ctx.params.get("dry_run"):
                    from openternary.config.loader import load_config

                    overrides = {
                        k: v
                        for k, v in ctx.params.items()
                        if k
                        in {
                            "seed",
                            "device",
                            "dtype",
                            "output",
                            "group_size",
                            "scale_granularity",
                            "grouping_scheme",
                            "suite",
                            "thinking",
                        }
                        and v is not None
                    }
                    if ctx.params.get("model") is not None:
                        overrides["model.id"] = ctx.params["model"]
                    operation.result.update(
                        status="planned", config=load_config(ctx.params.get("config"), overrides).model_dump()
                    )
                try:
                    super().invoke(ctx)
                except typer.Exit as exc:
                    code = error_code(exc.__cause__) if exc.exit_code and exc.__cause__ else exc.exit_code
                    if code:
                        error = (
                            str(exc.__cause__) if exc.__cause__ else f"{self.name} rejected the request (see stderr)"
                        )
                except (Exception, KeyboardInterrupt) as exc:
                    code, error = error_code(exc), str(exc) or type(exc).__name__
                if operation.result.get("run_dir"):
                    metrics = Path(operation.result["run_dir"]) / "metrics.json"
                    if metrics.is_file():
                        operation.result["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
                record_failure(operation, code, error)
                operation.event("completed" if code == 0 else "interrupted" if code == 130 else "failed")
        except (Exception, KeyboardInterrupt) as exc:
            code, error = error_code(exc), str(exc) or type(exc).__name__
            try:
                record_failure(operation, code, error)
            except Exception as recording_error:
                error += f"; failure record could not be saved: {recording_error}"
        finally:
            current.reset(token)
        if json_output:
            typer.echo(json.dumps(envelope(operation, code, error), ensure_ascii=False, allow_nan=False))
        elif error:
            typer.echo(error, err=True)
        elif quiet:
            typer.echo(json.dumps(envelope(operation, code, None), ensure_ascii=False, allow_nan=False))
        if code:
            raise typer.Exit(code)
        return None
