"""Metadata and diagnostics commands with the same JSON contract as execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from openternary.cli.common import app, console
from openternary.cli.protocol import Command
from openternary.services.reporting import publish

backends = typer.Typer(help="Backend installation and capability metadata", no_args_is_help=True)
artifacts = typer.Typer(help="Artifact integrity and metadata", no_args_is_help=True)
app.add_typer(backends, name="backends")
app.add_typer(artifacts, name="artifacts")


def display(result: dict[str, Any]) -> None:
    publish(**result)
    console.print_json(json.dumps(result, ensure_ascii=False, allow_nan=False))


@backends.command("list", cls=Command)
def backend_list() -> None:
    from openternary.backends import list_backends

    display({"backends": list_backends()})


@backends.command("info", cls=Command)
def backend_info(name: str) -> None:
    from openternary.backends import get_spec

    display(get_spec(name).describe())


@app.command("plan", cls=Command)
def plan(
    model: Annotated[str | None, typer.Argument()] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    backend: Annotated[str | None, typer.Option()] = None,
    scheme: Annotated[str | None, typer.Option()] = None,
    weight_dtype: Annotated[str | None, typer.Option()] = None,
    component: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Validate a local plan without downloads, tensor loading, or writes."""
    from openternary.services.planning import plan as service_plan

    overrides = {
        "model.id": model,
        "quantization.backend": backend,
        "quantization.scheme": scheme,
        "quantization.weight_dtype": weight_dtype,
        "model.component": component,
    }
    display(service_plan(config, overrides))


@app.command("doctor", cls=Command)
def doctor(
    probe_runtime: Annotated[bool, typer.Option("--probe-runtime")] = False,
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Read installed package metadata; never install, repair, or execute a model."""
    from openternary.services.planning import doctor as service_doctor

    result = service_doctor()
    if probe_runtime:
        from openternary.services.errors import HardwareError
        from openternary.services.runtime_probe import probe_runtime as probe

        result["runtime_probe"] = probe(device)
        display(result)
        if result["runtime_probe"]["status"] != "passed":
            raise HardwareError("runtime probe failed; see individual checks")
    else:
        display(result)


@artifacts.command("validate", cls=Command)
def artifact_validate(path: Path, level: Annotated[str, typer.Option("--level")] = "files") -> None:
    from openternary.services.artifacts import validate_artifact

    if level == "tensors":
        from openternary.services.tensor_validation import validate_tensors

        display(validate_tensors(path))
        return
    if level != "files":
        raise ValueError("validation level must be files or tensors")
    display(
        {"artifact": str(path.resolve()), "manifest": validate_artifact(path), "artifact_validation": "file_integrity"}
    )


@app.command("plugins", cls=Command)
def plugins() -> None:
    """List installed version-one extensions without executing their code."""
    from openternary.services.plugins import discover_plugins

    display({"plugins": discover_plugins()})
