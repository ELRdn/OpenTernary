"""`openternary setup`: detect the GPU, pick a PyTorch build, install it here, verify it on the device."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from openternary.cli.common import app, console, explain
from openternary.cli.protocol import Command
from openternary.ui.i18n import t

MAX_LISTED = 12


@app.command("setup", cls=Command, help=t("help.setup"), rich_help_panel=t("panel.environment"))
def setup(
    backend: Annotated[str, typer.Option("--backend", help="auto | cuda | rocm | mps | cpu")] = "auto",
    torch_version: Annotated[str | None, typer.Option("--torch", help="e.g. 2.13.0 or 2.13.0+rocm10.0.0")] = None,
    variant: Annotated[str | None, typer.Option("--variant", help="e.g. cu130, rocm10.0.0, cpu")] = None,
    amd_device: Annotated[
        str, typer.Option("--amd-device", help="AMD kernel set: auto (this GPU if published), all, or a gfx target")
    ] = "auto",
    list_only: Annotated[
        bool, typer.Option("--list", help="Show detected hardware and every compatible build")
    ] = False,
    restore: Annotated[
        bool, typer.Option("--restore", help="Reinstall exactly the build recorded by the last setup")
    ] = False,
    profile_file: Annotated[
        str | None, typer.Option("--profile", help="With --restore: read the record from this file")
    ] = None,
    pre: Annotated[bool, typer.Option("--pre", help="Include pre-release builds")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Install without asking")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the plan; install nothing")] = False,
) -> None:
    """Detect the GPU and install a matching PyTorch build into this environment."""
    from openternary.services import hardware
    from openternary.services import torch_setup as ts
    from openternary.services.errors import CapabilityError
    from openternary.services.reporting import milestone, publish
    from openternary.ui.render import setup_view

    if backend not in ts.BACKENDS:
        raise ValueError(f"--backend must be one of {', '.join(ts.BACKENDS)}")
    milestone("setup.detect")
    host = hardware.detect()
    want = ts.wanted_backend(host, backend)
    requirements, minimum = ts.openternary_requirements()
    if restore:
        recorded = ts.read_profile(Path(profile_file) if profile_file else None)
        if recorded is None:
            explain(t("setup.no_profile"))
            raise FileNotFoundError(str(ts.profile_path()))
        restored = ts.plan_from_profile(recorded, requirements, ts.find_uv())
        publish(
            host=host.to_dict(), backend=restored.backend, plan=restored.to_dict(), installed_torch=ts.installed_torch()
        )
        setup_view(console, {}, restored, limit=0)
        if not dry_run:
            _install_and_verify(restored, host, yes)
        return

    providers = ["pypi", "pytorch"] + (["amd"] if host.os in {"windows", "linux"} else [])
    if not list_only:
        providers = [p for p in providers if p in _providers_for(want, host.os)]
    listings: dict[str, list[str]] = {}
    vision: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for provider in providers:
        milestone("setup.index", provider=provider)
        try:
            listings[provider] = ts.list_wheels(provider, "torch")
            vision[provider] = ts.list_wheels(provider, "torchvision")
        except OSError as exc:  # network / HTTP failures (urllib raises OSError subclasses)
            errors[provider] = str(exc)
    offer = ts.build_offer(host, listings, pre=pre, minimum=minimum)
    offer.candidates = ts.with_torchvision(offer.candidates, vision)
    offer.errors = errors
    profile = ts.read_profile()
    pool, warnings = ts.rank(host, offer.candidates, want, ts.verified_here(profile))
    picked = ts.select(pool, torch_version, variant)
    result: dict[str, Any] = {
        "host": host.to_dict(),
        "backend": want,
        "warnings": warnings,
        "index_errors": errors,
        "other_pythons": offer.other_pythons,
        "candidates": [c.__dict__ for c in pool],
        "verified_here": sorted(ts.verified_here(profile)),
        "installed_torch": ts.installed_torch(),
    }
    if list_only:
        result["candidates_all"] = [c.__dict__ for c in offer.candidates]
        publish(**result)
        setup_view(console, result, None, limit=MAX_LISTED)
        return
    if not picked:
        publish(**result)
        setup_view(console, result, None, limit=MAX_LISTED)
        if errors and not listings:
            explain(t("setup.index_unreachable"))
            raise ConnectionError("; ".join(f"{k}: {v}" for k, v in errors.items()))
        needed = offer.other_pythons.get(want)
        if needed:
            explain(t("setup.python_mismatch", python=host.python, needed=", ".join(needed)))
        else:
            explain(t("setup.no_build", backend=want))
        raise CapabilityError(f"no compatible {want} PyTorch build for Python {host.python} on {host.os}")

    chosen = picked[0]
    current = ts.installed_torch()
    already = (
        profile is not None
        and (profile.get("verification") or {}).get("status") == "passed"
        and profile.get("version") == current == chosen.version
        and profile.get("backend") == want
    )
    if already:
        result["already_set_up"] = True
        publish(**result)
        setup_view(console, result, None, limit=MAX_LISTED)
        console.print(t("setup.already", version=chosen.version))
        return
    if chosen.torchvision is None:
        warnings.append("torchvision_missing")
    if chosen.provider == "amd":
        gpu = next((g for g in host.gpus if g.vendor == "amd" and g.arch and not g.integrated), None)
        amd_device = ts.amd_device_extra(
            amd_device, gpu.arch if gpu else None, lambda package: ts.list_wheels("amd", package)
        )
    plan = ts.make_plan(chosen, requirements, ts.find_uv(), amd_device=amd_device, backend=want)
    result["plan"] = plan.to_dict()
    publish(**result)
    setup_view(console, result, plan, limit=MAX_LISTED)
    if dry_run:
        return
    _install_and_verify(plan, host, yes)


def _install_and_verify(plan: Any, host: Any, yes: bool) -> None:
    from openternary.services import torch_setup as ts
    from openternary.services.errors import HardwareError
    from openternary.services.reporting import milestone, publish
    from openternary.ui.render import setup_view

    if not yes:
        if not sys.stdin.isatty() or not console.is_terminal:
            explain(t("setup.needs_yes"))
            raise ValueError("installing needs confirmation: rerun with --yes")
        if not typer.confirm(t("setup.confirm"), default=False):
            raise typer.Abort()
    milestone("setup.install", torch=plan.candidate.version, provider=plan.candidate.provider)
    code = ts.install(plan)
    if code != 0:
        explain(t("setup.install_failed"))
        raise RuntimeError(f"uv pip install exited with {code}")
    milestone("setup.verify", backend=plan.backend)
    verification = ts.verify(plan.backend)
    path = ts.write_profile(plan, host, verification)
    publish(verification=verification, profile=str(path), installed_torch=ts.installed_torch())
    setup_view(console, {"verification": verification, "profile": str(path)}, None, limit=0)
    if verification.get("status") != "passed":
        explain(t("setup.verify_failed"))
        raise HardwareError(str(verification.get("reason") or "runtime probe failed on the installed build"))


def _providers_for(backend: str, os_name: str) -> set[str]:
    if backend == "rocm":
        return {"amd"} if os_name == "windows" else {"amd", "pytorch"}
    if backend == "cuda":
        return {"pytorch", "pypi"} if os_name == "linux" else {"pytorch"}
    if backend == "mps":
        return {"pypi"}
    return {"pytorch", "pypi"}
