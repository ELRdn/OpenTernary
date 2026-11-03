"""Choose, install and verify a PyTorch build for the running environment (`openternary setup`).

Sources are queried live (no hard-coded version matrix) because CUDA / ROCm / AMD builds change
every few weeks. Only wheels whose tags match *this* interpreter and platform are offered.
Installation uses ``uv pip install`` into ``sys.executable``'s environment; the choice and its
runtime verification are recorded next to that environment so ``doctor`` and ``update`` can see
and restore it.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from openternary.services.hardware import Host

PYPI = "https://pypi.org/simple"
PROVIDERS = {
    # name: (index used for installing, listing root; "{package}" is appended for listings)
    "pypi": (PYPI, PYPI),
    "pytorch": ("https://download.pytorch.org/whl/{variant}", "https://download.pytorch.org/whl"),
    "amd": ("https://stable.repo.amd.com/rocm/whl-next", "https://stable.repo.amd.com/rocm/whl-next"),
}
PROVIDER_PRIORITY = {"amd": 0, "pytorch": 1, "pypi": 2}
BACKENDS = ("auto", "cuda", "rocm", "mps", "cpu")
PROFILE = "openternary-torch.json"
Fetch = Callable[[str], tuple[str, str]]  # url -> (content type, body)


def provider_urls(name: str) -> tuple[str, str]:
    install, listing = PROVIDERS[name]
    override = os.environ.get(f"OPENTERNARY_INDEX_{name.upper()}")
    return (override, override) if override else (install, listing)


# --- index listing ------------------------------------------------------------------------------
def http_get(url: str, timeout: float = 30.0) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.pypi.simple.v1+json, text/html;q=0.5", "User-Agent": "openternary-setup"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https index URLs only
        return response.headers.get("Content-Type", ""), response.read().decode("utf-8", "replace")


def parse_listing(content_type: str, body: str) -> list[str]:
    """Wheel filenames from a PEP 691 JSON or PEP 503 HTML simple page."""
    if "json" in content_type:
        try:
            files = json.loads(body).get("files", [])
            return [f["filename"] for f in files if str(f.get("filename", "")).endswith(".whl")]
        except (json.JSONDecodeError, AttributeError):
            return []
    names = []
    for href, text in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]*)</a>', body, re.IGNORECASE):
        candidate = text.strip() or urllib.parse.unquote(href.split("#")[0].rsplit("/", 1)[-1])
        candidate = urllib.parse.unquote(candidate)
        if candidate.endswith(".whl"):
            names.append(candidate)
    return names


def list_wheels(provider: str, package: str, fetch: Fetch = http_get) -> list[str]:
    _, listing = provider_urls(provider)
    content_type, body = fetch(f"{listing.rstrip('/')}/{package}/")
    return parse_listing(content_type, body)


# --- candidates ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    provider: str
    backend: str  # cuda | rocm | mps | cpu
    version: str  # full version, local tag included (e.g. 2.13.0+rocm10.0.0)
    variant: str  # cu130 | rocm10.0.0 | cpu | pypi
    index_url: str
    torchvision: str | None = None  # matching torchvision from the same source, if published

    @property
    def base(self) -> Version:
        return Version(Version(self.version).base_version)


def _variant_rank(variant: str) -> tuple[int, ...]:
    digits = re.findall(r"\d+", variant)
    return tuple(int(d) for d in digits) or (0,)


def cuda_version(variant: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"cu(\d+)(\d)", variant)
    return (int(match.group(1)), int(match.group(2))) if match else None


def classify(provider: str, version: Version, host: Host) -> tuple[str, str] | None:
    local = version.local or ""
    if provider == "amd":
        return "rocm", local or "rocm"
    if provider == "pytorch":
        if local == "cpu":
            return "cpu", "cpu"
        if local.startswith("cu"):
            return "cuda", local
        if local.startswith("rocm"):
            return "rocm", local
        return None  # xpu and others are not offered yet
    if local:
        return None
    if host.os == "macos":
        return ("mps" if host.machine == "arm64" else "cpu"), "pypi"
    return ("cuda" if host.os == "linux" else "cpu"), "pypi"


@dataclass
class Offer:
    candidates: list[Candidate] = field(default_factory=list)
    other_pythons: dict[str, list[str]] = field(default_factory=dict)  # backend -> python versions with wheels
    errors: dict[str, str] = field(default_factory=dict)  # provider -> failure


def build_offer(
    host: Host,
    listings: dict[str, list[str]],
    supported: set[Tag] | None = None,
    pre: bool = False,
    minimum: Version | None = None,
) -> Offer:
    supported = supported if supported is not None else set(sys_tags())
    platforms = {tag.platform for tag in supported}
    offer = Offer()
    seen: set[tuple[str, str]] = set()
    others: dict[str, set[str]] = {}
    for provider, filenames in listings.items():
        install_url, _ = provider_urls(provider)
        for filename in filenames:
            try:
                name, version, _, tags = parse_wheel_filename(filename)
            except (InvalidWheelFilename, InvalidVersion):
                continue
            if name != canonicalize_name("torch") or (version.is_prerelease and not pre):
                continue
            if minimum is not None and Version(version.base_version) < minimum:
                continue
            kind = classify(provider, version, host)
            if kind is None:
                continue
            backend, variant = kind
            if not tags & supported:
                for tag in tags:
                    if tag.platform in platforms and re.fullmatch(r"cp3\d+", tag.interpreter):
                        others.setdefault(backend, set()).add(f"3.{tag.interpreter[3:]}")
                continue
            key = (provider, str(version))
            if key in seen:
                continue
            seen.add(key)
            offer.candidates.append(
                Candidate(provider, backend, str(version), variant, install_url.format(variant=variant))
            )
    offer.other_pythons = {k: sorted(v, key=lambda s: tuple(map(int, s.split(".")))) for k, v in others.items()}
    return offer


def wanted_backend(host: Host, requested: str) -> str:
    if requested != "auto":
        return requested
    if host.first("nvidia"):
        return "cuda"
    if host.first("amd"):
        return "rocm"
    if host.first("apple"):
        return "mps"
    return "cpu"


def rank(host: Host, candidates: list[Candidate], backend: str) -> tuple[list[Candidate], list[str]]:
    """Candidates for `backend`, best first, plus warnings about the choice."""
    warnings: list[str] = []
    pool = [c for c in candidates if c.backend == backend]
    if backend == "cpu" and host.os == "linux":
        # PyPI's default Linux wheels bundle CUDA but run on CPU too: a larger fallback.
        pool += [c for c in candidates if c.provider == "pypi"]
    if backend == "cuda":
        gpu = host.first("nvidia")
        if gpu and gpu.cuda_driver:
            limit = tuple(int(x) for x in gpu.cuda_driver.split("."))
            fitting = [c for c in pool if (cuda_version(c.variant) or (0, 0)) <= limit or c.variant == "pypi"]
            if fitting:
                pool = fitting
            elif pool:
                warnings.append("cuda_driver_too_old")
        elif pool:
            warnings.append("cuda_driver_unknown")
    pool.sort(
        key=lambda c: (c.torchvision is not None, c.base, _variant_rank(c.variant), -PROVIDER_PRIORITY[c.provider]),
        reverse=True,
    )
    if backend == "cpu" and host.os == "linux":
        pool.sort(key=lambda c: c.variant != "cpu")  # stable: the CPU index avoids ~2 GB of CUDA libraries
    if backend == "rocm" and host.wsl:
        warnings.append("rocm_on_wsl")
    return pool, warnings


def with_torchvision(
    candidates: list[Candidate], vision: dict[str, list[str]], supported: set[Tag] | None = None
) -> list[Candidate]:
    """Attach the matching torchvision (same source, same local variant, 0.(minor+15)) to each torch."""
    return [
        replace(c, torchvision=torchvision_for(c, vision.get(c.provider, []), supported)) if c.provider in vision else c
        for c in candidates
    ]


def select(pool: list[Candidate], torch_version: str | None, variant: str | None) -> list[Candidate]:
    picked = pool
    if variant:
        picked = [c for c in picked if c.variant == variant]
    if torch_version:
        picked = [c for c in picked if c.version == torch_version or str(c.base) == torch_version]
    return picked


# --- companion packages and the install plan ----------------------------------------------------
def torchvision_for(candidate: Candidate, filenames: list[str], supported: set[Tag] | None = None) -> str | None:
    """torchvision built for the same torch release and the same local variant (0.(minor+15))."""
    supported = supported if supported is not None else set(sys_tags())
    base = candidate.base
    wanted_minor = base.minor + 15 if base.major == 2 else None
    local = Version(candidate.version).local
    best: Version | None = None
    for filename in filenames:
        try:
            name, version, _, tags = parse_wheel_filename(filename)
        except (InvalidWheelFilename, InvalidVersion):
            continue
        if name != "torchvision" or not tags & supported or version.local != local:
            continue
        if wanted_minor is not None and (version.major, version.minor) != (0, wanted_minor):
            continue
        if best is None or version > best:
            best = version
    return str(best) if best else None


def openternary_requirements() -> tuple[list[str], Version | None]:
    """ML/calibration requirements of the installed openternary, minus torch/torchvision."""
    try:
        declared = metadata.requires("openternary") or []
    except metadata.PackageNotFoundError:
        declared = []
    extras, minimum = [], None
    for line in declared:
        requirement = Requirement(line)
        marker = requirement.marker
        if marker is None or not any(marker.evaluate({"extra": e}) for e in ("ml", "calibration")):
            continue
        if requirement.name == "torch":
            for spec in requirement.specifier:
                if spec.operator == ">=":
                    minimum = Version(spec.version)
            continue
        if requirement.name == "torchvision":
            continue
        requirement.marker = None
        extras.append(str(requirement))
    return sorted(set(extras)), minimum


@dataclass
class Plan:
    candidate: Candidate
    backend: str  # what the user asked for; decides the device check (a CUDA-bundled wheel can serve "cpu")
    torchvision: str | None
    python: str
    command: list[str]
    requirements: list[str]
    amd_device: str | None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["candidate"] = asdict(self.candidate)
        return out


def find_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise FileNotFoundError("uv was not found on PATH; install it from https://docs.astral.sh/uv/")
    return uv


def make_plan(
    candidate: Candidate,
    requirements: list[str],
    uv: str,
    python: str | None = None,
    amd_device: str = "all",
    backend: str | None = None,
) -> Plan:
    python = python or sys.executable
    extras = f"[device-{amd_device}]" if candidate.provider == "amd" else ""
    command = [uv, "pip", "install", "--python", python, "--index-url", PYPI]
    if candidate.provider != "pypi":
        # uv tries --extra-index-url first, so torch/torchvision come from the chosen source and
        # everything else (numpy, transformers, ...) from PyPI.
        command += ["--extra-index-url", candidate.index_url]
    torchvision = candidate.torchvision
    command += ["--reinstall-package", "torch"]
    if torchvision:
        command += ["--reinstall-package", "torchvision"]
    command += [f"torch{extras}=={candidate.version}"]
    if torchvision:
        command += [f"torchvision=={torchvision}"]
    command += requirements
    return Plan(
        candidate,
        backend or candidate.backend,
        torchvision,
        python,
        command,
        requirements,
        amd_device if extras else None,
    )


def plan_from_profile(profile: dict[str, Any], requirements: list[str], uv: str) -> Plan:
    """Rebuild the exact recorded installation (used by `setup --restore` and `update`)."""
    try:
        candidate = Candidate(
            provider=str(profile["provider"]),
            backend=str(profile.get("build") or profile["backend"]),
            version=str(profile["version"]),
            variant=str(profile["variant"]),
            index_url=str(profile["index_url"]),
            torchvision=profile.get("torchvision"),
        )
    except KeyError as exc:
        raise ValueError(f"recorded setup profile is incomplete (missing {exc})") from exc
    if candidate.provider not in PROVIDERS:
        raise ValueError(f"recorded setup profile names an unknown source: {candidate.provider}")
    return make_plan(
        candidate, requirements, uv, amd_device=profile.get("amd_device") or "all", backend=str(profile["backend"])
    )


def install(plan: Plan) -> int:
    """Run the plan, streaming uv's own progress to stderr (stdout stays clean for --json)."""
    try:
        sink: Any = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        sink = None
    done = subprocess.run(plan.command, stdout=sink, stderr=sink, check=False)
    return done.returncode


PROBE = (
    "import json; from openternary.services.runtime_probe import probe_runtime as p; "
    "print(json.dumps(p({device!r}), default=str))"
)


def probe_device(backend: str) -> str:
    return {"cuda": "cuda", "rocm": "cuda", "mps": "mps"}.get(backend, "cpu")


def verify(backend: str, python: str | None = None, timeout: float = 600.0) -> dict[str, Any]:
    """Probe the freshly installed torch in a new process (this one may hold the old import)."""
    device = probe_device(backend)
    try:
        done = subprocess.run(
            [python or sys.executable, "-c", PROBE.format(device=device)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "device": device, "reason": f"probe timed out after {timeout:.0f}s"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return {"status": "failed", "device": device, "reason": " | ".join(tail) or f"exit {done.returncode}"}
    result: dict[str, Any] = json.loads(lines[-1])
    return result


# --- profile ------------------------------------------------------------------------------------
def profile_path() -> Path:
    return Path(sys.prefix) / PROFILE


def read_profile(path: Path | None = None) -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads((path or profile_path()).read_text(encoding="utf-8"))
        return data
    except (OSError, json.JSONDecodeError):
        return None


def write_profile(plan: Plan, host: Host, verification: dict[str, Any], path: Path | None = None) -> Path:
    path = path or profile_path()
    previous = read_profile(path) or {}
    entry = {
        "torch": plan.candidate.version,
        "backend": plan.backend,
        "status": verification.get("status"),
        "device_name": verification.get("device_name"),
        "checked_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    data = {
        "schema_version": 1,
        **asdict(plan.candidate),
        "backend": plan.backend,
        "build": plan.candidate.backend,
        "torchvision": plan.torchvision,
        "amd_device": plan.amd_device,
        "requirements": plan.requirements,
        "verification": verification,
        "host": host.to_dict(),
        "history": [*previous.get("history", [])[-19:], entry],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def installed_torch() -> str | None:
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


def verified_here(profile: dict[str, Any] | None) -> set[str]:
    if not profile:
        return set()
    return {h["torch"] for h in profile.get("history", []) if h.get("status") == "passed"}
