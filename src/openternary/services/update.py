"""Release checks and upgrades for `openternary update` (PyPI is the source of truth).

The check uses the network only to read PyPI's JSON for the package and is cached for a day.
Upgrading depends on how OpenTernary was installed:

- ``uv-tool``  : `uv tool upgrade openternary`, then restore the PyTorch build that
  `openternary setup` recorded (the tool environment may be rebuilt by the upgrade)
- ``venv``     : `uv pip install --upgrade openternary` into this interpreter, then restore
- ``editable`` : a source checkout; never touched automatically (git is the source of truth)
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

PACKAGE = "openternary"
PYPI_JSON = "https://pypi.org/pypi/openternary/json"
CACHE_TTL = datetime.timedelta(hours=24)


def current_version() -> str:
    """The version of the code that is running.

    An editable install's metadata is frozen when it was installed, so a `git pull` that bumps the
    version leaves it stale; there the source's `__version__` is the truth.
    """
    from openternary import __version__

    try:
        installed = metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return __version__
    return __version__ if install_mode() == "editable" else installed


def latest_from_pypi_json(payload: dict[str, Any], include_pre: bool) -> str | None:
    """Newest non-yanked release, pre-releases only when asked (or when nothing else exists)."""
    versions: list[Version] = []
    for raw, files in (payload.get("releases") or {}).items():
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if files and all(f.get("yanked") for f in files):
            continue
        versions.append(version)
    stable = [v for v in versions if not v.is_prerelease]
    pool = versions if include_pre or not stable else stable
    return str(max(pool)) if pool else None


def fetch_pypi_json(url: str = PYPI_JSON, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "openternary-update"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        return data


def cache_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "openternary" / "update-check.json"


@dataclass
class Status:
    current: str
    latest: str | None
    update_available: bool
    checked_at: str
    source: str = "pypi"
    error: str | None = None
    published: bool = True  # False while the project has no release on PyPI yet


def check(force: bool = False, include_pre: bool | None = None, fetch: Any = None) -> Status:
    """Compare the installed version with PyPI, using a one-day cache unless `force`."""
    fetch = fetch or fetch_pypi_json  # resolved at call time so it can be replaced (tests, mirrors)
    current = current_version()
    include_pre = Version(current).is_prerelease if include_pre is None else include_pre
    path = cache_path()
    now = datetime.datetime.now(datetime.UTC)
    if not force:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (
                cached.get("current") == current
                and now - datetime.datetime.fromisoformat(cached["checked_at"]) < CACHE_TTL
            ):
                return Status(**{k: cached.get(k) for k in Status.__dataclass_fields__})
        except (OSError, ValueError, KeyError, TypeError):
            pass
    published = True
    try:
        latest = latest_from_pypi_json(fetch(), include_pre)
        error = None
    except urllib.error.HTTPError as exc:
        latest, error = None, (None if exc.code == 404 else str(exc))
        published = exc.code != 404
    except (OSError, ValueError) as exc:
        latest, error = None, str(exc)
    status = Status(
        current=current,
        latest=latest,
        update_available=bool(latest and Version(latest) > Version(current)),
        checked_at=now.isoformat(timespec="seconds"),
        error=error,
        published=published,
    )
    if error is None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
        except OSError:
            pass
    return status


def cached_notice() -> Status | None:
    """Cached result only (no network); None when unknown or disabled."""
    if os.environ.get("OPENTERNARY_NO_UPDATE_CHECK"):
        return None
    try:
        cached = json.loads(cache_path().read_text(encoding="utf-8"))
        status = Status(**{k: cached.get(k) for k in Status.__dataclass_fields__})
    except (OSError, ValueError, TypeError):
        return None
    if status.current != current_version():
        return None
    return status


# --- install mode -----------------------------------------------------------------------------
def install_mode(prefix: str | None = None, dist: Any = None) -> str:
    try:
        distribution = dist or metadata.distribution(PACKAGE)
        direct = distribution.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct = None
    if direct:
        try:
            if json.loads(direct).get("dir_info", {}).get("editable"):
                return "editable"
        except json.JSONDecodeError:
            pass
    parts = [p.lower() for p in Path(prefix or sys.prefix).parts]
    if "tools" in parts and parts[-1] == PACKAGE and "uv" in parts:
        return "uv-tool"
    return "venv"


@dataclass
class UpgradePlan:
    mode: str
    commands: list[list[str]] = field(default_factory=list)
    restore: bool = False
    automatic: bool = True
    reason: str | None = None


def plan_upgrade(mode: str, has_profile: bool, uv: str | None, platform: str = sys.platform) -> UpgradePlan:
    if mode == "editable":
        return UpgradePlan(
            mode, [["git", "pull", "--ff-only"]], restore=has_profile, automatic=False, reason="editable"
        )
    if not uv:
        return UpgradePlan(mode, [], automatic=False, reason="no_uv")
    if mode == "uv-tool":
        commands = [[uv, "tool", "upgrade", PACKAGE]]
    else:
        commands = [[uv, "pip", "install", "--python", sys.executable, "--upgrade", PACKAGE]]
    restore = has_profile
    # A running Windows executable cannot be replaced; let the user run the commands themselves.
    automatic = not (platform == "win32" and mode == "uv-tool")
    return UpgradePlan(mode, commands, restore, automatic, None if automatic else "windows_running_tool")


def save_profile_copy(profile: dict[str, Any] | None) -> Path | None:
    """Keep the setup record outside the environment, which an upgrade may rebuild."""
    if profile is None:
        return None
    handle, name = tempfile.mkstemp(prefix="openternary-torch-", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(profile, stream)
    return Path(name)


def find_uv() -> str | None:
    return shutil.which("uv")
