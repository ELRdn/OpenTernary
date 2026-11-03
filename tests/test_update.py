"""`openternary update`: PyPI checks, cache, install modes, plans, restore, and no network in tests."""

from __future__ import annotations

import io
import json
import tomllib
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import openternary
from openternary.cli.main import app
from openternary.services import update as upd

ROOT = Path(__file__).resolve().parents[1]


def releases(*versions: str, yanked: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"releases": {v: [{"filename": f"openternary-{v}.whl", "yanked": v in yanked}] for v in versions}}


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "update-check.json"
    monkeypatch.setattr(upd, "cache_path", lambda: path)
    monkeypatch.setattr(upd, "current_version", lambda: "0.1.0a1")
    return path


def test_version_is_declared_once_in_each_place_and_they_agree() -> None:
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert declared == openternary.__version__ == "0.1.0a1"


def test_latest_release_selection() -> None:
    payload = releases("0.1.0a1", "0.1.0a2", "0.1.0", "0.2.0", yanked=("0.2.0",))
    assert upd.latest_from_pypi_json(payload, include_pre=False) == "0.1.0"
    assert upd.latest_from_pypi_json(releases("0.1.0a1", "0.1.0a2"), include_pre=False) == "0.1.0a2"
    assert upd.latest_from_pypi_json(releases("0.1.0", "0.2.0a1"), include_pre=True) == "0.2.0a1"
    assert upd.latest_from_pypi_json({}, include_pre=True) is None


def test_check_caches_for_a_day_and_force_refreshes(isolated_cache: Path) -> None:
    calls: list[int] = []

    def fetch() -> dict[str, Any]:
        calls.append(1)
        return releases("0.1.0a1", "0.1.0a2")

    first = upd.check(fetch=fetch)
    assert first.update_available and first.latest == "0.1.0a2"
    assert upd.check(fetch=fetch).latest == "0.1.0a2" and len(calls) == 1  # served from cache
    upd.check(force=True, fetch=fetch)
    assert len(calls) == 2
    assert json.loads(isolated_cache.read_text())["latest"] == "0.1.0a2"


def test_not_yet_published_is_not_an_error() -> None:
    def fetch() -> dict[str, Any]:
        raise urllib.error.HTTPError(upd.PYPI_JSON, 404, "Not Found", None, io.BytesIO())  # type: ignore[arg-type]

    status = upd.check(force=True, fetch=fetch)
    assert status.error is None and status.published is False and not status.update_available


def test_network_failure_is_reported_and_not_cached(isolated_cache: Path) -> None:
    def fetch() -> dict[str, Any]:
        raise OSError("offline")

    status = upd.check(force=True, fetch=fetch)
    assert status.error == "offline" and not isolated_cache.exists()


def test_cached_notice_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    upd.check(force=True, fetch=lambda: releases("0.1.0a1", "0.1.0a2"))
    monkeypatch.setattr(upd, "fetch_pypi_json", lambda *a, **k: pytest.fail("network used"))
    notice = upd.cached_notice()
    assert notice is not None and notice.update_available
    monkeypatch.setenv("OPENTERNARY_NO_UPDATE_CHECK", "1")
    assert upd.cached_notice() is None


class Dist:
    def __init__(self, direct_url: str | None) -> None:
        self.direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        return self.direct_url if name == "direct_url.json" else None


def test_install_mode_detection() -> None:
    editable = Dist(json.dumps({"url": "file:///src/OpenTernary", "dir_info": {"editable": True}}))
    assert upd.install_mode("/x/uv/tools/openternary", editable) == "editable"
    assert upd.install_mode("C:/Users/h/AppData/Roaming/uv/tools/openternary", Dist(None)) == "uv-tool"
    assert upd.install_mode("/home/h/.local/share/uv/tools/openternary", Dist(None)) == "uv-tool"
    assert upd.install_mode("/home/h/project/.venv", Dist(None)) == "venv"


def test_editable_install_reports_the_source_version_not_stale_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # the autouse fixture pins current_version; test the real one
    monkeypatch.setattr(upd.metadata, "version", lambda name: "0.1.0a0")  # installed before a pull
    monkeypatch.setattr(upd, "install_mode", lambda *a, **k: "editable")
    assert upd.current_version() == openternary.__version__
    monkeypatch.setattr(upd, "install_mode", lambda *a, **k: "uv-tool")
    assert upd.current_version() == "0.1.0a0"


def test_upgrade_plans_per_install_mode() -> None:
    tool = upd.plan_upgrade("uv-tool", has_profile=True, uv="uv", platform="linux")
    assert tool.commands == [["uv", "tool", "upgrade", "openternary"]] and tool.restore and tool.automatic
    windows = upd.plan_upgrade("uv-tool", has_profile=True, uv="uv", platform="win32")
    assert not windows.automatic and windows.reason == "windows_running_tool"
    venv = upd.plan_upgrade("venv", has_profile=False, uv="uv", platform="win32")
    assert venv.commands[0][:3] == ["uv", "pip", "install"] and venv.automatic and not venv.restore
    editable = upd.plan_upgrade("editable", has_profile=True, uv="uv")
    assert editable.commands == [["git", "pull", "--ff-only"]] and not editable.automatic
    assert upd.plan_upgrade("venv", False, uv=None).reason == "no_uv"


# --- CLI ----------------------------------------------------------------------------------------
@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    from openternary.cli import update as cli_update
    from openternary.services import torch_setup as ts

    state: dict[str, Any] = {"ran": [], "latest": "0.1.0a2", "mode": "uv-tool", "platform": "linux"}
    monkeypatch.setattr(upd, "fetch_pypi_json", lambda *a, **k: releases("0.1.0a1", state["latest"]))
    monkeypatch.setattr(upd, "install_mode", lambda *a, **k: state["mode"])
    monkeypatch.setattr(upd, "find_uv", lambda: "uv")
    real_plan = upd.plan_upgrade
    monkeypatch.setattr(upd, "plan_upgrade", lambda m, p, u: real_plan(m, p, u, platform=state["platform"]))
    monkeypatch.setattr(ts, "profile_path", lambda: tmp_path / ts.PROFILE)
    monkeypatch.setattr(cli_update, "_run", lambda command: state["ran"].append(command) or 0)
    return state


def envelope(result: Any) -> dict[str, Any]:
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_cli_check_only_reports_without_running_anything(cli_env: dict[str, Any]) -> None:
    env = envelope(CliRunner().invoke(app, ["update", "--check", "--json"]))
    assert env["exit_code"] == 0 and env["result"]["status"]["latest"] == "0.1.0a2"
    assert cli_env["ran"] == []


def test_cli_upgrade_then_restores_the_recorded_build(cli_env: dict[str, Any], tmp_path: Path) -> None:
    from openternary.services import torch_setup as ts

    (tmp_path / ts.PROFILE).write_text(json.dumps({"version": "2.14.0+rocm10.1.0", "backend": "rocm"}))
    env = envelope(CliRunner().invoke(app, ["update", "--yes", "--json"]))
    assert env["exit_code"] == 0
    upgrade, restore = cli_env["ran"]
    assert upgrade == ["uv", "tool", "upgrade", "openternary"]
    assert restore[-5:-2] == ["setup", "--restore", "--profile"] and restore[-1] == "--yes"
    saved = json.loads(Path(restore[-2]).read_text())
    assert saved["version"] == "2.14.0+rocm10.1.0"  # survives a rebuilt tool environment


def test_cli_windows_tool_install_prints_steps_instead_of_running(cli_env: dict[str, Any]) -> None:
    cli_env["platform"] = "win32"
    env = envelope(CliRunner().invoke(app, ["update", "--yes", "--json"]))
    assert env["exit_code"] == 0 and cli_env["ran"] == []
    assert env["result"]["plan"]["reason"] == "windows_running_tool"


def test_cli_needs_confirmation_outside_a_terminal(cli_env: dict[str, Any]) -> None:
    env = envelope(CliRunner().invoke(app, ["update", "--json"]))
    assert env["exit_code"] == 2 and cli_env["ran"] == []


def test_cli_up_to_date(cli_env: dict[str, Any]) -> None:
    cli_env["latest"] = "0.1.0a1"
    env = envelope(CliRunner().invoke(app, ["update", "--yes", "--json"]))
    assert env["exit_code"] == 0 and env["result"]["status"]["update_available"] is False
    assert cli_env["ran"] == []


def test_doctor_outside_a_terminal_never_checks_pypi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upd, "fetch_pypi_json", lambda *a, **k: pytest.fail("doctor used the network"))
    assert CliRunner().invoke(app, ["doctor"]).exit_code == 0
