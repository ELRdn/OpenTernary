"""The Textual TUI: forms derived from the CLI, navigation, typed deletion, plan screen, launch rules."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button, DataTable, Input, TabbedContent

from openternary.services import jobs as jobsvc
from openternary.ui.app import catalog
from openternary.ui.app.app import OpenTernaryApp, short
from openternary.ui.app.dialogs import Choice, JobForm, PlanScreen, TypedConfirm

ROOT = Path(__file__).resolve().parents[1]


# --- catalog ------------------------------------------------------------------------------------
def test_every_cli_command_has_a_form_except_the_tui_itself() -> None:
    names = {spec.name for spec in catalog.specs()}
    for expected in ("inspect", "quantize", "calibrate", "compare", "doctor", "setup", "update", "cache remove"):
        assert expected in names
    assert "ui" not in names
    for spec in catalog.specs():
        hidden = {"json_output", "quiet", "events_jsonl", "dry_run", "yes"}
        assert not hidden & {f.name for f in spec.fields}, spec.name  # the TUI supplies these itself


def test_forms_know_dry_run_and_confirmation() -> None:
    assert catalog.spec("quantize").dry_run and not catalog.spec("quantize").confirm
    assert catalog.spec("setup").dry_run and catalog.spec("setup").confirm
    assert not catalog.spec("doctor").dry_run


def test_build_args() -> None:
    compare = catalog.spec("compare")
    args = catalog.build_args(
        compare, {"baseline": "runs/a", "quantized": "runs/b", "candidate": "runs/c 'runs/d e'", "verbose": True}
    )
    assert args[0] == "compare" and args[-2:] == ["runs/a", "runs/b"]
    assert args.count("--candidate") == 2 and "runs/d e" in args and "--verbose" in args
    remove = catalog.spec("cache remove")
    with pytest.raises(ValueError, match="required"):
        catalog.build_args(remove, {"target": "", "root": "cache"})
    values = {"target": "cache/model-a", "root": "cache", "execute": True}
    assert catalog.build_args(remove, values) == ["cache", "remove", "--root", "cache", "--execute", "cache/model-a"]
    assert catalog.destructive_target(remove, values) == "model-a"
    assert catalog.destructive_target(remove, {**values, "execute": False}) is None


def test_long_run_names_keep_both_ends() -> None:
    name = "2026-11-03_gemma-4-E2B-it-qat-q4_0-unquantized-naive-g128-009"
    assert short(name).startswith("2026-11-03_gemma") and short(name).endswith("g128-009") and len(short(name)) == 44
    assert short("short") == "short"


# --- app ----------------------------------------------------------------------------------------
FAKE = r"""
import json, os, signal, sys, time
if os.environ.get("OPENTERNARY_BREAK_IS_INTERRUPT") and hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)
open(sys.argv[1], "x").write(json.dumps({"stage": "started", "elapsed_s": 0}) + "\n")
try:
    time.sleep(float(sys.argv[2]))
    print(json.dumps({"exit_code": 0, "result": {}, "error": None}))
except KeyboardInterrupt:
    print(json.dumps({"exit_code": 130, "result": {}, "error": "KeyboardInterrupt"})); sys.exit(130)
"""


def make_run(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    return path


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENTERNARY_LANG", "en")
    script = tmp_path / "fake.py"
    script.write_text(FAKE, encoding="utf-8")
    seconds = {"value": 30.0}
    manager = jobsvc.JobManager(
        tmp_path / "jobs",
        command=lambda args, events: [sys.executable, str(script), str(events), str(seconds["value"])],
    )

    def factory() -> OpenTernaryApp:
        return OpenTernaryApp(runs_root=tmp_path / "runs", manager=manager, check_updates=False, animate=False)

    factory.manager = manager  # type: ignore[attr-defined]
    factory.seconds = seconds  # type: ignore[attr-defined]
    yield factory
    for job in manager.running():
        manager.cancel(job, force=True)


def run(app: OpenTernaryApp, scenario: Any) -> None:
    async def main() -> None:
        async with app.run_test(size=(130, 40)) as pilot:
            await pilot.pause(0.5)
            await scenario(app, pilot)

    asyncio.run(main())


def test_number_keys_switch_tabs_and_focus_the_list(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        tabs = app.query_one("#tabs", TabbedContent)
        assert tabs.active == "home"
        for key, name in zip("234561", ("runs", "jobs", "compare", "doctor", "update", "home"), strict=True):
            await pilot.press(key)
            await pilot.pause(0.1)
            assert tabs.active == name
        await pilot.press("2")
        await pilot.pause(0.1)
        assert app.focused is app.query_one("#runs-table")

    run(app_factory(), scenario)


def test_deleting_a_run_needs_its_exact_name(app_factory: Any, tmp_path: Path) -> None:
    run_dir = make_run(tmp_path / "runs", "2026-11-03_demo")

    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        await pilot.press("2")
        await pilot.pause(0.5)
        assert app.query_one("#runs-table", DataTable).row_count == 1
        await pilot.press("x")
        await pilot.pause(0.2)
        assert isinstance(app.screen, TypedConfirm)
        delete = app.screen.query_one("#delete", Button)
        app.screen.query_one("#typed", Input).value = "2026-11-03"
        await pilot.pause(0.1)
        assert delete.disabled
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert run_dir.exists()  # backing out deletes nothing
        await pilot.press("x")
        await pilot.pause(0.2)
        app.screen.query_one("#typed", Input).value = "2026-11-03_demo"
        await pilot.pause(0.1)
        await pilot.click("#delete")
        await pilot.pause(0.5)
        assert not run_dir.exists()

    run(app_factory(), scenario)


def test_form_plan_and_start_launch_a_job(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        await pilot.press("n")
        await pilot.pause(0.5)
        assert isinstance(app.screen, JobForm)
        app.screen.query_one("#f-config", Input).value = "configs/gemma4-e2b.yaml"
        await pilot.click("#review")
        await pilot.pause(0.3)
        assert isinstance(app.screen, PlanScreen) and app.screen.args == [
            "inspect",
            "--config",
            "configs/gemma4-e2b.yaml",
        ]
        await pilot.click("#start")  # inspect has no --dry-run, so Start is available at once
        await pilot.pause(0.5)
        assert app.query_one("#tabs", TabbedContent).active == "jobs"
        (job,) = app_factory.manager.jobs()
        assert job.args == ["inspect", "--config", "configs/gemma4-e2b.yaml"] and job.status == "running"

    run(app_factory(), scenario)


def test_start_stays_disabled_when_the_dry_run_fails(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        screen = PlanScreen(catalog.spec("quantize"), ["quantize", "--config", "configs/missing.yaml"])
        app.push_screen(screen)
        await pilot.pause(0.2)
        assert screen.query_one("#start", Button).disabled  # until the preview has answered
        for _ in range(100):
            await pilot.pause(0.1)
            if "exit" in str(screen.query_one("#preview").render()):
                break
        assert screen.query_one("#start", Button).disabled

    run(app_factory(), scenario)


def test_quitting_with_a_running_job_asks_first(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        job = app.start_job(["calibrate"])
        await pilot.pause(0.3)
        await pilot.press("q")
        await pilot.pause(0.2)
        assert isinstance(app.screen, Choice)
        await pilot.click("#back")
        await pilot.pause(0.2)
        assert app.is_running and app_factory.manager.refresh(job).status == "running"

    run(app_factory(), scenario)


# --- launch rules -------------------------------------------------------------------------------
def cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "OPENTERNARY_LANG": "en", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "openternary", *args], cwd=ROOT, env=env, capture_output=True)


def test_without_a_terminal_no_arguments_still_prints_help() -> None:
    result = cli()
    assert result.returncode == 2 and b"Usage" in result.stdout  # unchanged CLI behaviour


def test_ui_command_refuses_without_a_terminal() -> None:
    result = cli("ui")
    assert result.returncode == 2 and b"interactive terminal" in result.stderr


def test_machine_paths_never_import_textual() -> None:
    env = {**os.environ, "OPENTERNARY_LANG": "en"}
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "openternary", "doctor", "--json"],
        cwd=ROOT,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0
    assert b" textual" not in result.stderr  # -X importtime lists every imported module on stderr
